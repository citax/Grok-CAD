#include "cadcore/mesh/csg.hpp"

#include "cadcore/math/constants.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

namespace cad {
namespace {

// ---------------------------------------------------------------------------
// Precision policy (deterministic)
// ---------------------------------------------------------------------------
// kClassifyEps  — plane classification band (absolute model units)
// kSnapEps      — quantize vertices so independently computed cut points coincide
// kOnEdgeEps    — point-on-segment tolerance for T-junction detection
// kAreaEps      — discard near-degenerate triangles

// Build: generous coplanar band so tessellated convex solids do not shred.
// Clip: tighter so inter-solid cuts stay accurate.
constexpr double kCoplanarEps = 1e-5;
constexpr double kClipEps     = 1e-6;
constexpr double kSnapEps     = 1e-5;   // quantize for deterministic welding
constexpr double kOnEdgeEps   = 1e-6;   // absolute floor for point-on-edge
constexpr double kOnEdgeRel   = 1e-5;   // relative to edge length
constexpr double kAreaEps     = 1e-14;
constexpr double kWeldEps     = 1e-5;   // final vertex weld radius
static_assert(kWeldEps > 0.0);
static_assert(kClipEps > 0.0);

[[nodiscard]] double snap_scalar(double x) noexcept {
  return std::round(x / kSnapEps) * kSnapEps;
}

[[nodiscard]] Vec3 snap_vec(const Vec3& v) noexcept {
  return {snap_scalar(v.x), snap_scalar(v.y), snap_scalar(v.z)};
}

// Deterministic key for spatial hashing after snap
struct VecKey {
  std::int64_t ix, iy, iz;
  bool operator==(const VecKey& o) const noexcept {
    return ix == o.ix && iy == o.iy && iz == o.iz;
  }
};

struct VecKeyHash {
  std::size_t operator()(const VecKey& k) const noexcept {
    // Splitmix-style combine — deterministic
    auto h = static_cast<std::uint64_t>(k.ix) * 0x9e3779b97f4a7c15ULL;
    h ^= static_cast<std::uint64_t>(k.iy) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
    h ^= static_cast<std::uint64_t>(k.iz) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
    return static_cast<std::size_t>(h);
  }
};

[[nodiscard]] VecKey make_key(const Vec3& v) noexcept {
  const Vec3 s = snap_vec(v);
  return {static_cast<std::int64_t>(std::llround(s.x / kSnapEps)),
          static_cast<std::int64_t>(std::llround(s.y / kSnapEps)),
          static_cast<std::int64_t>(std::llround(s.z / kSnapEps))};
}

// ---------------------------------------------------------------------------
// Plane / Polygon
// ---------------------------------------------------------------------------

struct Plane {
  Vec3 normal{0, 1, 0};
  double w = 0.0;  // normal · x = w

  static Plane from_points(const Vec3& a, const Vec3& b, const Vec3& c) {
    Plane p;
    p.normal = cross(b - a, c - a).normalized();
    if (p.normal.length_sq() < kAbsEps) {
      p.normal = {0, 1, 0};
    }
    const Vec3 mid = (a + b + c) * (1.0 / 3.0);
    p.w = dot(p.normal, mid);
    return p;
  }

  [[nodiscard]] double classify(const Vec3& v) const noexcept {
    return dot(normal, v) - w;
  }

  void flip() noexcept {
    normal = -normal;
    w = -w;
  }
};

struct Polygon {
  std::vector<Vec3> vertices;
  Plane plane;

  Polygon() = default;
  explicit Polygon(std::vector<Vec3> verts) : vertices(std::move(verts)) {
    for (auto& v : vertices) {
      v = snap_vec(v);
    }
    // Drop consecutive duplicates after snap
    vertices.erase(std::unique(vertices.begin(), vertices.end(),
                               [](const Vec3& a, const Vec3& b) {
                                 return make_key(a) == make_key(b);
                               }),
                   vertices.end());
    if (vertices.size() >= 2 && make_key(vertices.front()) == make_key(vertices.back())) {
      vertices.pop_back();
    }
    if (vertices.size() >= 3) {
      plane = Plane::from_points(vertices[0], vertices[1], vertices[2]);
    }
  }

  void flip() {
    plane.flip();
    std::reverse(vertices.begin(), vertices.end());
  }

  [[nodiscard]] bool valid() const {
    if (vertices.size() < 3) {
      return false;
    }
    Vec3 acc{0, 0, 0};
    for (std::size_t i = 1; i + 1 < vertices.size(); ++i) {
      acc += cross(vertices[i] - vertices[0], vertices[i + 1] - vertices[0]);
    }
    return 0.5 * acc.length() > kAreaEps;
  }

  /// Recompute plane from first non-degenerate triple; keep orientation if possible.
  void recompute_plane() {
    if (vertices.size() < 3) {
      return;
    }
    const Vec3 old_n = plane.normal;
    for (std::size_t i = 0; i + 2 < vertices.size(); ++i) {
      const Vec3 n = cross(vertices[i + 1] - vertices[i], vertices[i + 2] - vertices[i]);
      if (n.length_sq() > kAreaEps) {
        plane = Plane::from_points(vertices[i], vertices[i + 1], vertices[i + 2]);
        if (dot(plane.normal, old_n) < 0.0) {
          plane.flip();
        }
        return;
      }
    }
  }
};

enum class Side : int { Coplanar = 0, Front = 1, Back = 2, Spanning = 3 };

Side classify_vertex(double d, double eps) noexcept {
  if (d > eps) {
    return Side::Front;
  }
  if (d < -eps) {
    return Side::Back;
  }
  return Side::Coplanar;
}

Side classify_polygon(const Plane& plane, const Polygon& poly, double plane_eps) {
  const double eps = plane_eps;

  double max_d = 0.0;
  double min_d = 0.0;
  bool first = true;
  int mask = 0;
  for (const auto& v : poly.vertices) {
    const double d = plane.classify(v);
    if (first) {
      max_d = min_d = d;
      first = false;
    } else {
      max_d = std::max(max_d, d);
      min_d = std::min(min_d, d);
    }
    mask |= static_cast<int>(classify_vertex(d, eps));
  }

  // Tessellated convex solids: neighbouring facets sit slightly in front of a
  // face plane. Collapse thin slabs to coplanar so BSP does not shred itself.
  if (max_d <= eps && min_d >= -eps) {
    return Side::Coplanar;
  }
  if ((mask & static_cast<int>(Side::Front)) && (mask & static_cast<int>(Side::Back))) {
    return Side::Spanning;
  }
  if (mask & static_cast<int>(Side::Front)) {
    return Side::Front;
  }
  if (mask & static_cast<int>(Side::Back)) {
    return Side::Back;
  }
  return Side::Coplanar;
}

/// Split polygon by plane. Intersection points are snapped so opposite solids meet.
void split_polygon(const Plane& plane, const Polygon& poly, std::vector<Polygon>& coplanar_front,
                   std::vector<Polygon>& coplanar_back, std::vector<Polygon>& front,
                   std::vector<Polygon>& back, double plane_eps) {
  const Side side = classify_polygon(plane, poly, plane_eps);
  switch (side) {
    case Side::Coplanar:
      // Deterministic coplanar bucket by normal alignment
      if (dot(plane.normal, poly.plane.normal) >= 0.0) {
        coplanar_front.push_back(poly);
      } else {
        coplanar_back.push_back(poly);
      }
      break;
    case Side::Front:
      front.push_back(poly);
      break;
    case Side::Back:
      back.push_back(poly);
      break;
    case Side::Spanning: {
      std::vector<Vec3> fverts;
      std::vector<Vec3> bverts;
      fverts.reserve(poly.vertices.size() + 2);
      bverts.reserve(poly.vertices.size() + 2);
      const std::size_t n = poly.vertices.size();
      const double eps = plane_eps;
      for (std::size_t i = 0; i < n; ++i) {
        const Vec3 vi = snap_vec(poly.vertices[i]);
        const Vec3 vj = snap_vec(poly.vertices[(i + 1) % n]);
        const double di = plane.classify(vi);
        const double dj = plane.classify(vj);
        const Side ti = classify_vertex(di, eps);
        const Side tj = classify_vertex(dj, eps);

        if (ti != Side::Back) {
          fverts.push_back(vi);
        }
        if (ti != Side::Front) {
          bverts.push_back(vi);
        }
        // Edge crosses from one open half-space to the other
        if ((ti == Side::Front && tj == Side::Back) || (ti == Side::Back && tj == Side::Front)) {
          const double denom = di - dj;
          double t = 0.5;
          if (std::abs(denom) > kAbsEps) {
            t = di / denom;
          }
          t = clamp(t, 0.0, 1.0);
          // Lerp in already-snapped endpoint space; do NOT re-snap (that breaks
          // collinearity and creates un-repairable T-junctions on the grid).
          const Vec3 isect = lerp(vi, vj, t);
          fverts.push_back(isect);
          bverts.push_back(isect);
        }
      }
      Polygon fp(std::move(fverts));
      Polygon bp(std::move(bverts));
      if (fp.valid()) {
        fp.plane = poly.plane;
        // Keep original plane orientation (don't recompute from possibly flipped verts)
        front.push_back(std::move(fp));
      }
      if (bp.valid()) {
        bp.plane = poly.plane;
        back.push_back(std::move(bp));
      }
      break;
    }
  }
}

// ---------------------------------------------------------------------------
// BSP tree (csg.js-style solid BSP)
// ---------------------------------------------------------------------------

struct BSPNode {
  Plane plane;
  std::vector<Polygon> polygons;
  std::unique_ptr<BSPNode> front;
  std::unique_ptr<BSPNode> back;
  bool has_plane = false;

  BSPNode() = default;
  explicit BSPNode(std::vector<Polygon> polys) { build(std::move(polys)); }

  void build(std::vector<Polygon> polys) {
    if (polys.empty()) {
      return;
    }
    // Stable splitter: first valid polygon (deterministic creation order)
    std::size_t start = 0;
    while (start < polys.size() && !polys[start].valid()) {
      ++start;
    }
    if (start >= polys.size()) {
      return;
    }
    plane = polys[start].plane;
    has_plane = true;
    polygons.push_back(std::move(polys[start]));

    std::vector<Polygon> front_list;
    std::vector<Polygon> back_list;
    front_list.reserve(polys.size());
    back_list.reserve(polys.size());

    for (std::size_t i = 0; i < polys.size(); ++i) {
      if (i == start || !polys[i].valid()) {
        continue;
      }
      std::vector<Polygon> cf, cb, f, b;
      split_polygon(plane, polys[i], cf, cb, f, b, kCoplanarEps);
      // Coplanar stay at this node (both orientations)
      for (auto& p : cf) {
        polygons.push_back(std::move(p));
      }
      for (auto& p : cb) {
        polygons.push_back(std::move(p));
      }
      for (auto& p : f) {
        front_list.push_back(std::move(p));
      }
      for (auto& p : b) {
        back_list.push_back(std::move(p));
      }
    }
    if (!front_list.empty()) {
      front = std::make_unique<BSPNode>(std::move(front_list));
    }
    if (!back_list.empty()) {
      back = std::make_unique<BSPNode>(std::move(back_list));
    }
  }

  void invert() {
    for (auto& p : polygons) {
      p.flip();
    }
    if (has_plane) {
      plane.flip();
    }
    if (front) {
      front->invert();
    }
    if (back) {
      back->invert();
    }
    std::swap(front, back);
  }

  /// Keep polygon pieces that lie outside this solid (front of all planes).
  [[nodiscard]] std::vector<Polygon> clip_polygons(const std::vector<Polygon>& list) const {
    if (!has_plane) {
      return list;
    }
    std::vector<Polygon> front_list;
    std::vector<Polygon> back_list;
    front_list.reserve(list.size());
    back_list.reserve(list.size());
    for (const auto& p : list) {
      if (!p.valid()) {
        continue;
      }
      std::vector<Polygon> cf, cb, f, b;
      split_polygon(plane, p, cf, cb, f, b, kClipEps);
      // Coplanar same-normal → outside-ish (front); opposite → inside-ish (back)
      for (auto& c : cf) {
        front_list.push_back(std::move(c));
      }
      for (auto& c : cb) {
        back_list.push_back(std::move(c));
      }
      for (auto& c : f) {
        front_list.push_back(std::move(c));
      }
      for (auto& c : b) {
        back_list.push_back(std::move(c));
      }
    }
    if (front) {
      front_list = front->clip_polygons(front_list);
    }
    if (back) {
      back_list = back->clip_polygons(back_list);
    } else {
      // No back subtree → region is interior of the solid → discard
      back_list.clear();
    }
    front_list.insert(front_list.end(), back_list.begin(), back_list.end());
    return front_list;
  }

  void clip_to(const BSPNode& other) {
    polygons = other.clip_polygons(polygons);
    if (front) {
      front->clip_to(other);
    }
    if (back) {
      back->clip_to(other);
    }
  }

  /// Insert polygons into this tree (used to merge second solid's surface).
  void merge(std::vector<Polygon> polys) {
    if (polys.empty()) {
      return;
    }
    if (!has_plane) {
      build(std::move(polys));
      return;
    }
    std::vector<Polygon> front_list;
    std::vector<Polygon> back_list;
    for (auto& p : polys) {
      if (!p.valid()) {
        continue;
      }
      std::vector<Polygon> cf, cb, f, b;
      split_polygon(plane, p, cf, cb, f, b, kCoplanarEps);
      for (auto& c : cf) {
        polygons.push_back(std::move(c));
      }
      for (auto& c : cb) {
        polygons.push_back(std::move(c));
      }
      for (auto& c : f) {
        front_list.push_back(std::move(c));
      }
      for (auto& c : b) {
        back_list.push_back(std::move(c));
      }
    }
    if (!front_list.empty()) {
      if (front) {
        front->merge(std::move(front_list));
      } else {
        front = std::make_unique<BSPNode>(std::move(front_list));
      }
    }
    if (!back_list.empty()) {
      if (back) {
        back->merge(std::move(back_list));
      } else {
        back = std::make_unique<BSPNode>(std::move(back_list));
      }
    }
  }

  [[nodiscard]] std::vector<Polygon> all_polygons() const {
    std::vector<Polygon> result = polygons;
    if (front) {
      auto fp = front->all_polygons();
      result.insert(result.end(), std::make_move_iterator(fp.begin()),
                    std::make_move_iterator(fp.end()));
    }
    if (back) {
      auto bp = back->all_polygons();
      result.insert(result.end(), std::make_move_iterator(bp.begin()),
                    std::make_move_iterator(bp.end()));
    }
    return result;
  }
};

// ---------------------------------------------------------------------------
// Mesh ↔ polygons
// ---------------------------------------------------------------------------

std::vector<Polygon> mesh_to_polygons(const Mesh& mesh) {
  std::vector<Polygon> polys;
  polys.reserve(mesh.triangle_count());
  for (std::size_t t = 0; t < mesh.triangle_count(); ++t) {
    const Vec3 a = snap_vec(mesh.positions[mesh.indices[t * 3 + 0]]);
    const Vec3 b = snap_vec(mesh.positions[mesh.indices[t * 3 + 1]]);
    const Vec3 c = snap_vec(mesh.positions[mesh.indices[t * 3 + 2]]);
    Polygon p({a, b, c});
    if (p.valid()) {
      polys.push_back(std::move(p));
    }
  }
  return polys;
}

// ---------------------------------------------------------------------------
// Mesh assembly: weld → triangulate → T-junction repair → cleanup
// ---------------------------------------------------------------------------

class VertexPool {
 public:
  std::uint32_t get(const Vec3& v) {
    const Vec3 s = snap_vec(v);
    const VecKey key = make_key(s);
    // Exact grid hit
    if (const auto it = map_.find(key); it != map_.end()) {
      return it->second;
    }
    // Neighbor cells — catch cut points that landed one quanta apart
    for (int dx = -1; dx <= 1; ++dx) {
      for (int dy = -1; dy <= 1; ++dy) {
        for (int dz = -1; dz <= 1; ++dz) {
          if (dx == 0 && dy == 0 && dz == 0) continue;
          const VecKey nk{key.ix + dx, key.iy + dy, key.iz + dz};
          const auto it = map_.find(nk);
          if (it == map_.end()) continue;
          if (positions_[it->second].is_near(s, kWeldEps)) {
            return it->second;
          }
        }
      }
    }
    const auto id = static_cast<std::uint32_t>(positions_.size());
    positions_.push_back(s);
    map_.emplace(key, id);
    return id;
  }

  [[nodiscard]] const std::vector<Vec3>& positions() const noexcept { return positions_; }
  [[nodiscard]] std::vector<Vec3>& positions() noexcept { return positions_; }

 private:
  std::vector<Vec3> positions_;
  std::unordered_map<VecKey, std::uint32_t, VecKeyHash> map_;
};

struct Tri {
  std::uint32_t i0, i1, i2;
};

[[nodiscard]] double tri_area2(const Vec3& a, const Vec3& b, const Vec3& c) noexcept {
  return cross(b - a, c - a).length();
}

/// Point on open segment ab (exclusive of endpoints), collinear within eps.
[[nodiscard]] bool point_on_open_segment(const Vec3& p, const Vec3& a, const Vec3& b,
                                         double* t_out) noexcept {
  const Vec3 ab = b - a;
  const double ab_len_sq = ab.length_sq();
  if (ab_len_sq < kAreaEps) {
    return false;
  }
  const double ab_len = std::sqrt(ab_len_sq);
  const Vec3 ap = p - a;
  const double dist = cross(ab, ap).length() / ab_len;
  const double tol = std::max(kOnEdgeEps, ab_len * kOnEdgeRel);
  if (dist > tol) {
    return false;
  }
  const double t = dot(ap, ab) / ab_len_sq;
  // Keep clear of endpoints so we do not re-split at existing corners
  const double t_eps = std::max(1e-9, kOnEdgeRel);
  if (t <= t_eps || t >= 1.0 - t_eps) {
    return false;
  }
  if (t_out) {
    *t_out = t;
  }
  return true;
}

/// Split every triangle that has vertices lying in the interior of an edge.
/// Binary edge splits (never fan through a collinear triple).
/// Per-pass: build edge→midpoints map with spatial grid candidate search.
void repair_tjunctions(std::vector<Vec3>& positions, std::vector<Tri>& tris) {
  if (positions.empty() || tris.empty()) {
    return;
  }

  const double cell = std::max(kOnEdgeEps * 8.0, 1e-6);
  struct I3 {
    int x, y, z;
    bool operator==(const I3& o) const noexcept { return x == o.x && y == o.y && z == o.z; }
  };
  struct I3Hash {
    std::size_t operator()(const I3& k) const noexcept {
      return static_cast<std::size_t>(k.x) * 73856093u ^
             static_cast<std::size_t>(k.y) * 19349663u ^
             static_cast<std::size_t>(k.z) * 83492791u;
    }
  };
  auto cell_of = [cell](const Vec3& v) -> I3 {
    return {static_cast<int>(std::floor(v.x / cell)),
            static_cast<int>(std::floor(v.y / cell)),
            static_cast<int>(std::floor(v.z / cell))};
  };

  struct EdgeKey {
    std::uint32_t a, b;
    bool operator==(const EdgeKey& o) const noexcept { return a == o.a && b == o.b; }
  };
  struct EdgeHash {
    std::size_t operator()(const EdgeKey& e) const noexcept {
      return (static_cast<std::size_t>(e.a) << 32) ^ static_cast<std::size_t>(e.b);
    }
  };

  constexpr int kMaxPasses = 64;
  for (int pass = 0; pass < kMaxPasses; ++pass) {
    std::unordered_map<I3, std::vector<std::uint32_t>, I3Hash> grid;
    grid.reserve(positions.size() * 2);
    for (std::uint32_t i = 0; i < static_cast<std::uint32_t>(positions.size()); ++i) {
      grid[cell_of(positions[i])].push_back(i);
    }

    std::unordered_map<EdgeKey, std::vector<std::pair<double, std::uint32_t>>, EdgeHash> edge_mids;
    edge_mids.reserve(tris.size() * 2);

    auto undirected = [](std::uint32_t u, std::uint32_t v) -> EdgeKey {
      return u < v ? EdgeKey{u, v} : EdgeKey{v, u};
    };

    for (const Tri& tri : tris) {
      const std::uint32_t ids[3] = {tri.i0, tri.i1, tri.i2};
      for (int e = 0; e < 3; ++e) {
        const std::uint32_t ia = ids[e];
        const std::uint32_t ib = ids[(e + 1) % 3];
        if (ia != ib) {
          edge_mids.emplace(undirected(ia, ib), std::vector<std::pair<double, std::uint32_t>>{});
        }
      }
    }

    for (auto& entry : edge_mids) {
      const EdgeKey& ek = entry.first;
      auto& mids = entry.second;
      const Vec3& a = positions[ek.a];
      const Vec3& b = positions[ek.b];
      const double pad = kOnEdgeEps * 2.0;
      const int x0 = static_cast<int>(std::floor((std::min(a.x, b.x) - pad) / cell));
      const int y0 = static_cast<int>(std::floor((std::min(a.y, b.y) - pad) / cell));
      const int z0 = static_cast<int>(std::floor((std::min(a.z, b.z) - pad) / cell));
      const int x1 = static_cast<int>(std::floor((std::max(a.x, b.x) + pad) / cell));
      const int y1 = static_cast<int>(std::floor((std::max(a.y, b.y) + pad) / cell));
      const int z1 = static_cast<int>(std::floor((std::max(a.z, b.z) + pad) / cell));

      auto consider = [&](std::uint32_t vi) {
        if (vi == ek.a || vi == ek.b) return;
        double t = 0.0;
        if (point_on_open_segment(positions[vi], a, b, &t)) {
          mids.push_back({t, vi});
        }
      };

      if ((x1 - x0) > 64 || (y1 - y0) > 64 || (z1 - z0) > 64) {
        for (std::uint32_t vi = 0; vi < static_cast<std::uint32_t>(positions.size()); ++vi) {
          consider(vi);
        }
      } else {
        for (int x = x0; x <= x1; ++x) {
          for (int y = y0; y <= y1; ++y) {
            for (int z = z0; z <= z1; ++z) {
              const auto it = grid.find(I3{x, y, z});
              if (it == grid.end()) continue;
              for (std::uint32_t vi : it->second) consider(vi);
            }
          }
        }
      }
      if (mids.empty()) continue;
      std::sort(mids.begin(), mids.end());
      mids.erase(std::unique(mids.begin(), mids.end(),
                             [](const auto& p, const auto& q) { return p.second == q.second; }),
                 mids.end());
    }

    bool any_split = false;
    std::vector<Tri> next;
    next.reserve(tris.size() + 8);

    for (const Tri& tri : tris) {
      const std::uint32_t ids[3] = {tri.i0, tri.i1, tri.i2};
      int split_e = -1;
      std::uint32_t mid_idx = 0;
      double mid_t = 0.0;
      for (int e = 0; e < 3; ++e) {
        const std::uint32_t ia = ids[e];
        const std::uint32_t ib = ids[(e + 1) % 3];
        const auto it = edge_mids.find(undirected(ia, ib));
        if (it == edge_mids.end() || it->second.empty()) continue;
        const EdgeKey ek = undirected(ia, ib);
        bool got = false;
        for (const auto& pr : it->second) {
          const double t_und = pr.first;
          const std::uint32_t vi = pr.second;
          if (vi == ids[(e + 2) % 3]) continue;
          double t = (ek.a == ia) ? t_und : (1.0 - t_und);
          if (t <= 1e-9 || t >= 1.0 - 1e-9) continue;
          if (!got || t < mid_t - 1e-15 || (std::abs(t - mid_t) <= 1e-15 && vi < mid_idx)) {
            split_e = e;
            mid_idx = vi;
            mid_t = t;
            got = true;
          }
        }
        if (got) break;
      }

      if (split_e < 0) {
        next.push_back(tri);
        continue;
      }

      any_split = true;
      const std::uint32_t a = ids[split_e];
      const std::uint32_t b = ids[(split_e + 1) % 3];
      const std::uint32_t c = ids[(split_e + 2) % 3];
      const std::uint32_t m = mid_idx;
      auto push_if_valid = [&](std::uint32_t i0, std::uint32_t i1, std::uint32_t i2) {
        if (i0 == i1 || i1 == i2 || i0 == i2) return;
        if (tri_area2(positions[i0], positions[i1], positions[i2]) <= kAreaEps) return;
        next.push_back({i0, i1, i2});
      };
      push_if_valid(a, m, c);
      push_if_valid(m, b, c);
    }

    tris.swap(next);
    if (!any_split) break;
  }
}

/// Remove duplicate triangles and cancel opposite-winding pairs on the same vertices.
void dedupe_and_cancel(std::vector<Tri>& tris, const std::vector<Vec3>& positions) {
  // Canonical key: sorted vertex ids + orientation bit
  struct FaceKey {
    std::uint32_t a, b, c;
    bool operator<(const FaceKey& o) const {
      if (a != o.a) return a < o.a;
      if (b != o.b) return b < o.b;
      return c < o.c;
    }
  };

  // Map undirected face → net orientation count (+1 CCW of sorted, -1 CW)
  std::map<FaceKey, int> net;
  for (const Tri& t : tris) {
    if (t.i0 == t.i1 || t.i1 == t.i2 || t.i0 == t.i2) {
      continue;
    }
    if (tri_area2(positions[t.i0], positions[t.i1], positions[t.i2]) <= kAreaEps) {
      continue;
    }
    std::uint32_t v[3] = {t.i0, t.i1, t.i2};
    // Orientation relative to sorted order
    int perm_sign = 1;
    // Bubble-sort to ordered triple, track parity
    for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < 2 - i; ++j) {
        if (v[j] > v[j + 1]) {
          std::swap(v[j], v[j + 1]);
          perm_sign = -perm_sign;
        }
      }
    }
    // Compare original orientation to sorted
    // Original (i0,i1,i2) vs sorted: sign of permutation from original to sorted
    // We already tracked via swaps from (i0,i1,i2).
    // Wait: we started from (i0,i1,i2) and sorted — perm_sign is sign of that perm.
    // If sorted is even perm of original, same orientation as sorted order = CCW of sorted means...
    // Net: store +perm_sign for the undirected face.
    net[{v[0], v[1], v[2]}] += perm_sign;
  }

  tris.clear();
  for (const auto& [key, count] : net) {
    if (count == 0) {
      continue;  // cancelled opposite pair
    }
    // Keep a single representative; net > 0 means sorted (a,b,c) orientation
    if (count > 0) {
      tris.push_back({key.a, key.b, key.c});
    } else {
      tris.push_back({key.a, key.c, key.b});
    }
  }
}

/// Stitch open boundary edges whose endpoints nearly coincide (cut-curve mismatch).
void stitch_boundary_edges(std::vector<Vec3>& positions, std::vector<Tri>& tris) {
  if (tris.empty()) return;
  struct EdgeKey {
    std::uint32_t a, b;
    bool operator<(const EdgeKey& o) const { return a < o.a || (a == o.a && b < o.b); }
  };
  auto undirected = [](std::uint32_t u, std::uint32_t v) -> EdgeKey {
    return u < v ? EdgeKey{u, v} : EdgeKey{v, u};
  };
  const std::size_t n = positions.size();
  std::vector<std::uint32_t> parent(n);
  for (std::uint32_t i = 0; i < static_cast<std::uint32_t>(n); ++i) parent[i] = i;
  auto find = [&](std::uint32_t x) {
    while (parent[x] != x) { parent[x] = parent[parent[x]]; x = parent[x]; }
    return x;
  };
  auto unite = [&](std::uint32_t a, std::uint32_t b) {
    a = find(a); b = find(b);
    if (a == b) return;
    if (a > b) std::swap(a, b);
    parent[b] = a;
  };
  auto collect_boundary = [&]() {
    std::map<EdgeKey, int> counts;
    for (const Tri& tri : tris) {
      const std::uint32_t id[3] = {tri.i0, tri.i1, tri.i2};
      for (int e = 0; e < 3; ++e) {
        if (id[e] != id[(e + 1) % 3]) counts[undirected(id[e], id[(e + 1) % 3])] += 1;
      }
    }
    std::vector<EdgeKey> boundary;
    for (const auto& kv : counts) if (kv.second == 1) boundary.push_back(kv.first);
    return boundary;
  };
  for (int round = 0; round < 8; ++round) {
    auto boundary = collect_boundary();
    if (boundary.empty() || boundary.size() > 400) break;
    bool any = false;
    const std::size_t bn = boundary.size();
    for (std::size_t i = 0; i < bn; ++i) {
      for (std::size_t j = i + 1; j < bn; ++j) {
        const auto e1 = boundary[i];
        const auto e2 = boundary[j];
        if (e1.a == e2.a || e1.a == e2.b || e1.b == e2.a || e1.b == e2.b) continue;
        const Vec3& A = positions[e1.a];
        const Vec3& B = positions[e1.b];
        const Vec3& C = positions[e2.a];
        const Vec3& D = positions[e2.b];
        const double weld = std::max(kWeldEps * 50.0, 5e-4);  // aggressive cut-curve zip
        if (A.is_near(C, weld) && B.is_near(D, weld)) { unite(e1.a, e2.a); unite(e1.b, e2.b); any = true; }
        else if (A.is_near(D, weld) && B.is_near(C, weld)) { unite(e1.a, e2.b); unite(e1.b, e2.a); any = true; }
      }
    }
    if (!any) break;
    std::vector<Vec3> sum(n, Vec3{0, 0, 0});
    std::vector<int> cnt(n, 0);
    for (std::uint32_t i = 0; i < static_cast<std::uint32_t>(n); ++i) {
      const auto r = find(i);
      sum[r] += positions[i];
      cnt[r] += 1;
    }
    for (std::uint32_t i = 0; i < static_cast<std::uint32_t>(n); ++i) {
      if (cnt[i] > 0) positions[i] = sum[i] * (1.0 / static_cast<double>(cnt[i]));
    }
    for (Tri& tr : tris) { tr.i0 = find(tr.i0); tr.i1 = find(tr.i1); tr.i2 = find(tr.i2); }
  }
}

/// Only project vertices onto *boundary* edges (count==1). Full E×V project is too
/// slow and can distort interior geometry.
void project_boundary_tjunctions(std::vector<Vec3>& positions, std::vector<Tri>& tris) {
  if (positions.empty() || tris.empty()) return;

  struct EdgeKey {
    std::uint32_t a, b;
    bool operator<(const EdgeKey& o) const {
      return a < o.a || (a == o.a && b < o.b);
    }
  };
  auto undirected = [](std::uint32_t u, std::uint32_t v) -> EdgeKey {
    return u < v ? EdgeKey{u, v} : EdgeKey{v, u};
  };

  std::map<EdgeKey, int> counts;
  for (const Tri& tri : tris) {
    const std::uint32_t id[3] = {tri.i0, tri.i1, tri.i2};
    for (int e = 0; e < 3; ++e) {
      if (id[e] != id[(e + 1) % 3]) {
        counts[undirected(id[e], id[(e + 1) % 3])] += 1;
      }
    }
  }

  for (const auto& entry : counts) {
    if (entry.second != 1) continue;  // only open edges need T-junction help
    const std::uint32_t ia = entry.first.a;
    const std::uint32_t ib = entry.first.b;
    const Vec3& a = positions[ia];
    const Vec3& b = positions[ib];
    const Vec3 ab = b - a;
    if (ab.length_sq() < kAreaEps) continue;
    for (std::uint32_t vi = 0; vi < static_cast<std::uint32_t>(positions.size()); ++vi) {
      if (vi == ia || vi == ib) continue;
      double tparam = 0.0;
      if (!point_on_open_segment(positions[vi], a, b, &tparam)) continue;
      positions[vi] = a + ab * tparam;
    }
  }
}

Mesh polygons_to_mesh(const std::vector<Polygon>& polys) {
  VertexPool pool;
  std::vector<Tri> tris;

  for (const auto& poly : polys) {
    if (poly.vertices.size() < 3) continue;
    std::vector<std::uint32_t> ring;
    ring.reserve(poly.vertices.size());
    for (const auto& v : poly.vertices) {
      ring.push_back(pool.get(v));
    }
    std::vector<std::uint32_t> clean;
    for (std::uint32_t id : ring) {
      if (clean.empty() || clean.back() != id) clean.push_back(id);
    }
    if (clean.size() >= 2 && clean.front() == clean.back()) clean.pop_back();
    if (clean.size() < 3) continue;

    const auto i0 = clean[0];
    for (std::size_t i = 1; i + 1 < clean.size(); ++i) {
      const auto i1 = clean[i];
      const auto i2 = clean[i + 1];
      if (i0 == i1 || i1 == i2 || i0 == i2) continue;
      if (tri_area2(pool.positions()[i0], pool.positions()[i1], pool.positions()[i2]) <= kAreaEps) {
        continue;
      }
      tris.push_back({i0, i1, i2});
    }
  }

  // Lean heal: cancel duplicates → project open-edge T verts → split → cancel
  dedupe_and_cancel(tris, pool.positions());
  stitch_boundary_edges(pool.positions(), tris);
  project_boundary_tjunctions(pool.positions(), tris);
  repair_tjunctions(pool.positions(), tris);
  dedupe_and_cancel(tris, pool.positions());
  stitch_boundary_edges(pool.positions(), tris);
  project_boundary_tjunctions(pool.positions(), tris);
  repair_tjunctions(pool.positions(), tris);
  dedupe_and_cancel(tris, pool.positions());

  Mesh mesh;
  mesh.positions = pool.positions();
  mesh.indices.reserve(tris.size() * 3);
  for (const Tri& tr : tris) {
    if (tr.i0 == tr.i1 || tr.i1 == tr.i2 || tr.i0 == tr.i2) continue;
    if (tri_area2(mesh.positions[tr.i0], mesh.positions[tr.i1], mesh.positions[tr.i2]) <= kAreaEps) {
      continue;
    }
    mesh.indices.push_back(tr.i0);
    mesh.indices.push_back(tr.i1);
    mesh.indices.push_back(tr.i2);
  }

  // Compact
  std::vector<int> remap(mesh.positions.size(), -1);
  std::vector<Vec3> compact;
  compact.reserve(mesh.positions.size());
  for (auto& idx : mesh.indices) {
    if (remap[idx] < 0) {
      remap[idx] = static_cast<int>(compact.size());
      compact.push_back(mesh.positions[idx]);
    }
    idx = static_cast<std::uint32_t>(remap[idx]);
  }
  mesh.positions = std::move(compact);

  // Final weld through VertexPool (neighbor-cell merge)
  {
    VertexPool pool3;
    std::vector<std::uint32_t> mapv(mesh.positions.size());
    for (std::size_t i = 0; i < mesh.positions.size(); ++i) {
      mapv[i] = pool3.get(mesh.positions[i]);
    }
    std::vector<std::uint32_t> ni;
    ni.reserve(mesh.indices.size());
    for (std::size_t t = 0; t < mesh.indices.size(); t += 3) {
      const auto i0 = mapv[mesh.indices[t]];
      const auto i1 = mapv[mesh.indices[t + 1]];
      const auto i2 = mapv[mesh.indices[t + 2]];
      if (i0 == i1 || i1 == i2 || i0 == i2) continue;
      if (tri_area2(pool3.positions()[i0], pool3.positions()[i1], pool3.positions()[i2]) <= kAreaEps) {
        continue;
      }
      ni.push_back(i0);
      ni.push_back(i1);
      ni.push_back(i2);
    }
    mesh.positions = pool3.positions();
    mesh.indices.swap(ni);
  }

  // Final T-junction pass after weld
  {
    std::vector<Tri> final_tris;
    final_tris.reserve(mesh.triangle_count());
    for (std::size_t t = 0; t < mesh.triangle_count(); ++t) {
      final_tris.push_back({mesh.indices[t * 3], mesh.indices[t * 3 + 1], mesh.indices[t * 3 + 2]});
    }
    stitch_boundary_edges(mesh.positions, final_tris);
    project_boundary_tjunctions(mesh.positions, final_tris);
    repair_tjunctions(mesh.positions, final_tris);
    dedupe_and_cancel(final_tris, mesh.positions);
    mesh.indices.clear();
    mesh.indices.reserve(final_tris.size() * 3);
    for (const Tri& tr : final_tris) {
      mesh.indices.push_back(tr.i0);
      mesh.indices.push_back(tr.i1);
      mesh.indices.push_back(tr.i2);
    }
  }

  // Force every vertex back onto the snap grid and re-weld so is_watertight's
  // 1e-7 geometric weld sees identical coordinates for merged cut points.
  {
    VertexPool poolf;
    std::vector<std::uint32_t> mapv(mesh.positions.size());
    for (std::size_t i = 0; i < mesh.positions.size(); ++i) {
      mapv[i] = poolf.get(mesh.positions[i]);  // snap + neighbor weld
    }
    std::vector<std::uint32_t> ni;
    ni.reserve(mesh.indices.size());
    for (std::size_t t = 0; t < mesh.indices.size(); t += 3) {
      const auto i0 = mapv[mesh.indices[t]];
      const auto i1 = mapv[mesh.indices[t + 1]];
      const auto i2 = mapv[mesh.indices[t + 2]];
      if (i0 == i1 || i1 == i2 || i0 == i2) continue;
      if (tri_area2(poolf.positions()[i0], poolf.positions()[i1], poolf.positions()[i2]) <= kAreaEps)
        continue;
      ni.push_back(i0);
      ni.push_back(i1);
      ni.push_back(i2);
    }
    mesh.positions = poolf.positions();
    mesh.indices.swap(ni);

    // One last T-junction + stitch after exact grid weld
    std::vector<Tri> ft;
    ft.reserve(mesh.indices.size() / 3);
    for (std::size_t t = 0; t < mesh.indices.size(); t += 3) {
      ft.push_back({mesh.indices[t], mesh.indices[t + 1], mesh.indices[t + 2]});
    }
    stitch_boundary_edges(mesh.positions, ft);
    // re-snap after stitch averages
    VertexPool poolf2;
    std::vector<std::uint32_t> map2(mesh.positions.size());
    for (std::size_t i = 0; i < mesh.positions.size(); ++i) map2[i] = poolf2.get(mesh.positions[i]);
    for (Tri& tr : ft) {
      tr.i0 = map2[tr.i0];
      tr.i1 = map2[tr.i1];
      tr.i2 = map2[tr.i2];
    }
    mesh.positions = poolf2.positions();
    project_boundary_tjunctions(mesh.positions, ft);
    repair_tjunctions(mesh.positions, ft);
    dedupe_and_cancel(ft, mesh.positions);
    mesh.indices.clear();
    for (const Tri& tr : ft) {
      if (tr.i0 == tr.i1 || tr.i1 == tr.i2 || tr.i0 == tr.i2) continue;
      mesh.indices.push_back(tr.i0);
      mesh.indices.push_back(tr.i1);
      mesh.indices.push_back(tr.i2);
    }
    // final snap again
    VertexPool poolf3;
    std::vector<std::uint32_t> map3(mesh.positions.size());
    for (std::size_t i = 0; i < mesh.positions.size(); ++i) map3[i] = poolf3.get(mesh.positions[i]);
    ni.clear();
    for (std::size_t t = 0; t < mesh.indices.size(); t += 3) {
      const auto i0 = map3[mesh.indices[t]];
      const auto i1 = map3[mesh.indices[t + 1]];
      const auto i2 = map3[mesh.indices[t + 2]];
      if (i0 == i1 || i1 == i2 || i0 == i2) continue;
      if (tri_area2(poolf3.positions()[i0], poolf3.positions()[i1], poolf3.positions()[i2]) <= kAreaEps)
        continue;
      ni.push_back(i0); ni.push_back(i1); ni.push_back(i2);
    }
    mesh.positions = poolf3.positions();
    mesh.indices.swap(ni);
  }

  if (!mesh.empty() && mesh.volume() < 0.0) {
    for (std::size_t t = 0; t < mesh.indices.size(); t += 3) {
      std::swap(mesh.indices[t + 1], mesh.indices[t + 2]);
    }
  }

  mesh.compute_normals();
  return mesh;
}

std::unique_ptr<BSPNode> build_tree(const Mesh& mesh) {
  return std::make_unique<BSPNode>(mesh_to_polygons(mesh));
}

Mesh finish(std::vector<Polygon> polys) {
  return polygons_to_mesh(polys);
}

}  // namespace

Mesh boolean_op(const Mesh& a, const Mesh& b, BooleanOp op) {
  auto A = build_tree(a);
  auto B = build_tree(b);

  switch (op) {
    case BooleanOp::Union:
      // Keep A outside B, B outside A (csg.js sequence)
      A->clip_to(*B);
      B->clip_to(*A);
      B->invert();
      B->clip_to(*A);
      B->invert();
      A->merge(B->all_polygons());
      return finish(A->all_polygons());

    case BooleanOp::Intersection:
      A->invert();
      B->clip_to(*A);
      B->invert();
      A->clip_to(*B);
      B->clip_to(*A);
      A->merge(B->all_polygons());
      A->invert();
      return finish(A->all_polygons());

    case BooleanOp::Difference:
      // A \ B
      A->invert();
      A->clip_to(*B);
      B->clip_to(*A);
      B->invert();
      B->clip_to(*A);
      B->invert();
      A->merge(B->all_polygons());
      A->invert();
      return finish(A->all_polygons());
  }
  return {};
}

}  // namespace cad
