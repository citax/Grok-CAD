#include "cadcore/document/json_io.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>

namespace cad {
namespace {

std::string escape_string(const std::string& s) {
  std::string out;
  out.reserve(s.size() + 8);
  for (char c : s) {
    switch (c) {
      case '"':
        out += "\\\"";
        break;
      case '\\':
        out += "\\\\";
        break;
      case '\n':
        out += "\\n";
        break;
      case '\r':
        out += "\\r";
        break;
      case '\t':
        out += "\\t";
        break;
      default:
        out += c;
        break;
    }
  }
  return out;
}

void write_vec3(std::ostringstream& os, const Vec3& v) {
  os << '[' << v.x << ',' << v.y << ',' << v.z << ']';
}

const char* type_to_string(FeatureType t) {
  switch (t) {
    case FeatureType::PlaneFront:
      return "plane_front";
    case FeatureType::PlaneTop:
      return "plane_top";
    case FeatureType::PlaneRight:
      return "plane_right";
    case FeatureType::Box:
      return "box";
    case FeatureType::Sphere:
      return "sphere";
    case FeatureType::Cylinder:
      return "cylinder";
    case FeatureType::BooleanUnion:
      return "union";
    case FeatureType::BooleanDifference:
      return "difference";
    case FeatureType::BooleanIntersection:
      return "intersection";
  }
  return "box";
}

bool type_from_string(const std::string& s, FeatureType& out) {
  if (s == "plane_front") {
    out = FeatureType::PlaneFront;
    return true;
  }
  if (s == "plane_top") {
    out = FeatureType::PlaneTop;
    return true;
  }
  if (s == "plane_right") {
    out = FeatureType::PlaneRight;
    return true;
  }
  if (s == "box") {
    out = FeatureType::Box;
    return true;
  }
  if (s == "sphere") {
    out = FeatureType::Sphere;
    return true;
  }
  if (s == "cylinder") {
    out = FeatureType::Cylinder;
    return true;
  }
  if (s == "union") {
    out = FeatureType::BooleanUnion;
    return true;
  }
  if (s == "difference") {
    out = FeatureType::BooleanDifference;
    return true;
  }
  if (s == "intersection") {
    out = FeatureType::BooleanIntersection;
    return true;
  }
  return false;
}

// Minimal JSON tokenizer for our document schema
struct Parser {
  const std::string& s;
  std::size_t i = 0;
  std::string err;

  explicit Parser(const std::string& in) : s(in) {}

  void skip_ws() {
    while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i]))) {
      ++i;
    }
  }

  bool match(char c) {
    skip_ws();
    if (i < s.size() && s[i] == c) {
      ++i;
      return true;
    }
    return false;
  }

  bool expect(char c) {
    if (!match(c)) {
      err = std::string("expected '") + c + "'";
      return false;
    }
    return true;
  }

  bool parse_string(std::string& out) {
    skip_ws();
    if (i >= s.size() || s[i] != '"') {
      err = "expected string";
      return false;
    }
    ++i;
    out.clear();
    while (i < s.size() && s[i] != '"') {
      if (s[i] == '\\' && i + 1 < s.size()) {
        ++i;
        switch (s[i]) {
          case '"':
          case '\\':
          case '/':
            out += s[i];
            break;
          case 'n':
            out += '\n';
            break;
          case 'r':
            out += '\r';
            break;
          case 't':
            out += '\t';
            break;
          default:
            out += s[i];
            break;
        }
        ++i;
      } else {
        out += s[i++];
      }
    }
    if (i >= s.size() || s[i] != '"') {
      err = "unterminated string";
      return false;
    }
    ++i;
    return true;
  }

  bool parse_number(double& out) {
    skip_ws();
    const std::size_t start = i;
    if (i < s.size() && (s[i] == '-' || s[i] == '+')) {
      ++i;
    }
    while (i < s.size() &&
           (std::isdigit(static_cast<unsigned char>(s[i])) || s[i] == '.' || s[i] == 'e' ||
            s[i] == 'E' || s[i] == '+' || s[i] == '-')) {
      // careful: stop if we left the number (next key etc.) — digit/dot/e only after start
      if (s[i] == '+' || s[i] == '-') {
        if (i == start || (s[i - 1] != 'e' && s[i - 1] != 'E')) {
          break;
        }
      }
      ++i;
    }
    if (i == start) {
      err = "expected number";
      return false;
    }
    try {
      out = std::stod(s.substr(start, i - start));
    } catch (...) {
      err = "invalid number";
      return false;
    }
    return true;
  }

  bool parse_int(int& out) {
    double d = 0;
    if (!parse_number(d)) {
      return false;
    }
    out = static_cast<int>(d);
    return true;
  }

  bool parse_bool(bool& out) {
    skip_ws();
    if (s.compare(i, 4, "true") == 0) {
      i += 4;
      out = true;
      return true;
    }
    if (s.compare(i, 5, "false") == 0) {
      i += 5;
      out = false;
      return true;
    }
    err = "expected bool";
    return false;
  }

  bool parse_vec3(Vec3& v) {
    if (!expect('[')) {
      return false;
    }
    if (!parse_number(v.x)) {
      return false;
    }
    if (!expect(',')) {
      return false;
    }
    if (!parse_number(v.y)) {
      return false;
    }
    if (!expect(',')) {
      return false;
    }
    if (!parse_number(v.z)) {
      return false;
    }
    return expect(']');
  }

  bool parse_feature(Feature& f) {
    if (!expect('{')) {
      return false;
    }
    bool first = true;
    while (true) {
      skip_ws();
      if (match('}')) {
        break;
      }
      if (!first && !expect(',')) {
        return false;
      }
      first = false;
      std::string key;
      if (!parse_string(key) || !expect(':')) {
        return false;
      }
      if (key == "id") {
        if (!parse_int(f.id)) return false;
      } else if (key == "name") {
        if (!parse_string(f.name)) return false;
      } else if (key == "type") {
        std::string ts;
        if (!parse_string(ts) || !type_from_string(ts, f.type)) {
          err = "unknown feature type";
          return false;
        }
      } else if (key == "width") {
        if (!parse_number(f.width)) return false;
      } else if (key == "height") {
        if (!parse_number(f.height)) return false;
      } else if (key == "depth") {
        if (!parse_number(f.depth)) return false;
      } else if (key == "radius") {
        if (!parse_number(f.radius)) return false;
      } else if (key == "segments") {
        if (!parse_int(f.segments)) return false;
      } else if (key == "rings") {
        if (!parse_int(f.rings)) return false;
      } else if (key == "operand_a") {
        if (!parse_int(f.operand_a)) return false;
      } else if (key == "operand_b") {
        if (!parse_int(f.operand_b)) return false;
      } else if (key == "translation") {
        if (!parse_vec3(f.translation)) return false;
      } else if (key == "rotation_deg") {
        if (!parse_vec3(f.rotation_deg)) return false;
      } else if (key == "scale") {
        if (!parse_vec3(f.scale)) return false;
      } else if (key == "visible") {
        if (!parse_bool(f.visible)) return false;
      } else if (key == "suppressed") {
        if (!parse_bool(f.suppressed)) return false;
      } else {
        // skip unknown value: number, string, bool, array, object — simple skip
        skip_ws();
        if (i < s.size() && s[i] == '"') {
          std::string tmp;
          if (!parse_string(tmp)) return false;
        } else if (i < s.size() && (s[i] == '-' || std::isdigit(static_cast<unsigned char>(s[i])))) {
          double tmp;
          if (!parse_number(tmp)) return false;
        } else if (s.compare(i, 4, "true") == 0) {
          i += 4;
        } else if (s.compare(i, 5, "false") == 0) {
          i += 5;
        } else if (s.compare(i, 4, "null") == 0) {
          i += 4;
        } else {
          err = "unsupported value for key " + key;
          return false;
        }
      }
    }
    return true;
  }
};

}  // namespace

std::string document_to_json(const Document& doc) {
  std::ostringstream os;
  os.precision(17);
  os << "{\n";
  os << "  \"format\": \"cad-document-v1\",\n";
  os << "  \"name\": \"" << escape_string(doc.name()) << "\",\n";
  os << "  \"selected_id\": " << doc.selected_id() << ",\n";
  os << "  \"features\": [\n";
  const auto& feats = doc.features();
  for (std::size_t i = 0; i < feats.size(); ++i) {
    const auto& f = feats[i];
    os << "    {\n";
    os << "      \"id\": " << f.id << ",\n";
    os << "      \"name\": \"" << escape_string(f.name) << "\",\n";
    os << "      \"type\": \"" << type_to_string(f.type) << "\",\n";
    os << "      \"width\": " << f.width << ",\n";
    os << "      \"height\": " << f.height << ",\n";
    os << "      \"depth\": " << f.depth << ",\n";
    os << "      \"radius\": " << f.radius << ",\n";
    os << "      \"segments\": " << f.segments << ",\n";
    os << "      \"rings\": " << f.rings << ",\n";
    os << "      \"operand_a\": " << f.operand_a << ",\n";
    os << "      \"operand_b\": " << f.operand_b << ",\n";
    os << "      \"translation\": ";
    write_vec3(os, f.translation);
    os << ",\n";
    os << "      \"rotation_deg\": ";
    write_vec3(os, f.rotation_deg);
    os << ",\n";
    os << "      \"scale\": ";
    write_vec3(os, f.scale);
    os << ",\n";
    os << "      \"visible\": " << (f.visible ? "true" : "false") << ",\n";
    os << "      \"suppressed\": " << (f.suppressed ? "true" : "false") << "\n";
    os << "    }" << (i + 1 < feats.size() ? "," : "") << "\n";
  }
  os << "  ]\n";
  os << "}\n";
  return os.str();
}

bool document_from_json(const std::string& json, Document& out, std::string* error) {
  Parser p(json);
  if (!p.expect('{')) {
    if (error) *error = p.err;
    return false;
  }

  Document doc;
  int selected = -1;
  int max_id = 0;
  bool first = true;

  while (true) {
    p.skip_ws();
    if (p.match('}')) {
      break;
    }
    if (!first && !p.expect(',')) {
      if (error) *error = p.err;
      return false;
    }
    first = false;
    std::string key;
    if (!p.parse_string(key) || !p.expect(':')) {
      if (error) *error = p.err;
      return false;
    }
    if (key == "format") {
      std::string fmt;
      if (!p.parse_string(fmt)) {
        if (error) *error = p.err;
        return false;
      }
      if (fmt != "cad-document-v1") {
        if (error) *error = "unsupported format: " + fmt;
        return false;
      }
    } else if (key == "name") {
      std::string name;
      if (!p.parse_string(name)) {
        if (error) *error = p.err;
        return false;
      }
      doc.set_name(std::move(name));
    } else if (key == "selected_id") {
      if (!p.parse_int(selected)) {
        if (error) *error = p.err;
        return false;
      }
    } else if (key == "features") {
      if (!p.expect('[')) {
        if (error) *error = p.err;
        return false;
      }
      bool ffirst = true;
      while (true) {
        p.skip_ws();
        if (p.match(']')) {
          break;
        }
        if (!ffirst && !p.expect(',')) {
          if (error) *error = p.err;
          return false;
        }
        ffirst = false;
        Feature f;
        if (!p.parse_feature(f)) {
          if (error) *error = p.err;
          return false;
        }
        max_id = std::max(max_id, f.id);
        doc.features().push_back(std::move(f));
      }
    } else {
      // skip simple values
      p.skip_ws();
      if (p.i < p.s.size() && p.s[p.i] == '"') {
        std::string tmp;
        if (!p.parse_string(tmp)) {
          if (error) *error = p.err;
          return false;
        }
      } else if (p.i < p.s.size() &&
                 (p.s[p.i] == '-' || std::isdigit(static_cast<unsigned char>(p.s[p.i])))) {
        double tmp;
        if (!p.parse_number(tmp)) {
          if (error) *error = p.err;
          return false;
        }
      } else {
        if (error) *error = "unexpected key value for " + key;
        return false;
      }
    }
  }

  out.clear();
  out.set_name(doc.name());
  int max_seen = 0;
  for (const auto& f : doc.features()) {
    out.features().push_back(f);
    max_seen = std::max(max_seen, f.id);
  }
  out.ensure_next_id_after(max_seen);
  out.set_selected_id(selected);
  return true;
}

bool save_document(const Document& doc, const std::string& path) {
  std::ofstream ofs(path);
  if (!ofs) {
    return false;
  }
  ofs << document_to_json(doc);
  return static_cast<bool>(ofs);
}

bool load_document(const std::string& path, Document& out, std::string* error) {
  std::ifstream ifs(path);
  if (!ifs) {
    if (error) *error = "cannot open file";
    return false;
  }
  std::ostringstream ss;
  ss << ifs.rdbuf();
  return document_from_json(ss.str(), out, error);
}

}  // namespace cad
