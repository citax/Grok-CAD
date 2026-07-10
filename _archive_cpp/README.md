# Grok CAD

Interactive SolidWorks-style 3D CAD foundation in **C++20** with a pure geometry **core** library and a **Qt6 + OpenGL 3.3** application shell.

This turn centers on **reference planes** and a reliable viewport/camera. The solid mesh/CSG kernel remains for later extrude/revolve and optional solid features.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  app/  (Qt6 Widgets + QOpenGLWidget)                    │
│  MainWindow · Feature Tree · Properties · Viewport      │
│  Reference planes · yaw/pitch camera · view presets     │
└───────────────────────────┬─────────────────────────────┘
                            │ uses
┌───────────────────────────▼─────────────────────────────┐
│  core/  (pure C++20, zero Qt dependency)                │
│  math · mesh · CSG (BSP) · document / feature history   │
└─────────────────────────────────────────────────────────┘
```

| Target | Description |
|--------|-------------|
| `cadcore` | Geometry & document library |
| `cadcore_tests` | Core unit tests (no GUI) |
| `cadapp` | Desktop application |

### Feature tree (SolidWorks-style roots)

Every new document is seeded with three permanent reference planes through the origin:

| Name | Plane | Tint |
|------|--------|------|
| Front Plane | XY | Blue |
| Top Plane | XZ | Green |
| Right Plane | YZ | Red |

Planes cannot be deleted. **Insert → Sketch on Plane…** is the extension point for the next turn (sketch mode on the selected plane).

Solid primitives (Box / Sphere / Cylinder) and booleans remain available under the **Solids (legacy)** toolbar for testing the mesh kernel.

## Build

### Dependencies

- CMake ≥ 3.21, C++20 compiler  
- Qt6: Core, Gui, Widgets, OpenGL, OpenGLWidgets  
- OpenGL 3.3 Core (or software GL — see WSL note)

### Qt path (aqt install under `$HOME/Qt`)

```bash
export CMAKE_PREFIX_PATH="$HOME/Qt/6.7.2/gcc_64"
export LD_LIBRARY_PATH="$HOME/Qt/6.7.2/gcc_64/lib:$LD_LIBRARY_PATH"
```

### Configure & build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$HOME/Qt/6.7.2/gcc_64"
cmake --build build -j4
```

### Tests

```bash
./build/tests/cadcore_tests
# or: ctest --test-dir build --output-on-failure
```

## Run

### One-click (recommended)

Double-click **Grok CAD** on the Windows Desktop (`Grok CAD.vbs` or `Grok CAD.bat`).  
It launches via WSL with the correct `LD_LIBRARY_PATH` and software GL fallback.

### From a terminal

```bash
./run_cad.sh
# equivalent:
# export LD_LIBRARY_PATH="$HOME/Qt/6.7.2/gcc_64/lib:$LD_LIBRARY_PATH"
# export LIBGL_ALWAYS_SOFTWARE=1
# ./build/app/cadapp
```

### WSL / WSLg OpenGL note

On some WSL hosts the hardware GL path (Zink / D3D12) fails (`ZINK: failed to choose pdev`).  
**`run_cad.sh` forces software GL** with `LIBGL_ALWAYS_SOFTWARE=1`.  

If you run the binary manually and see a blank or error viewport message, set that variable and check **stderr** for:

```
[viewport] GL_VENDOR / GL_RENDERER / GL_VERSION
```

## Viewport controls

| Input | Action |
|-------|--------|
| LMB / MMB drag | Orbit (yaw/pitch, pitch clamped) |
| Shift + drag | Pan |
| Mouse wheel | Zoom |
| Click plane or solid | Select (tree + status bar update) |
| View menu / toolbar | Front, Back, Top, Bottom, Right, Left, Iso, Zoom to Fit |

Corner **axis gizmo** shows world orientation. Status bar shows e.g. `Selected: Front Plane`.

## File formats

- **Save/Open**: `*.cad.json` (`cad-document-v1`)  
- **Export STL**: solid bodies only (planes are not exported)

## Project layout

```
CMakeLists.txt
run_cad.sh
core/
app/
tests/
README.md
```
