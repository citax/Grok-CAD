# Grok CAD (Python)

SolidWorks-style CAD foundation rewritten in **Python**:

- **PySide6** — application shell  
- **PyVista + pyvistaqt** — 3D viewport (VTK camera, lighting, orientation widget)  
- **manifold3d** — watertight CSG  
- **numpy** — math  

The previous C++/Qt6/OpenGL tree is preserved under `_archive_cpp/`.

## Layout

```
cadcore/     # pure geometry + document (no GUI imports)
app/         # PySide6 + pyvistaqt UI
tests/       # pytest for cadcore
run_cad.sh   # launch with software GL
.venv/       # project virtualenv
```

## Setup

**Important (WSL):** put the virtualenv on the Linux filesystem (`~/.venvs/...`), not under `/mnt/c`.  
VTK loads hundreds of `.so` files; installing them on the Windows mount is slow and can fail with:

`Failed to load vtkRenderingVolumeOpenGL2: No module named vtkmodules.vtkImagingCore`

```bash
uv venv "$HOME/.venvs/grok-cad" --python 3.12
uv pip install --python "$HOME/.venvs/grok-cad/bin/python" -r requirements.txt
ln -sfn "$HOME/.venvs/grok-cad" .venv
```

## Run

**Desktop:** double-click **Grok CAD** (`Grok CAD.vbs`).

**Terminal:**

```bash
./run_cad.sh
```

`run_cad.sh` forces `LIBGL_ALWAYS_SOFTWARE=1` for WSL/WSLg hosts where Zink/D3D12 fails.

## Tests

```bash
source .venv/bin/activate
pytest -q
```

## Controls

| Input | Action |
|-------|--------|
| Trackball drag | Orbit (VTK) |
| Shift / middle (VTK defaults) | Pan / zoom variants |
| View toolbar | Front / Top / Right / Iso / Fit |
| Click plane/solid | Select → tree + status bar |
| Insert → Sketch on Plane | Stub for next turn |

Reference planes: **Front (XY, blue)**, **Top (XZ, green)**, **Right (YZ, red)** at opacity ~0.42 with solid borders.
