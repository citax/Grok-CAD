# Contributing to Grok CAD

Thanks for helping build an open, SolidWorks-style CAD product in Python.

## Ways to contribute

- **Bugs** — crash, wrong solid, constraint fight, UI glitch  
- **Features** — modeling ops, sketch tools, UX polish  
- **Tests** — pytest coverage for `cadcore/` and headless UI paths  
- **Docs** — README, tutorials, sample `.gcad` parts  
- **Perf** — viewport FPS, rebuild time, mesh ops  

## Development setup

```bash
git clone https://github.com/citax/Grok-CAD.git
cd Grok-CAD

# Prefer venv on Linux FS (WSL): not under /mnt/c
uv venv "$HOME/.venvs/grok-cad" --python 3.12
uv pip install --python "$HOME/.venvs/grok-cad/bin/python" -r requirements.txt
ln -sfn "$HOME/.venvs/grok-cad" .venv

source .venv/bin/activate
pytest -q
./run_cad.sh
```

See [README.md](README.md) for platform notes.

## Project layout

| Path | Role |
|------|------|
| `cadcore/` | Geometry + document — **no GUI imports** |
| `app/` | PySide6 UI, viewport, workers |
| `tests/` | pytest |
| `bench/` | Verification / UI scripts |

Keep GUI out of `cadcore`. Prefer real manifold3d solids over sketch-only hacks for solid features.

## Coding guidelines

- Match existing style in the files you edit  
- Keep PRs focused — avoid drive-by refactors  
- Failed features must **leave the part unchanged** and show a clear message  
- Sketch constraints / driving dimensions must survive drag and save/reopen; conflicts refuse  
- Face extrudes **merge** (boolean union) into the solid  
- Fillet = **edge fillet on a solid** (`EDGE_FILLET`), not sketch-profile corners  
- PropertyManager stays compact (~240px preferred, ≤300px)  

Full product rules: [AGENTS.md](AGENTS.md).

## Tests

```bash
source .venv/bin/activate
pytest -q
```

For UI / fillet benches that must not hang on dialogs:

```bash
GROK_CAD_UNATTENDED=1 QT_QPA_PLATFORM=offscreen python bench/edge_fillet_verify.py
```

Add or update tests when behavior changes.

## Pull requests

1. Branch from `main`  
2. One concern per PR when possible  
3. Describe **what** and **why**  
4. Link issues  
5. Ensure `pytest -q` is green  

## Unattended / CI note

When `GROK_CAD_UNATTENDED=1` or `QT_QPA_PLATFORM=offscreen`, the app must **always exit** — discard dirty docs without modal dialogs.

## Code of conduct

Be respectful. Assume good intent. No harassment. Maintainers may close issues/PRs that are hostile or spam.

## License

By contributing, you agree your contributions are licensed under the MIT License (see [LICENSE](LICENSE)).
