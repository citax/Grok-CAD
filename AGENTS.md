# Grok CAD — agent rules

SolidWorks-style CAD app (Python / PySide6 / PyVista / manifold3d).  
These rules apply to every session working in this repository.

## Git: commit early, commit often

**Do not leave meaningful work uncommitted.**

- After finishing a feature, fix, or verification pass that changed project files, **create a git commit** before ending the turn.
- Commit before risky git operations (reset, checkout, rebase, force-push, bulk cleanup) so improvements cannot be lost.
- Prefer small, focused commits with clear messages (what + why).
- Do **not** commit secrets, `.venv`, or large binary junk. Leave unrelated dirt (e.g. accidental fixture noise) unstaged unless the user wants it.
- Do not push to remote unless the user asks.

If the session is about to end or switch tasks and the working tree has intentional changes, **commit first**.

## Product direction

- Match **SolidWorks-like** workflow: feature tree, PropertyManager (select → set params → OK/Cancel), undo, save/reopen (`.gcad`).
- **Fillet** rounds **edges on a solid** (`EDGE_FILLET`), not sketch-profile corners.
- **Extrude** from a sketch on a solid face **merges** (boolean union) into that solid — one continuous body, no double-counted volume. Base extrudes from reference planes stay standalone.
- **Sketch constraints** are persistent (coincident, parallel, perpendicular, H/V, equal, fix). They must survive drag, save/reopen; conflicts refuse and leave the sketch unchanged. Partial under-constraint is normal.
- Keep the PropertyManager **compact** (~240px preferred, ≤300px max); selection/hint text must not clip and must not balloon the panel.
- Failed features must show a clear message and **leave the part unchanged**.

## Architecture

| Path | Role |
|------|------|
| `cadcore/` | Geometry + document (no GUI imports) |
| `app/` | PySide6 UI, viewport, workers |
| `tests/` | pytest |
| `bench/` | Verification / UI scripts |

- Kernel: watertight CSG via **manifold3d**. Prefer real solid booleans over sketch-only hacks for solid features.
- Use the project venv (`.venv` → Linux path under WSL); prefer Python 3.12.

## Verification / unattended runs

- Automated or headless runs must **always exit** (report exit code). Never leave a process blocked on "unsaved changes" or any modal.
- Unattended detection: `GROK_CAD_UNATTENDED=1` or `QT_QPA_PLATFORM=offscreen` → discard dirty docs without dialogs. Interactive users still get save prompts.
- For solid fillets and similar: report volume before/after when relevant; result must stay watertight; material-removing ops should change volume.

## Testing

```bash
source .venv/bin/activate
pytest -q
# optional full check
GROK_CAD_UNATTENDED=1 QT_QPA_PLATFORM=offscreen python bench/edge_fillet_verify.py
```

Prefer adding or updating tests when behavior changes.

## Style

- Keep changes scoped to the request; avoid drive-by refactors and unsolicited markdown docs.
- Match existing code style in the files you edit.
