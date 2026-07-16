@echo off
setlocal
REM Native Windows launcher for Grok CAD.
REM Do NOT set LIBGL_ALWAYS_SOFTWARE / QT_QPA_PLATFORM=xcb — those are WSL-only.

set "ROOT=%~dp0"
set "VENV=%USERPROFILE%\.venvs\grok-cad-win"
if not "%GROK_CAD_VENV_WIN%"=="" set "VENV=%GROK_CAD_VENV_WIN%"

if not exist "%VENV%\Scripts\python.exe" (
  echo ERROR: Windows venv not found at %VENV%
  echo Create it with:
  echo   py -3.12 -m venv "%%USERPROFILE%%\.venvs\grok-cad-win"
  echo   "%%USERPROFILE%%\.venvs\grok-cad-win\Scripts\python.exe" -m pip install -r requirements.txt
  exit /b 1
)

set "PYTHONPATH=%ROOT%"
REM Clear WSL leftovers if launched from a mixed shell
set "LIBGL_ALWAYS_SOFTWARE="
set "QT_QPA_PLATFORM="
set "QT_XCB_GL_INTEGRATION="

cd /d "%ROOT%"
REM Prefer pythonw for Desktop/GUI launch (no console window). Fall back to python.exe.
if exist "%VENV%\Scripts\pythonw.exe" (
  start "" "%VENV%\Scripts\pythonw.exe" -m app.main %*
) else (
  "%VENV%\Scripts\python.exe" -m app.main %*
)
exit /b %ERRORLEVEL%
