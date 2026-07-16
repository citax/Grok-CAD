# Native Windows launcher for Grok CAD (no WSL Qt/GL overrides).
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Venv = if ($env:GROK_CAD_VENV_WIN) { $env:GROK_CAD_VENV_WIN } else { Join-Path $env:USERPROFILE ".venvs\grok-cad-win" }
$Py = Join-Path $Venv "Scripts\python.exe"
if (-not (Test-Path $Py)) {
    Write-Error "Windows venv not found at $Venv"
    exit 1
}
$env:PYTHONPATH = $Root
Remove-Item Env:LIBGL_ALWAYS_SOFTWARE -ErrorAction SilentlyContinue
Remove-Item Env:QT_QPA_PLATFORM -ErrorAction SilentlyContinue
Remove-Item Env:QT_XCB_GL_INTEGRATION -ErrorAction SilentlyContinue
Set-Location $Root
& $Py -m app.main @args
exit $LASTEXITCODE
