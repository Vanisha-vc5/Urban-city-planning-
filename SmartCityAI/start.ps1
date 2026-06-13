# ============================================================
# SmartCityAI — PowerShell Launcher
# Run with: .\start.ps1  (from project directory)
# Or:       .\start.ps1 -Demo  (synthetic data, no internet)
# Or:       .\start.ps1 -InstallOnly  (install packages only)
# ============================================================

param(
    [switch]$Demo,
    [switch]$InstallOnly,
    [switch]$RunPipeline,
    [string]$City = "Mumbai, India"
)

$PYTHON  = "C:\Users\Vanisha\AppData\Local\Python\pythoncore-3.14-64\python.exe"
$SCRIPTS = "C:\Users\Vanisha\AppData\Local\Python\pythoncore-3.14-64\Scripts"
$PROJECT = $PSScriptRoot

# Add Scripts folder to PATH for this session
$env:PATH = "$SCRIPTS;" + $env:PATH

Write-Host ""
Write-Host "  ============================================" -ForegroundColor Cyan
Write-Host "    SmartCityAI -- Urban Infrastructure AI   " -ForegroundColor Cyan
Write-Host "  ============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Python:  $PYTHON" -ForegroundColor Gray
Write-Host "  Project: $PROJECT" -ForegroundColor Gray
Write-Host ""

# ── Install packages ─────────────────────────────────────────
function Install-Packages {
    Write-Host "  Installing dependencies from requirements.txt..." -ForegroundColor Yellow
    & $PYTHON -m pip install -r "$PROJECT\requirements.txt" --quiet --no-warn-script-location
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  Dependencies installed successfully." -ForegroundColor Green
    } else {
        Write-Host "  Some packages may have failed. Check the output." -ForegroundColor Red
    }
}

# ── Check if core packages present ───────────────────────────
$missingPackages = & $PYTHON -c "
missing = []
try: import streamlit
except: missing.append('streamlit')
try: import sklearn
except: missing.append('scikit-learn')
try: import pandas
except: missing.append('pandas')
print(','.join(missing))
" 2>$null

if ($missingPackages) {
    Write-Host "  Missing packages: $missingPackages" -ForegroundColor Red
    Install-Packages
} else {
    Write-Host "  Core packages OK." -ForegroundColor Green
}

if ($InstallOnly) {
    Install-Packages
    Write-Host "  Done. Run .\start.ps1 to launch dashboard." -ForegroundColor Green
    exit 0
}

# ── Run full pipeline if requested ───────────────────────────
if ($RunPipeline) {
    Write-Host ""
    Write-Host "  Running SmartCityAI pipeline..." -ForegroundColor Yellow
    if ($Demo) {
        & $PYTHON "$PROJECT\run_pipeline.py" --demo --skip-dashboard
    } else {
        & $PYTHON "$PROJECT\run_pipeline.py" --city $City --skip-dashboard
    }
}

# ── Launch dashboard ──────────────────────────────────────────
Write-Host ""
Write-Host "  Starting Streamlit dashboard..." -ForegroundColor Cyan
Write-Host "  Open http://localhost:8501 in your browser" -ForegroundColor Green
Write-Host "  Press Ctrl+C to stop." -ForegroundColor Gray
Write-Host ""

Set-Location $PROJECT
& "$SCRIPTS\streamlit.exe" run "$PROJECT\dashboard\app.py" --server.port 8501 --server.headless false
