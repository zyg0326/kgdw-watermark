# Multi-Channel Watermark System Experiment Script
# Set UTF-8 encoding
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8
chcp 65001 | Out-Null

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Multi-Channel Watermark System Experiments" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# Check Python
Write-Host "`nChecking Python environment..." -ForegroundColor Yellow
python --version
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Python not found" -ForegroundColor Red
    exit 1
}

# Check dependencies
Write-Host "`nChecking dependencies..." -ForegroundColor Yellow
$dependencies = @("colorama", "tabulate", "tqdm", "spacy", "nltk", "pandas", "numpy", "scikit-learn")
foreach ($dep in $dependencies) {
    python -c "import $dep" 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "WARNING: $dep not installed, trying to install..." -ForegroundColor Yellow
        pip install $dep
    }
}

# Check spaCy model
Write-Host "`nChecking spaCy model..." -ForegroundColor Yellow
python -c "import spacy; spacy.load('en_core_web_sm')" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Downloading spaCy English model..." -ForegroundColor Yellow
    python -m spacy download en_core_web_sm
}

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "Experiment A: Quick Verification (Debug Mode)" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Purpose: Verify workflow, check output format" -ForegroundColor Gray
Write-Host "Config: Low intensity attacks, minimal samples" -ForegroundColor Gray

$env:PREPROC_GROUP = "base"
$env:WM_ATTACK_PROB = "0.5"
$env:WM_MAX_LINES = "5"
$env:WM_SKIP_SEMANTIC = "true"
$env:WM_SKIP_ATTACKS = "false"

Write-Host "`nStarting Experiment A..." -ForegroundColor Green
python improved_main.py

if ($LASTEXITCODE -eq 0) {
    Write-Host "`nExperiment A Completed!" -ForegroundColor Green
} else {
    Write-Host "`nExperiment A Failed!" -ForegroundColor Red
    exit 1
}

Write-Host "`nPress any key to continue to full experiment, or Ctrl+C to exit..." -ForegroundColor Yellow
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "Experiment B: Full Benchmark" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Purpose: Generate final report" -ForegroundColor Gray
Write-Host "Config: High intensity attacks, full samples" -ForegroundColor Gray

# Check for cleaned data
if (Test-Path "data/input_cleaned") {
    Write-Host "`nUsing cleaned data: data/input_cleaned" -ForegroundColor Green
} else {
    Write-Host "`nWARNING: Cleaned data not found, using raw data" -ForegroundColor Yellow
    Write-Host "Recommend running: python tools/preprocess_data.py" -ForegroundColor Yellow
}

$env:PREPROC_GROUP = "base32_crc_spacy_newton"
$env:WM_ATTACK_PROB = "0.8"
$env:WM_MAX_LINES = "50"
$env:WM_SKIP_SEMANTIC = "false"
$env:WM_SKIP_ATTACKS = "false"
$env:WM_USE_PHYSICAL_ATTACK = "false"

Write-Host "`nStarting Experiment B..." -ForegroundColor Green
python improved_main.py | Tee-Object -FilePath "experiment_log_full.txt"

if ($LASTEXITCODE -eq 0) {
    Write-Host "`nExperiment B Completed!" -ForegroundColor Green
    Write-Host "Log saved to: experiment_log_full.txt" -ForegroundColor Gray
} else {
    Write-Host "`nExperiment B Failed!" -ForegroundColor Red
    exit 1
}

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "Exporting Results to CSV" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

if (Test-Path "tools/export_results_to_csv.py") {
    python tools/export_results_to_csv.py --src "data/output" --dst "data/report_csv"
    if ($LASTEXITCODE -eq 0) {
        Write-Host "`nCSV export completed: data/report_csv" -ForegroundColor Green
    }
}

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "All Experiments Completed!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Result files:" -ForegroundColor Yellow
Write-Host "  - data/output/improved_watermark_results.json" -ForegroundColor Gray
Write-Host "  - data/output/improved_watermark_summary.json" -ForegroundColor Gray
Write-Host "  - data/report_csv/all_outputs_numeric.csv" -ForegroundColor Gray
Write-Host "  - experiment_log_full.txt" -ForegroundColor Gray
