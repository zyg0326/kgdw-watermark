# Quick Start - English Version (No Encoding Issues)
# UTF-8 Encoding Setup
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8
chcp 65001 | Out-Null

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Quick Start - Watermark Experiment" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

Write-Host "`n[Step 1/3] Checking Python..." -ForegroundColor Yellow
python --version
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Python not found!" -ForegroundColor Red
    exit 1
}
Write-Host "OK - Python installed" -ForegroundColor Green

Write-Host "`n[Step 2/3] Clearing cache..." -ForegroundColor Yellow
if (Test-Path "data/cache") {
    Remove-Item "data/cache/*" -Force -ErrorAction SilentlyContinue
    Write-Host "OK - Cache cleared" -ForegroundColor Green
} else {
    Write-Host "OK - No cache to clear" -ForegroundColor Gray
}

Write-Host "`n[Step 3/3] Running quick test..." -ForegroundColor Yellow
Write-Host "Config: 3 samples, skip semantic & attacks" -ForegroundColor Gray

$env:PREPROC_GROUP = "base"
$env:WM_MAX_LINES = "3"
$env:WM_SKIP_SEMANTIC = "true"
$env:WM_SKIP_ATTACKS = "true"

Write-Host "`nStarting experiment...`n" -ForegroundColor Green

python improved_main.py

Write-Host "`n========================================" -ForegroundColor Cyan
if ($LASTEXITCODE -eq 0) {
    Write-Host "SUCCESS - Test completed!" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "`nNext steps:" -ForegroundColor Yellow
    Write-Host "  1. Run full test: .\run_full_test.ps1" -ForegroundColor Gray
    Write-Host "  2. View results: cat data/output/improved_watermark_summary.json" -ForegroundColor Gray
} else {
    Write-Host "FAILED - Test failed!" -ForegroundColor Red
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "`nTroubleshooting:" -ForegroundColor Yellow
    Write-Host "  1. Check dependencies: .\verify_simple.ps1" -ForegroundColor Gray
    Write-Host "  2. Install packages: pip install -r requirements.txt" -ForegroundColor Gray
    Write-Host "  3. View guide: cat TROUBLESHOOTING.md" -ForegroundColor Gray
}
