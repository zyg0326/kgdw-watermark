# Complete Workflow: Preprocess Data and Run Experiment
# Set UTF-8 encoding
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8
chcp 65001 | Out-Null

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Complete Workflow: Preprocess + Experiment" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# Step 1: Data Preprocessing
Write-Host "`n[Step 1/3] Data Preprocessing..." -ForegroundColor Yellow
Write-Host "Using enhanced preprocessing script" -ForegroundColor Gray

python tools/preprocess_data_enhanced.py --input "data/input" --output "data/input_cleaned" --max_samples 50

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Preprocessing failed!" -ForegroundColor Red
    exit 1
}

Write-Host "`nPreprocessing completed!" -ForegroundColor Green

# Step 2: Clear cache
Write-Host "`n[Step 2/3] Clearing cache..." -ForegroundColor Yellow
if (Test-Path "data/cache") {
    Remove-Item "data/cache/*" -Force -ErrorAction SilentlyContinue
    Write-Host "Cache cleared" -ForegroundColor Green
}

# Step 3: Run experiment with cleaned data
Write-Host "`n[Step 3/3] Running experiment with cleaned data..." -ForegroundColor Yellow

# Temporarily modify INPUT_DIR in improved_main.py
$mainFile = "improved_main.py"
$content = Get-Content $mainFile -Raw
$originalContent = $content

# Check if INPUT_DIR is already set to input_cleaned
if ($content -match 'INPUT_DIR = "data/input_cleaned"') {
    Write-Host "INPUT_DIR already set to cleaned data" -ForegroundColor Gray
} else {
    Write-Host "Note: To use cleaned data permanently, modify INPUT_DIR in improved_main.py" -ForegroundColor Yellow
    Write-Host "      Change: INPUT_DIR = 'data/input'" -ForegroundColor Gray
    Write-Host "      To:     INPUT_DIR = 'data/input_cleaned'" -ForegroundColor Gray
}

# Set environment variables
$env:PREPROC_GROUP = "base32_crc_spacy_newton"
$env:WM_MAX_LINES = "10"
$env:WM_SKIP_SEMANTIC = "false"
$env:WM_SKIP_ATTACKS = "false"
$env:WM_ATTACK_PROB = "0.5"

Write-Host "`nStarting experiment..." -ForegroundColor Green
python improved_main.py | Tee-Object -FilePath "experiment_log_preprocessed.txt"

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n========================================" -ForegroundColor Cyan
    Write-Host "Workflow Completed Successfully!" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "`nOutput files:" -ForegroundColor Yellow
    Write-Host "  - Cleaned data: data/input_cleaned/" -ForegroundColor Gray
    Write-Host "  - Preprocessing stats: data/input_cleaned/_preprocessing_stats.json" -ForegroundColor Gray
    Write-Host "  - Experiment results: data/output/" -ForegroundColor Gray
    Write-Host "  - Experiment log: experiment_log_preprocessed.txt" -ForegroundColor Gray
} else {
    Write-Host "`nWorkflow Failed!" -ForegroundColor Red
    exit 1
}
