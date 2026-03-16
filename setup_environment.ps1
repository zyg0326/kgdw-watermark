# Environment Setup Script
# Set UTF-8 encoding
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8
chcp 65001 | Out-Null

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Multi-Channel Watermark System - Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# Check Python
Write-Host "`n[1/4] Checking Python..." -ForegroundColor Yellow
python --version
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Python not found, please install Python 3.8+" -ForegroundColor Red
    exit 1
}
Write-Host "OK Python installed" -ForegroundColor Green

# Upgrade pip
Write-Host "`n[2/4] Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip
Write-Host "OK pip upgraded" -ForegroundColor Green

# Install dependencies
Write-Host "`n[3/4] Installing dependencies..." -ForegroundColor Yellow
pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    Write-Host "WARNING: Some dependencies failed, trying individual install..." -ForegroundColor Yellow
    
    $packages = @(
        "numpy", "scipy", "scikit-learn", "nltk", "spacy", 
        "joblib", "pandas", "tqdm", "colorama", "tabulate", "reedsolo"
    )
    
    foreach ($pkg in $packages) {
        Write-Host "  Installing $pkg..." -ForegroundColor Gray
        pip install $pkg
    }
}
Write-Host "OK Dependencies installed" -ForegroundColor Green

# Download spaCy model
Write-Host "`n[4/4] Downloading spaCy English model..." -ForegroundColor Yellow
python -m spacy download en_core_web_sm
if ($LASTEXITCODE -eq 0) {
    Write-Host "OK spaCy model downloaded" -ForegroundColor Green
} else {
    Write-Host "WARNING: spaCy model download failed, please run manually:" -ForegroundColor Yellow
    Write-Host "  python -m spacy download en_core_web_sm" -ForegroundColor Gray
}

# Download NLTK data
Write-Host "`nDownloading NLTK data..." -ForegroundColor Yellow
python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger')"

# Verify installation
Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "Verifying Installation" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

$success = $true

Write-Host "`nChecking core dependencies..." -ForegroundColor Yellow
$core_packages = @("numpy", "pandas", "sklearn", "nltk", "spacy")
foreach ($pkg in $core_packages) {
    python -c "import $pkg" 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  OK $pkg" -ForegroundColor Green
    } else {
        Write-Host "  ERROR $pkg" -ForegroundColor Red
        $success = $false
    }
}

Write-Host "`nChecking visualization dependencies..." -ForegroundColor Yellow
$viz_packages = @("colorama", "tabulate")
foreach ($pkg in $viz_packages) {
    python -c "import $pkg" 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  OK $pkg" -ForegroundColor Green
    } else {
        Write-Host "  WARN $pkg (optional)" -ForegroundColor Yellow
    }
}

Write-Host "`nChecking spaCy model..." -ForegroundColor Yellow
python -c "import spacy; spacy.load('en_core_web_sm')" 2>$null
if ($LASTEXITCODE -eq 0) {
    Write-Host "  OK en_core_web_sm" -ForegroundColor Green
} else {
    Write-Host "  ERROR en_core_web_sm" -ForegroundColor Red
    $success = $false
}

# Create directories
Write-Host "`nCreating output directories..." -ForegroundColor Yellow
$dirs = @("data/output", "data/cache", "data/input_cleaned", "data/report_csv")
foreach ($dir in $dirs) {
    if (!(Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        Write-Host "  OK Created $dir" -ForegroundColor Green
    } else {
        Write-Host "  OK $dir exists" -ForegroundColor Gray
    }
}

# Final result
Write-Host "`n========================================" -ForegroundColor Cyan
if ($success) {
    Write-Host "Setup Successful!" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "`nNext steps:" -ForegroundColor Yellow
    Write-Host "  1. Run quick test: .\run_quick_test.ps1" -ForegroundColor Gray
    Write-Host "  2. Run full experiment: .\run_experiments.ps1" -ForegroundColor Gray
    Write-Host "  3. View guide: cat EXPERIMENT_GUIDE.md" -ForegroundColor Gray
} else {
    Write-Host "Setup Partially Failed" -ForegroundColor Yellow
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "`nPlease check errors above and install missing packages manually" -ForegroundColor Yellow
}
