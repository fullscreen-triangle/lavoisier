# PowerShell Script to Setup and Run Metabolomics Analysis
# =========================================================
#
# This script will:
# 1. Create a virtual environment (.venv)
# 2. Activate it
# 3. Install all dependencies
# 4. Run the metabolomics analysis on experimental files
#
# Usage: .\setup_and_run.ps1

Write-Host "=" -ForegroundColor Cyan -NoNewline; Write-Host ("=" * 79) -ForegroundColor Cyan
Write-Host "Hardware-Constrained Categorical Completion - Metabolomics Pipeline" -ForegroundColor Green
Write-Host "=" -ForegroundColor Cyan -NoNewline; Write-Host ("=" * 79) -ForegroundColor Cyan
Write-Host ""

# Step 1: Check Python installation
Write-Host "[1/5] Checking Python installation..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    Write-Host "  Found: $pythonVersion" -ForegroundColor Green
}
catch {
    Write-Host "  ERROR: Python not found. Please install Python 3.9+ first." -ForegroundColor Red
    exit 1
}

# Step 2: Create virtual environment if it doesn't exist
Write-Host ""
Write-Host "[2/5] Setting up virtual environment..." -ForegroundColor Yellow
if (-Not (Test-Path ".venv")) {
    Write-Host "  Creating new virtual environment (.venv)..." -ForegroundColor Cyan
    python -m venv .venv
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  ERROR: Failed to create virtual environment" -ForegroundColor Red
        exit 1
    }
    Write-Host "  Virtual environment created successfully" -ForegroundColor Green
}
else {
    Write-Host "  Virtual environment already exists" -ForegroundColor Green
}

# Step 3: Activate virtual environment
Write-Host ""
Write-Host "[3/5] Activating virtual environment..." -ForegroundColor Yellow

# Check if activation script exists
if (Test-Path ".venv\Scripts\Activate.ps1") {
    Write-Host "  Activating .venv..." -ForegroundColor Cyan
    & ".venv\Scripts\Activate.ps1"
    Write-Host "  Virtual environment activated" -ForegroundColor Green
}
else {
    Write-Host "  ERROR: Activation script not found" -ForegroundColor Red
    exit 1
}

# Step 4: Install dependencies
Write-Host ""
Write-Host "[4/5] Installing dependencies..." -ForegroundColor Yellow
Write-Host "  This may take a few minutes..." -ForegroundColor Cyan

# Upgrade pip first
Write-Host "  Upgrading pip..." -ForegroundColor Cyan
python -m pip install --upgrade pip --quiet

# Install requirements
Write-Host "  Installing packages from requirements.txt..." -ForegroundColor Cyan
python -m pip install -r requirements.txt --quiet

if ($LASTEXITCODE -eq 0) {
    Write-Host "  All dependencies installed successfully" -ForegroundColor Green
}
else {
    Write-Host "  WARNING: Some dependencies may have failed to install" -ForegroundColor Yellow
    Write-Host "  Continuing anyway..." -ForegroundColor Yellow
}

# Step 5: Verify experimental files exist
Write-Host ""
Write-Host "[5/5] Verifying experimental files..." -ForegroundColor Yellow
$files = @(
    "public\metabolomics\PL_Neg_Waters_qTOF.mzML",
    "public\metabolomics\TG_Pos_Thermo_Orbi.mzML"
)

$allFilesExist = $true
foreach ($file in $files) {
    if (Test-Path $file) {
        $fileSize = (Get-Item $file).Length / 1MB
        Write-Host "  Found: $file ($([math]::Round($fileSize, 2)) MB)" -ForegroundColor Green
    }
    else {
        Write-Host "  ERROR: Missing file: $file" -ForegroundColor Red
        $allFilesExist = $false
    }
}

if (-Not $allFilesExist) {
    Write-Host ""
    Write-Host "ERROR: Some experimental files are missing!" -ForegroundColor Red
    Write-Host "Please ensure files are in public/metabolomics/" -ForegroundColor Yellow
    exit 1
}

# All checks passed, ready to run
Write-Host ""
Write-Host "=" -ForegroundColor Cyan -NoNewline; Write-Host ("=" * 79) -ForegroundColor Cyan
Write-Host "Setup Complete! Starting Analysis..." -ForegroundColor Green
Write-Host "=" -ForegroundColor Cyan -NoNewline; Write-Host ("=" * 79) -ForegroundColor Cyan
Write-Host ""
Write-Host "Processing:" -ForegroundColor Yellow
Write-Host "  1. PL_Neg_Waters_qTOF.mzML (Phospholipids, Waters)" -ForegroundColor Cyan
Write-Host "  2. TG_Pos_Thermo_Orbi.mzML (Triglycerides, Thermo)" -ForegroundColor Cyan
Write-Host ""
Write-Host "Pipeline stages:" -ForegroundColor Yellow
Write-Host "  Stage 1: Spectral Preprocessing (BMD Input Filter)" -ForegroundColor Cyan
Write-Host "  Stage 2: S-Entropy Transformation (Bijective, Platform-Independent)" -ForegroundColor Cyan
Write-Host "  Stage 3: Hardware BMD Grounding (Reality Check)" -ForegroundColor Cyan
Write-Host "  Stage 4: Categorical Completion (Temporal Navigation)" -ForegroundColor Cyan
Write-Host ""

# Run the analysis
python run_metabolomics_analysis.py

# Check exit code
if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "=" -ForegroundColor Cyan -NoNewline; Write-Host ("=" * 79) -ForegroundColor Cyan
    Write-Host "Analysis Completed Successfully!" -ForegroundColor Green
    Write-Host "=" -ForegroundColor Cyan -NoNewline; Write-Host ("=" * 79) -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Results saved to: results\metabolomics_analysis\" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "To view results:" -ForegroundColor Yellow
    Write-Host "  cd results\metabolomics_analysis" -ForegroundColor Cyan
    Write-Host "  dir" -ForegroundColor Cyan
    Write-Host ""
}
else {
    Write-Host ""
    Write-Host "Analysis completed with errors (exit code: $LASTEXITCODE)" -ForegroundColor Yellow
    Write-Host "Check the log file for details: metabolomics_analysis.log" -ForegroundColor Cyan
}
