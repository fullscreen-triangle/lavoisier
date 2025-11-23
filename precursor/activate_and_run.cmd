@echo off
REM Batch Script to Activate Virtual Environment and Run Metabolomics Analysis
REM ===========================================================================
REM
REM This is a simpler alternative if PowerShell script execution is disabled
REM
REM Usage: activate_and_run.cmd

echo ================================================================================
echo Hardware-Constrained Categorical Completion - Metabolomics Pipeline
echo ================================================================================
echo.

REM Step 1: Check if .venv exists
echo [1/3] Checking virtual environment...
if not exist ".venv" (
    echo   ERROR: Virtual environment not found!
    echo   Please run: python -m venv .venv
    echo   Then run this script again.
    pause
    exit /b 1
)
echo   Found: .venv
echo.

REM Step 2: Activate virtual environment
echo [2/3] Activating virtual environment...
call .venv\Scripts\activate.bat
if errorlevel 1 (
    echo   ERROR: Failed to activate virtual environment
    pause
    exit /b 1
)
echo   Activated successfully
echo.

REM Step 3: Run analysis
echo [3/3] Starting metabolomics analysis...
echo.
echo ================================================================================
echo Processing experimental files...
echo ================================================================================
echo.

python run_metabolomics_analysis.py

if errorlevel 1 (
    echo.
    echo ================================================================================
    echo Analysis completed with errors
    echo ================================================================================
    echo Check metabolomics_analysis.log for details
    pause
    exit /b 1
) else (
    echo.
    echo ================================================================================
    echo Analysis Completed Successfully!
    echo ================================================================================
    echo.
    echo Results saved to: results\metabolomics_analysis\
    echo.
    pause
)
