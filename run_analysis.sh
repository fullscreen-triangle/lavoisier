#!/bin/bash

# MTBLS1707 Complete Analysis Workflow
# This script runs the complete systematic analysis and updates GitHub Pages

set -e  # Exit on any error

echo "=========================================="
echo "MTBLS1707 Lavoisier Analysis Workflow"
echo "=========================================="

# Configuration
DATA_PATH="public/laboratory/MTBLS1707"
RESULTS_PATH="results/mtbls1707_analysis"
DOCS_PATH="docs"
QUICK_TEST=${1:-false}

# Create necessary directories
echo "Setting up analysis environment..."
mkdir -p "$RESULTS_PATH"
mkdir -p "$DOCS_PATH/assets/images"

# Verify environment
echo "Checking environment..."
if [ ! -f ".env" ]; then
    echo "Warning: .env file not found. Make sure Hugging Face API keys are configured."
fi

if [ ! -d "$DATA_PATH" ]; then
    echo "Error: MTBLS1707 data not found at $DATA_PATH"
    exit 1
fi

# Check if Python dependencies are installed
echo "Checking Python dependencies..."
python3 -c "import lavoisier, pandas, numpy, matplotlib, seaborn, tqdm" 2>/dev/null || {
    echo "Error: Required Python packages not installed."
    echo "Please install: pip install lavoisier pandas numpy matplotlib seaborn tqdm"
    exit 1
}

# Step 1: Run the main analysis
echo ""
echo "Step 1: Running MTBLS1707 systematic analysis..."
echo "----------------------------------------"

if [ "$QUICK_TEST" = "true" ]; then
    echo "Running quick test with subset of samples..."
    python3 scripts/run_mtbls1707_analysis.py \
        --data_path "$DATA_PATH" \
        --output_path "$RESULTS_PATH" \
        --quick_test
else
    echo "Running full analysis on all samples..."
    python3 scripts/run_mtbls1707_analysis.py \
        --data_path "$DATA_PATH" \
        --output_path "$RESULTS_PATH"
fi

# Check if analysis completed successfully
if [ $? -ne 0 ]; then
    echo "Error: Analysis failed!"
    exit 1
fi

echo "✅ Analysis completed successfully!"

# Step 2: Generate GitHub Pages documentation
echo ""
echo "Step 2: Generating GitHub Pages documentation..."
echo "------------------------------------------------"

python3 scripts/generate_documentation.py \
    --results_path "$RESULTS_PATH" \
    --docs_path "$DOCS_PATH"

if [ $? -ne 0 ]; then
    echo "Warning: Documentation generation encountered issues"
else
    echo "✅ Documentation updated successfully!"
fi

# Step 3: Generate summary report
echo ""
echo "Step 3: Generating summary report..."
echo "------------------------------------"

REPORT_FILE="$RESULTS_PATH/MTBLS1707_Analysis_Report.md"
if [ -f "$REPORT_FILE" ]; then
    echo "📊 Analysis Summary:"
    echo "==================="
    head -20 "$REPORT_FILE"
    echo ""
    echo "📁 Full report available at: $REPORT_FILE"
else
    echo "Warning: Summary report not found"
fi

# Step 4: Validation summary
echo ""
echo "Step 4: Validation summary..."
echo "-----------------------------"

VALIDATION_FILE="$RESULTS_PATH/performance_metrics/validation_results.json"
if [ -f "$VALIDATION_FILE" ]; then
    echo "🔍 Performance Validation Results:"
    python3 -c "
import json
import sys
try:
    with open('$VALIDATION_FILE', 'r') as f:
        validation = json.load(f)
    
    passed = sum(1 for result in validation.values() if result.get('passed', False))
    total = len(validation)
    
    print(f'Total Metrics: {total}')
    print(f'Passed: {passed}')
    print(f'Failed: {total - passed}')
    print(f'Success Rate: {passed/total*100:.1f}%')
    print()
    
    for metric, result in validation.items():
        status = '✅ PASSED' if result.get('passed', False) else '❌ FAILED'
        print(f'{metric}: {result.get(\"actual\", 0):.3f} (target: {result.get(\"target\", 0):.3f}) {status}')
        
except Exception as e:
    print(f'Error reading validation results: {e}')
"
else
    echo "Warning: Validation results not found"
fi

# Step 5: GitHub Pages integration
echo ""
echo "Step 5: GitHub Pages integration status..."
echo "-----------------------------------------"

if [ -f "$DOCS_PATH/live-results.md" ]; then
    echo "✅ Live results page updated"
fi

if [ -f "$DOCS_PATH/performance-dashboard.md" ]; then
    echo "✅ Performance dashboard generated"
fi

if [ -f "$DOCS_PATH/benchmarking.md" ]; then
    echo "✅ Benchmarking page updated"
fi

echo ""
echo "📋 GitHub Pages files generated:"
find "$DOCS_PATH" -name "*.md" -type f | sort

# Step 6: Data files for GitHub Pages
echo ""
echo "Step 6: GitHub Pages assets generated..."
echo "---------------------------------------"

ASSETS_DIR="$RESULTS_PATH/github_pages_assets"
if [ -d "$ASSETS_DIR" ]; then
    echo "📁 GitHub Pages assets:"
    find "$ASSETS_DIR" -type f | sort
    
    # Copy assets to docs directory
    if [ -d "$ASSETS_DIR/images" ]; then
        cp -r "$ASSETS_DIR/images/"* "$DOCS_PATH/assets/images/" 2>/dev/null || true
        echo "✅ Images copied to docs/assets/images/"
    fi
    
    if [ -d "$ASSETS_DIR/data_tables" ]; then
        mkdir -p "$DOCS_PATH/assets/data"
        cp -r "$ASSETS_DIR/data_tables/"* "$DOCS_PATH/assets/data/" 2>/dev/null || true
        echo "✅ Data tables copied to docs/assets/data/"
    fi
fi

echo ""
echo "🎉 ANALYSIS WORKFLOW COMPLETED!"
echo "==============================="
echo ""
echo "📂 Results Location: $RESULTS_PATH"
echo "🌐 GitHub Pages: $DOCS_PATH"
echo "📊 Analysis Report: $REPORT_FILE"
echo ""
echo "Next Steps:"
echo "1. Review the analysis results in $RESULTS_PATH"
echo "2. Check the updated GitHub Pages documentation in $DOCS_PATH"
echo "3. Commit and push changes to enable GitHub Pages"
echo "4. Optionally run with 'true' argument for quick test: ./run_analysis.sh true"
echo ""

# Optional: Git integration
if command -v git &> /dev/null && [ -d ".git" ]; then
    echo "🔄 Git status:"
    git status --porcelain | head -10
    echo ""
    echo "To commit results: git add . && git commit -m 'Update MTBLS1707 analysis results'"
fi

echo "Analysis workflow completed successfully! 🚀" 