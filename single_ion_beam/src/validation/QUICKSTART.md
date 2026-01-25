# Quick Start Guide
## Experimental Validation Framework

This guide will help you quickly run all validation tests and generate all figures for the quintupartite single-ion observatory paper.

---

## Prerequisites

### Required Python Packages

```bash
pip install numpy scipy matplotlib seaborn scikit-learn matplotlib-venn
```

### Verify Installation

```bash
python -c "import numpy, scipy, matplotlib, seaborn, sklearn, matplotlib_venn; print('All packages installed successfully!')"
```

---

## Running Validations

### Option 1: Generate All Plots (Recommended)

This is the fastest way to generate all 13 validation figures:

```bash
cd single_ion_beam/src/validation
python generate_all_plots.py
```

**Output:**
- All figures saved to `./figures/experimental/`
- Console output shows validation metrics
- Processing time: ~30-40 seconds

### Option 2: Run Individual Validators

For more detailed output and testing:

```bash
# Test categorical thermodynamics
python experimental_validators.py

# Test distributed observers
python -c "from distributed_observer_validator import demonstrate_distributed_observation; demonstrate_distributed_observation()"

# Test information catalysts
python -c "from information_catalyst_validator import demonstrate_information_catalysts; demonstrate_information_catalysts()"
```

### Option 3: Full Validation Suite

Run all validators including the original framework:

```bash
python run_validation.py
```

---

## Generated Figures

After running `generate_all_plots.py`, you will find 13 PNG files in `./figures/experimental/`:

| Figure | Description | Key Metric |
|--------|-------------|------------|
| 01 | 3D S-space with convex hulls | Sample separation |
| 02 | Ionization mode comparison | Mode invariance |
| 03 | Classification confusion matrix | 85.84% accuracy |
| 04 | Network properties | Modularity = 0.68 |
| 05 | MS2 coverage heatmap | Sample-mode coverage |
| 07 | Maxwell-Boltzmann distribution | Scale = 4.77 |
| 08 | Entropy production curves | dS/dt profiles |
| 09 | Ideal gas law validation | R² = 0.72 |
| 10 | Performance profiling | 35s total time |
| 13 | PCA with confidence ellipses | 83% variance |
| 14 | Metabolite overlap Venn | 234 core metabolites |
| 15 | Correlation heatmap | Sample clustering |
| 16 | Platform independence | 0.98 average score |

---

## Validation Metrics Summary

Run this to get a quick summary:

```python
from experimental_validators import generate_synthetic_experimental_data, CategoricalThermodynamicsValidator, SEntropyValidator

# Generate data
data = generate_synthetic_experimental_data()

# Validators
thermo = CategoricalThermodynamicsValidator()
s_val = SEntropyValidator()

# Key metrics
gas = thermo.validate_ideal_gas_law(data.s_coordinates, data.sample_labels)
mb = thermo.validate_maxwell_boltzmann(data.intensities)
clf = s_val.train_classifier(data.s_coordinates, data.sample_labels)
pca = s_val.perform_pca(data.s_coordinates, data.sample_labels)

print(f"Classification Accuracy: {clf['accuracy']:.2%}")
print(f"Ideal Gas R²: {gas['r_squared']:.4f}")
print(f"PCA Variance (PC1+PC2): {pca['cumulative_variance'][1]:.2%}")
```

---

## Troubleshooting

### Issue: ModuleNotFoundError

**Solution:** Install missing packages:
```bash
pip install <package-name>
```

### Issue: UnicodeEncodeError on Windows

**Solution:** The code has been updated to avoid Unicode characters. If you still see errors, try:
```bash
chcp 65001  # Set console to UTF-8
python generate_all_plots.py
```

### Issue: Memory Error

**Solution:** Reduce the number of spectra:
```python
# In generate_all_plots.py, change:
data = generate_synthetic_experimental_data(n_spectra=10000)  # Instead of 46458
```

### Issue: Figures Not Displaying

**Solution:** Figures are saved to disk automatically. To display them:
```python
import matplotlib.pyplot as plt
plt.show()  # Add this after each plot
```

---

## Customization

### Change Number of Spectra

Edit `generate_all_plots.py`:
```python
data = generate_synthetic_experimental_data(n_spectra=YOUR_NUMBER)
```

### Change Output Directory

Edit `generate_all_plots.py`:
```python
output_dir = './your/custom/path'
```

### Change Classifier Type

Edit `generate_all_plots.py`:
```python
clf_results = s_validator.train_classifier(
    data.s_coordinates,
    data.sample_labels,
    classifier_type='svm'  # Options: 'svm', 'rf'
)
```

### Adjust Figure DPI

Edit `experimental_plots.py`:
```python
plt.savefig(save_path, dpi=600, bbox_inches='tight')  # Higher resolution
```

---

## File Structure

```
single_ion_beam/src/validation/
├── __init__.py
├── modality_validators.py          # Five modality validators
├── chromatography_validator.py     # Van Deemter equation
├── temporal_resolution_validator.py # Trans-Planckian precision
├── distributed_observer_validator.py # Finite observer limits
├── information_catalyst_validator.py # Two-sided information
├── experimental_validators.py       # Categorical thermodynamics
├── experimental_plots.py            # All plotting functions
├── panel_charts.py                  # Multi-panel charts
├── generate_all_plots.py            # Main orchestration script
├── run_validation.py                # Full validation suite
├── README.md                        # Detailed documentation
├── QUICKSTART.md                    # This file
├── EXPERIMENTAL_VALIDATION_SUMMARY.md # Results summary
└── figures/
    └── experimental/                # Generated figures
```

---

## Next Steps

1. **Review Figures:** Check `./figures/experimental/` for all generated plots
2. **Read Summary:** See `EXPERIMENTAL_VALIDATION_SUMMARY.md` for detailed results
3. **Customize:** Modify parameters in `generate_all_plots.py` as needed
4. **Real Data:** Replace synthetic data with actual experimental data
5. **Paper Integration:** Include figures in the main paper

---

## Expected Output

When you run `generate_all_plots.py`, you should see:

```
================================================================================
GENERATING ALL EXPERIMENTAL VALIDATION PLOTS
================================================================================

[1/16] Generating synthetic experimental data...
  Total spectra: 46,458
  Samples: {'M3': 15481, 'M4': 15299, 'M5': 15678}
  Ionization modes: {'positive': 23229, 'negative': 23229}

[2/16] Plotting 3D S-space with convex hulls...
Saved: ./figures/experimental\01_3d_s_space_convex_hulls.png

... (continues for all 16 steps) ...

================================================================================
VALIDATION SUMMARY
================================================================================

Total spectra analyzed: 46,458
Samples: 3
Ionization modes: 2

Classification accuracy: 85.84%
Ideal gas law R²: 0.7228
Maxwell-Boltzmann KS p-value: 0.0000
PCA cumulative variance (PC1+PC2): 83.24%

All figures saved to: ./figures/experimental/

================================================================================
VALIDATION COMPLETE!
================================================================================
```

---

## Support

For issues or questions:
1. Check `README.md` for detailed documentation
2. Review `EXPERIMENTAL_VALIDATION_SUMMARY.md` for expected results
3. Examine the source code in `experimental_validators.py` and `experimental_plots.py`

---

**Version:** 1.0  
**Last Updated:** January 21, 2026  
**Status:** ✓ Fully functional
