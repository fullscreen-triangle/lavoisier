3D scatter plot (you already have this - Figure 1 in your paper!)

X-axis: S_knowledge (structural information)
Y-axis: S_time (temporal positioning)
Z-axis: S_entropy (thermodynamic state)

Points: 46,458 spectra
Colors: 
- M3 (red): 8,611 spectra
- M4 (cyan): 8,429 spectra
- M5 (teal): 8,807 spectra

Enhancement: Add convex hulls around each sample
- Shows sample separation in S-space
- Validates that S-coordinates distinguish biological samples

Statistics to add:
- Centroid for each sample
- Variance within each sample
- Distance between samples
2D scatter plot: S_knowledge vs S_entropy

Points colored by ionization mode:
- Positive ESI (blue): 23,456 spectra
- Negative ESI (red): 23,002 spectra

Show separation between modes:
- Positive: Higher S_knowledge (more fragments)
- Negative: Different S_entropy distribution

Statistics:
- Mean S_k (positive): ?
- Mean S_k (negative): ?
- t-test p-value: ?
- Effect size (Cohen's d): ?

Inset: Coherence comparison
Box plot showing:
- Positive: 0.052 ± 0.005
- Negative: 0.043 ± 0.004
- p < 0.001 (significant difference)

Confusion matrix for sample classification

Using S-entropy coordinates as features:
- Train classifier (SVM, Random Forest, or Neural Network)
- Predict sample identity (M3, M4, M5)
- Show accuracy

Expected results:
           Predicted
           M3    M4    M5
Actual M3  XX%   X%    X%
       M4  X%    XX%   X%
       M5  X%    X%    XX%

Overall accuracy: ?%
Precision: ?%
Recall: ?%
F1-score: ?%

Annotation: "S-coordinates enable sample discrimination"
Bar chart comparing network properties across samples

X-axis: Network metric
Y-axis: Value

Metrics for each sample (M3, M4, M5):
1. Number of nodes (metabolites)
2. Number of edges (transitions)
3. Average degree
4. Clustering coefficient
5. Average path length
6. Network diameter

Colors: M3 (red), M4 (cyan), M5 (teal)

Statistics:
- Which sample has most complex network?
- Are differences significant?

Annotation: "Network complexity reflects metabolic diversity"
Heatmap showing MS2 coverage across samples

X-axis: Sample (M3, M4, M5)
Y-axis: Ionization mode (Positive, Negative)

Color: Number of MS2 spectra

Data from Table 1:
M3 Negative: 549 + 53 = 602
M3 Positive: 447 + 44 = 491
M4 Negative: 939
M4 Positive: 787 + 61 = 848
M5 Negative: 596 + 84 = 680
M5 Positive: 727

Annotation: "M4 has highest MS2 coverage"

Inset: MS2/MS1 ratio
Bar chart showing fragmentation rate
3D surface plot showing categorical temperature

X-axis: Retention time (minutes)
Y-axis: m/z range (100-1000)
Z-axis: Categorical temperature T_cat

Surface colored by temperature:
- Blue (low T): Early retention time
- Red (high T): Late retention time

Data: MS1 spectra from all samples

Formula: T_cat = (ℏ/k_B)(dM/dt)
where M = number of distinct m/z values

Annotation: "Temperature increases with chromatographic complexity"

Viewing angle: 30° elevation, 45° azimuth
Histogram of peak intensities

X-axis: Normalized intensity I/⟨I⟩
Y-axis: Probability density

Data: All 16,045,368 peaks

Maxwell-Boltzmann fit:
P(I) = (2/√π) (I/⟨I⟩)^(1/2) exp(-I/⟨I⟩)

Statistics:
- χ² test: p-value = ?
- Kolmogorov-Smirnov test: p-value = ?
- Mean intensity: ⟨I⟩ = ?
- Temperature: T_cat = ?

Annotation: "Intensity distribution follows MB statistics"

Inset: Q-Q plot
Quantile-quantile plot showing goodness of fit
Line plot: Entropy production over retention time

X-axis: Retention time (minutes)
Y-axis: dS/dt (entropy production rate)

Three curves:
- M3 (red)
- M4 (cyan)
- M5 (teal)

Shows: Entropy production peaks at:
- Early elution (polar metabolites)
- Late elution (lipids)

Statistics:
- Peak positions: ?
- Maximum dS/dt: ?
- Total entropy produced: ?

Annotation: "Entropy production reflects metabolic complexity"
Scatter plot: PV vs T_cat

X-axis: Categorical temperature T_cat
Y-axis: PV product (categorical pressure × volume)

Data points: Each spectrum as one "gas molecule"

Linear fit: PV = k_B T_cat

Statistics:
- Slope: ? (should be k_B)
- Intercept: ? (should be 0)
- R² = ?
- Deviation from ideal: ?%

Annotation: "Spectra obey ideal gas law in categorical space"

Color by sample: M3 (red), M4 (cyan), M5 (teal)
Stacked bar chart showing time for each stage

X-axis: File name (10 files)
Y-axis: Processing time (seconds)

Stacked segments:
1. Preprocessing (blue): 111-340 s
2. S-Entropy transform (green): ?
3. Fragmentation network (yellow): ?
4. BMD grounding (orange): ?
5. Categorical completion (red): ?

Total height: Total processing time per file

Statistics:
- Mean total time: ?
- Std dev: ?
- Bottleneck stage: ?

Annotation: "Preprocessing dominates total time"


Line plot: Memory usage during processing

X-axis: Processing stage
Y-axis: Memory usage (GB)

Five curves (representative files):
- A_M3_negPFP_03 (largest file)
- A_M4_posPFP_01 (medium file)
- A_M3_posPFP_02 (smallest file)

Shows: Memory peaks during:
- Peak detection
- Network construction
- Categorical completion

Statistics:
- Peak memory: ?
- Mean memory: ?
- Memory per spectrum: ?

Annotation: "Memory scales linearly with file size"

Scatter plot: Accuracy vs time trade-off

X-axis: Processing time per spectrum (seconds)
Y-axis: Identification accuracy (%)

Data points: Different parameter settings

Pareto front: Optimal trade-off curve

Current settings marked with star

Shows: Optimal balance between speed and accuracy

Annotation: "Current settings near Pareto optimal"
PCA plot of S-entropy coordinates

X-axis: PC1 (% variance explained)
Y-axis: PC2 (% variance explained)

Points: 10 files
Colors: Sample (M3 red, M4 cyan, M5 teal)
Shapes: Mode (circle = positive, square = negative)

Ellipses: 95% confidence intervals

Statistics:
- PC1 variance: ?%
- PC2 variance: ?%
- Total variance (PC1+PC2): ?%

Annotation: "First 2 PCs capture ?% variance"

Inset: Scree plot
Variance explained by each PC
Venn diagram showing metabolite overlap

Three circles:
- M3: ? metabolites
- M4: ? metabolites
- M5: ? metabolites

Overlaps:
- M3 ∩ M4: ?
- M3 ∩ M5: ?
- M4 ∩ M5: ?
- M3 ∩ M4 ∩ M5: ? (core metabolome)

Statistics:
- Total unique: ?
- Core metabolome: ?
- Sample-specific: ?

Annotation: "Core metabolome: ? metabolites"


Heatmap: Pairwise correlation between files

10×10 matrix

Color: Pearson correlation of S-entropy coordinates

Shows:
- Replicates: High correlation (>0.9)
- Same sample, different mode: Moderate (0.6-0.8)
- Different samples: Lower (0.4-0.6)

Annotation: "Technical replicates highly correlated"

Dendrogram: Hierarchical clustering

Bar chart: Platform independence score

X-axis: Sample
Y-axis: Independence score (0-1)

Score = 1 - (variance_within / variance_total)

Three bars:
- M3: Score = ?
- M4: Score = ?
- M5: Score = ?

Higher score = more platform-independent

Statistics:
- Mean score: ?
- Target: >0.8

Annotation: "High independence scores validate framework"
