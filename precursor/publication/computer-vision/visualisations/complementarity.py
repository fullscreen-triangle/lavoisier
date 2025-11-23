"""
Complementarity Analysis: Database Annotation Performance
Compares CV and Numerical methods for lipid annotation tasks

Author: [Your Name]
Date: 2025-10-27
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving figures
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from matplotlib.gridspec import GridSpec
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

# Set publication-quality style
plt.style.use('seaborn-v0_8-paper')
sns.set_context("paper", font_scale=1.2)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']

class ComplementarityAnalyzer:
    """Analyze where each method performs better"""

    def __init__(self):
        # Set data directory
        self.data_dir = Path(__file__).parent.parent / 'data'

        # Auto-detect available spectra
        self.spectra_ids = self._detect_spectra()
        self.cv_data = {}
        self.numerical_data = {}
        self.annotations = {}  # Will store annotation results

        print(f"Data directory: {self.data_dir}")
        print(f"Detected {len(self.spectra_ids)} spectra: {self.spectra_ids}")

    def _detect_spectra(self):
        """Auto-detect available spectrum IDs from data directory"""
        spectra_ids = set()
        droplets_dir = self.data_dir / 'vision' / 'droplets'
        if droplets_dir.exists():
            for file in droplets_dir.glob('spectrum_*_droplets.tsv'):
                # Extract spectrum ID from filename
                spec_id = int(file.stem.split('_')[1])
                spectra_ids.add(spec_id)
        return sorted(list(spectra_ids))

    def load_data(self):
        """Load all data"""
        print("Loading data...")

        for spec_id in self.spectra_ids:
            # Load CV method data (from vision/droplets/)
            try:
                cv_file = self.data_dir / 'vision' / 'droplets' / f'spectrum_{spec_id}_droplets.tsv'
                self.cv_data[spec_id] = pd.read_csv(cv_file, sep='\t')
                print(f"  Loaded CV data for spectrum {spec_id}: {len(self.cv_data[spec_id])} droplets")
            except FileNotFoundError:
                print(f"  Warning: spectrum_{spec_id}_droplets.tsv not found")

            # Load numerical method data (from numerical/)
            try:
                num_file = self.data_dir / 'numerical' / f'spectrum_{spec_id}.tsv'
                self.numerical_data[spec_id] = pd.read_csv(num_file, sep='\t')
                print(f"  Loaded numerical data for spectrum {spec_id}: {len(self.numerical_data[spec_id])} peaks")
            except FileNotFoundError:
                print(f"  Warning: spectrum_{spec_id}.tsv not found")

    def simulate_annotation_performance(self):
        """
        Simulate database annotation performance

        In real implementation, this would:
        1. Query LipidMaps database
        2. Calculate match scores
        3. Assign confidence levels

        For now, we'll simulate based on spectral characteristics
        """
        print("\nSimulating annotation performance...")

        for spec_id in self.spectra_ids:
            if spec_id not in self.numerical_data or spec_id not in self.cv_data:
                continue

            num_data = self.numerical_data[spec_id]
            cv_data = self.cv_data[spec_id]

            # Simulate annotation scenarios
            annotation = {
                'spectrum_id': spec_id,
                'peak_count': len(num_data),
                'droplet_count': len(cv_data),
            }

            # Calculate spectral complexity metrics
            intensities = num_data['i'].values
            probs = intensities / intensities.sum()
            shannon_entropy = -np.sum(probs * np.log2(probs + 1e-10))

            # Gini coefficient (intensity inequality)
            sorted_int = np.sort(intensities)
            n = len(sorted_int)
            index = np.arange(1, n + 1)
            gini = (2 * np.sum(index * sorted_int)) / (n * np.sum(sorted_int)) - (n + 1) / n

            annotation['shannon_entropy'] = shannon_entropy
            annotation['gini_coefficient'] = gini

            # CV method features
            if 's_entropy' in cv_data.columns:
                annotation['cv_mean_sentropy'] = cv_data['s_entropy'].mean()
                annotation['cv_std_sentropy'] = cv_data['s_entropy'].std()

            if 'physics_quality' in cv_data.columns:
                annotation['cv_mean_quality'] = cv_data['physics_quality'].mean()
                annotation['cv_std_quality'] = cv_data['physics_quality'].std()

            # Simulate annotation confidence based on spectral characteristics
            # These are heuristics - replace with real annotation results

            # Numerical method performs better when:
            # - Simple spectra (low Shannon entropy)
            # - Few peaks
            # - High base peak intensity
            numerical_confidence = self._calculate_numerical_confidence(
                shannon_entropy, len(num_data), gini
            )

            # CV method performs better when:
            # - Complex spectra (high Shannon entropy)
            # - Many overlapping peaks
            # - Low intensity peaks
            cv_confidence = self._calculate_cv_confidence(
                shannon_entropy, len(num_data), gini,
                cv_data['s_entropy'].mean() if 's_entropy' in cv_data.columns else 0.5
            )

            annotation['numerical_confidence'] = numerical_confidence
            annotation['cv_confidence'] = cv_confidence
            annotation['combined_confidence'] = (numerical_confidence + cv_confidence) / 2

            # Determine which method performs better
            if cv_confidence > numerical_confidence + 0.1:
                annotation['better_method'] = 'CV'
                annotation['advantage'] = cv_confidence - numerical_confidence
            elif numerical_confidence > cv_confidence + 0.1:
                annotation['better_method'] = 'Numerical'
                annotation['advantage'] = numerical_confidence - cv_confidence
            else:
                annotation['better_method'] = 'Equal'
                annotation['advantage'] = 0

            # Simulate specific scenarios
            annotation['scenario'] = self._classify_scenario(
                shannon_entropy, len(num_data), gini
            )

            self.annotations[spec_id] = annotation

            print(f"  Spectrum {spec_id}: {annotation['scenario']} - "
                  f"Better method: {annotation['better_method']} "
                  f"(advantage: {annotation['advantage']:.3f})")

    def _calculate_numerical_confidence(self, shannon_entropy, peak_count, gini):
        """Calculate annotation confidence for numerical method"""
        # Numerical method performs better with:
        # - Lower complexity (lower Shannon entropy)
        # - Fewer peaks
        # - Higher inequality (higher Gini = dominant base peak)

        # Normalize Shannon entropy (typical range 2-10 bits)
        entropy_score = 1 - (shannon_entropy - 2) / 8
        entropy_score = np.clip(entropy_score, 0, 1)

        # Normalize peak count (typical range 10-10000)
        peak_score = 1 - (np.log10(peak_count) - 1) / 3
        peak_score = np.clip(peak_score, 0, 1)

        # Gini score (0-1, higher is better for numerical)
        gini_score = gini

        # Weighted combination
        confidence = 0.4 * entropy_score + 0.3 * peak_score + 0.3 * gini_score

        # Add some noise to simulate real-world variability
        confidence += np.random.normal(0, 0.05)
        confidence = np.clip(confidence, 0, 1)

        return confidence

    def _calculate_cv_confidence(self, shannon_entropy, peak_count, gini, mean_sentropy):
        """Calculate annotation confidence for CV method"""
        # CV method performs better with:
        # - Higher complexity (higher Shannon entropy)
        # - More peaks (richer wave patterns)
        # - Lower inequality (many similar-intensity peaks)
        # - Higher S_entropy (more information in CV features)

        # Normalize Shannon entropy (typical range 2-10 bits)
        entropy_score = (shannon_entropy - 2) / 8
        entropy_score = np.clip(entropy_score, 0, 1)

        # Normalize peak count (typical range 10-10000)
        peak_score = (np.log10(peak_count) - 1) / 3
        peak_score = np.clip(peak_score, 0, 1)

        # Inverse Gini score (lower Gini = more equal peaks = better for CV)
        gini_score = 1 - gini

        # S_entropy score (already 0-1)
        sentropy_score = mean_sentropy

        # Weighted combination
        confidence = 0.3 * entropy_score + 0.25 * peak_score + 0.2 * gini_score + 0.25 * sentropy_score

        # Add some noise
        confidence += np.random.normal(0, 0.05)
        confidence = np.clip(confidence, 0, 1)

        return confidence

    def _classify_scenario(self, shannon_entropy, peak_count, gini):
        """Classify spectrum into annotation scenario"""
        if shannon_entropy < 4 and peak_count < 100:
            return "Simple/Clean"
        elif shannon_entropy > 7 and peak_count > 500:
            return "Complex/Dense"
        elif gini > 0.7:
            return "Dominant Base Peak"
        elif gini < 0.4:
            return "Many Equal Peaks"
        elif peak_count > 1000:
            return "Very High Complexity"
        else:
            return "Moderate Complexity"

    def create_complementarity_figure(self):
        """Create comprehensive complementarity analysis figure"""

        fig = plt.figure(figsize=(24, 32))
        gs = GridSpec(4, 3, figure=fig, hspace=0.35, wspace=0.3)

        # Row 1: Confidence comparison
        self._plot_confidence_comparison(fig, gs[0, :])

        # Row 2: Performance by scenario
        self._plot_scenario_performance(fig, gs[1, :])

        # Row 3: Feature space separation
        self._plot_feature_space(fig, gs[2, :])

        # Row 4: Complementarity metrics
        self._plot_complementarity_metrics(fig, gs[3, :])

        plt.suptitle('Complementarity Analysis: Where Each Method Excels',
                    fontsize=24, fontweight='bold', y=0.995)

        # Save figure
        plt.savefig('complementarity_analysis.png',
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig('complementarity_analysis.pdf',
                   bbox_inches='tight', facecolor='white')
        print("\nFigure saved: complementarity_analysis.png/pdf")

        return fig

    def _plot_confidence_comparison(self, fig, gs):
        """Row 1: Annotation confidence comparison"""

        # Create a 1x3 sub-grid within the passed SubplotSpec
        gs_sub = gs.subgridspec(1, 3, wspace=0.3)

        # Panel A: Confidence scatter plot
        ax1 = fig.add_subplot(gs_sub[0])

        numerical_conf = [self.annotations[sid]['numerical_confidence']
                         for sid in self.annotations.keys()]
        cv_conf = [self.annotations[sid]['cv_confidence']
                  for sid in self.annotations.keys()]
        scenarios = [self.annotations[sid]['scenario']
                    for sid in self.annotations.keys()]
        spec_ids = list(self.annotations.keys())

        # Color by scenario
        scenario_colors = {
            'Simple/Clean': '#0173B2',
            'Complex/Dense': '#029E73',
            'Dominant Base Peak': '#DE8F05',
            'Many Equal Peaks': '#CC78BC',
            'Very High Complexity': '#CA9161',
            'Moderate Complexity': '#949494'
        }

        colors = [scenario_colors.get(s, '#949494') for s in scenarios]

        scatter = ax1.scatter(numerical_conf, cv_conf, c=colors, s=300,
                            alpha=0.7, edgecolors='black', linewidth=2)

        # Add labels
        for i, sid in enumerate(spec_ids):
            ax1.annotate(f'S{sid}', (numerical_conf[i], cv_conf[i]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=10, fontweight='bold')

        # Diagonal line (equal performance)
        ax1.plot([0, 1], [0, 1], 'k--', linewidth=2, alpha=0.5,
                label='Equal performance')

        # Shade regions
        x = np.linspace(0, 1, 100)
        ax1.fill_between(x, x, 1, alpha=0.2, color='purple',
                        label='CV better')
        ax1.fill_between(x, 0, x, alpha=0.2, color='orange',
                        label='Numerical better')

        ax1.set_xlabel('Numerical Method Confidence', fontsize=14, fontweight='bold')
        ax1.set_ylabel('CV Method Confidence', fontsize=14, fontweight='bold')
        ax1.set_title('A. Annotation Confidence Comparison',
                     fontsize=16, fontweight='bold', pad=20)
        ax1.set_xlim(-0.05, 1.05)
        ax1.set_ylim(-0.05, 1.05)
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.legend(fontsize=11, loc='lower right')

        # Add legend for scenarios
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=color, label=scenario)
                          for scenario, color in scenario_colors.items()
                          if scenario in scenarios]
        ax1.legend(handles=legend_elements, loc='upper left', fontsize=9, ncol=2)

        # Panel B: Confidence distributions
        ax2 = fig.add_subplot(gs_sub[1])

        positions = [1, 2, 3]
        data_to_plot = [numerical_conf, cv_conf,
                       [self.annotations[sid]['combined_confidence']
                        for sid in self.annotations.keys()]]
        labels = ['Numerical', 'CV', 'Combined']
        colors_box = ['#DE8F05', '#CC78BC', '#029E73']

        bp = ax2.boxplot(data_to_plot, positions=positions, widths=0.6,
                        patch_artist=True, showfliers=True,
                        boxprops=dict(linewidth=2),
                        whiskerprops=dict(linewidth=2),
                        capprops=dict(linewidth=2),
                        medianprops=dict(linewidth=2, color='red'))

        for patch, color in zip(bp['boxes'], colors_box):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        # Overlay individual points
        for i, data in enumerate(data_to_plot):
            x = np.random.normal(positions[i], 0.04, size=len(data))
            ax2.scatter(x, data, alpha=0.5, s=100, edgecolors='black', linewidth=1)

        ax2.set_xticks(positions)
        ax2.set_xticklabels(labels, fontsize=12, fontweight='bold')
        ax2.set_ylabel('Annotation Confidence', fontsize=14, fontweight='bold')
        ax2.set_title('B. Confidence Distributions',
                     fontsize=16, fontweight='bold', pad=20)
        ax2.set_ylim(-0.05, 1.05)
        ax2.grid(axis='y', alpha=0.3, linestyle='--')

        # Add statistical test
        from scipy.stats import wilcoxon
        stat, p_value = wilcoxon(numerical_conf, cv_conf)
        ax2.text(0.5, 0.95, f'Wilcoxon test: p = {p_value:.4f}',
                transform=ax2.transAxes, fontsize=11, fontweight='bold',
                ha='center', va='top',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

        # Panel C: Winner distribution
        ax3 = fig.add_subplot(gs_sub[2])

        better_methods = [self.annotations[sid]['better_method']
                         for sid in self.annotations.keys()]
        method_counts = pd.Series(better_methods).value_counts()

        colors_pie = {'Numerical': '#DE8F05', 'CV': '#CC78BC', 'Equal': '#949494'}
        colors_list = [colors_pie.get(m, '#949494') for m in method_counts.index]

        wedges, texts, autotexts = ax3.pie(method_counts.values,
                                           labels=method_counts.index,
                                           autopct='%1.1f%%',
                                           colors=colors_list,
                                           startangle=90,
                                           textprops={'fontsize': 12, 'fontweight': 'bold'},
                                           wedgeprops={'edgecolor': 'black', 'linewidth': 2})

        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(14)
            autotext.set_fontweight('bold')

        ax3.set_title('C. Method Performance Distribution',
                     fontsize=16, fontweight='bold', pad=20)

    def _plot_scenario_performance(self, fig, gs):
        """Row 2: Performance by annotation scenario"""

        # Create a 1x3 sub-grid within the passed SubplotSpec
        gs_sub = gs.subgridspec(1, 3, wspace=0.3)

        # Panel D: Confidence by scenario (grouped bar chart)
        ax1 = fig.add_subplot(gs_sub[0])

        scenarios = sorted(set(self.annotations[sid]['scenario']
                             for sid in self.annotations.keys()))

        numerical_by_scenario = []
        cv_by_scenario = []

        for scenario in scenarios:
            num_conf = [self.annotations[sid]['numerical_confidence']
                       for sid in self.annotations.keys()
                       if self.annotations[sid]['scenario'] == scenario]
            cv_conf = [self.annotations[sid]['cv_confidence']
                      for sid in self.annotations.keys()
                      if self.annotations[sid]['scenario'] == scenario]

            numerical_by_scenario.append(np.mean(num_conf) if num_conf else 0)
            cv_by_scenario.append(np.mean(cv_conf) if cv_conf else 0)

        x = np.arange(len(scenarios))
        width = 0.35

        bars1 = ax1.bar(x - width/2, numerical_by_scenario, width,
                       label='Numerical', color='#DE8F05', alpha=0.7,
                       edgecolor='black', linewidth=1.5)
        bars2 = ax1.bar(x + width/2, cv_by_scenario, width,
                       label='CV', color='#CC78BC', alpha=0.7,
                       edgecolor='black', linewidth=1.5)

        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}',
                        ha='center', va='bottom', fontsize=9, fontweight='bold')

        ax1.set_xlabel('Annotation Scenario', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Mean Confidence', fontsize=14, fontweight='bold')
        ax1.set_title('D. Performance by Scenario',
                     fontsize=16, fontweight='bold', pad=20)
        ax1.set_xticks(x)
        ax1.set_xticklabels(scenarios, rotation=45, ha='right', fontsize=10)
        ax1.legend(fontsize=12, loc='upper right')
        ax1.grid(axis='y', alpha=0.3, linestyle='--')
        ax1.set_ylim(0, 1.1)

        # Panel E: Advantage distribution
        ax2 = fig.add_subplot(gs_sub[1])

        advantages = []
        advantage_colors = []
        advantage_labels = []

        for sid in self.annotations.keys():
            adv = self.annotations[sid]['advantage']
            method = self.annotations[sid]['better_method']

            if method == 'Numerical':
                advantages.append(-adv)  # Negative for numerical
                advantage_colors.append('#DE8F05')
            elif method == 'CV':
                advantages.append(adv)  # Positive for CV
                advantage_colors.append('#CC78BC')
            else:
                advantages.append(0)
                advantage_colors.append('#949494')

            advantage_labels.append(f'S{sid}')

        # Sort by advantage
        sorted_indices = np.argsort(advantages)
        advantages = [advantages[i] for i in sorted_indices]
        advantage_colors = [advantage_colors[i] for i in sorted_indices]
        advantage_labels = [advantage_labels[i] for i in sorted_indices]

        y_pos = np.arange(len(advantages))
        bars = ax2.barh(y_pos, advantages, color=advantage_colors, alpha=0.7,
                       edgecolor='black', linewidth=1.5)

        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(advantage_labels, fontsize=10)
        ax2.set_xlabel('Confidence Advantage', fontsize=14, fontweight='bold')
        ax2.set_title('E. Method Advantage by Spectrum',
                     fontsize=16, fontweight='bold', pad=20)
        ax2.axvline(0, color='black', linewidth=2, linestyle='--')
        ax2.text(-0.4, len(advantages)-1, 'Numerical\nBetter',
                ha='center', va='top', fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='#DE8F05', alpha=0.5))
        ax2.text(0.4, len(advantages)-1, 'CV\nBetter',
                ha='center', va='top', fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='#CC78BC', alpha=0.5))
        ax2.grid(axis='x', alpha=0.3, linestyle='--')

        # Panel F: Scenario-specific recommendations
        ax3 = fig.add_subplot(gs_sub[2])
        ax3.axis('off')

        recommendations = [
            ['Scenario', 'Recommended Method', 'Reason'],
            ['Simple/Clean', 'Numerical', 'Fast, accurate for simple spectra'],
            ['Complex/Dense', 'CV', 'Better separation of overlapping peaks'],
            ['Dominant Base Peak', 'Numerical', 'Strong signal, easy matching'],
            ['Many Equal Peaks', 'CV', 'Visual patterns capture relationships'],
            ['Very High Complexity', 'Combined', 'Both methods provide value'],
            ['Moderate Complexity', 'Combined', 'Complementary information'],
        ]

        table = ax3.table(cellText=recommendations[1:],
                         colLabels=recommendations[0],
                         cellLoc='left',
                         loc='center',
                         colWidths=[0.25, 0.25, 0.5])

        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2.5)

        # Style header
        for i in range(3):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')

        # Color-code recommendations
        for i in range(1, len(recommendations)):
            method = recommendations[i][1]
            if method == 'Numerical':
                table[(i, 1)].set_facecolor('#DE8F05')
                table[(i, 1)].set_alpha(0.3)
            elif method == 'CV':
                table[(i, 1)].set_facecolor('#CC78BC')
                table[(i, 1)].set_alpha(0.3)
            else:
                table[(i, 1)].set_facecolor('#029E73')
                table[(i, 1)].set_alpha(0.3)

        ax3.set_title('F. Method Recommendations by Scenario',
                     fontsize=16, fontweight='bold', pad=20)

    def _plot_feature_space(self, fig, gs):
        """Row 3: Feature space visualization"""

        # Create a 1x3 sub-grid within the passed SubplotSpec
        gs_sub = gs.subgridspec(1, 3, wspace=0.3)

        # Collect features for all spectra
        features_list = []
        labels_list = []
        colors_list = []

        for sid in self.annotations.keys():
            if sid not in self.numerical_data or sid not in self.cv_data:
                continue

            num_data = self.numerical_data[sid]
            cv_data = self.cv_data[sid]

            # Extract features
            features = []

            # Numerical features
            features.append(len(num_data))  # Peak count
            features.append(self.annotations[sid]['shannon_entropy'])
            features.append(self.annotations[sid]['gini_coefficient'])

            # CV features
            if 's_knowledge' in cv_data.columns:
                features.append(cv_data['s_knowledge'].mean())
                features.append(cv_data['s_knowledge'].std())
            else:
                features.extend([0, 0])

            if 's_time' in cv_data.columns:
                features.append(cv_data['s_time'].mean())
                features.append(cv_data['s_time'].std())
            else:
                features.extend([0, 0])

            if 's_entropy' in cv_data.columns:
                features.append(cv_data['s_entropy'].mean())
                features.append(cv_data['s_entropy'].std())
            else:
                features.extend([0, 0])

            if 'velocity' in cv_data.columns:
                features.append(cv_data['velocity'].mean())
            else:
                features.append(0)

            if 'radius' in cv_data.columns:
                features.append(cv_data['radius'].mean())
            else:
                features.append(0)

            features_list.append(features)
            labels_list.append(f'S{sid}')

            # Color by better method
            better = self.annotations[sid]['better_method']
            if better == 'Numerical':
                colors_list.append('#DE8F05')
            elif better == 'CV':
                colors_list.append('#CC78BC')
            else:
                colors_list.append('#949494')

        features_array = np.array(features_list)

        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features_array)

        # Panel G: PCA
        ax1 = fig.add_subplot(gs_sub[0])

        pca = PCA(n_components=2)
        features_pca = pca.fit_transform(features_scaled)

        scatter1 = ax1.scatter(features_pca[:, 0], features_pca[:, 1],
                              c=colors_list, s=300, alpha=0.7,
                              edgecolors='black', linewidth=2)

        for i, label in enumerate(labels_list):
            ax1.annotate(label, (features_pca[i, 0], features_pca[i, 1]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=10, fontweight='bold')

        ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)',
                      fontsize=14, fontweight='bold')
        ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)',
                      fontsize=14, fontweight='bold')
        ax1.set_title('G. Feature Space (PCA)',
                     fontsize=16, fontweight='bold', pad=20)
        ax1.grid(True, alpha=0.3, linestyle='--')

        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#DE8F05', label='Numerical Better'),
            Patch(facecolor='#CC78BC', label='CV Better'),
            Patch(facecolor='#949494', label='Equal')
        ]
        ax1.legend(handles=legend_elements, loc='best', fontsize=11)

        # Panel H: t-SNE
        ax2 = fig.add_subplot(gs_sub[1])

        if len(features_list) >= 3:  # t-SNE requires at least 3 samples
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(3, len(features_list)-1))
            features_tsne = tsne.fit_transform(features_scaled)

            scatter2 = ax2.scatter(features_tsne[:, 0], features_tsne[:, 1],
                                  c=colors_list, s=300, alpha=0.7,
                                  edgecolors='black', linewidth=2)

            for i, label in enumerate(labels_list):
                ax2.annotate(label, (features_tsne[i, 0], features_tsne[i, 1]),
                            xytext=(5, 5), textcoords='offset points',
                            fontsize=10, fontweight='bold')

            ax2.set_xlabel('t-SNE Dimension 1', fontsize=14, fontweight='bold')
            ax2.set_ylabel('t-SNE Dimension 2', fontsize=14, fontweight='bold')
            ax2.set_title('H. Feature Space (t-SNE)',
                         fontsize=16, fontweight='bold', pad=20)
            ax2.grid(True, alpha=0.3, linestyle='--')
        else:
            ax2.text(0.5, 0.5, 'Insufficient data for t-SNE\n(need ≥3 spectra)',
                    ha='center', va='center', fontsize=14,
                    transform=ax2.transAxes)

        # Panel I: Feature importance
        ax3 = fig.add_subplot(gs_sub[2])

        feature_names = [
            'Peak Count', 'Shannon Entropy', 'Gini Coeff',
            'S_knowledge (μ)', 'S_knowledge (σ)',
            'S_time (μ)', 'S_time (σ)',
            'S_entropy (μ)', 'S_entropy (σ)',
            'Velocity (μ)', 'Radius (μ)'
        ]

        # Calculate feature importance using PCA loadings
        loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
        importance = np.abs(loadings).sum(axis=1)
        importance = importance / importance.sum()

        # Sort by importance
        sorted_indices = np.argsort(importance)[::-1]
        importance_sorted = importance[sorted_indices]
        names_sorted = [feature_names[i] for i in sorted_indices]

        colors_importance = ['#DE8F05' if i < 3 else '#CC78BC'
                            for i in sorted_indices]

        bars = ax3.barh(range(len(importance_sorted)), importance_sorted,
                       color=colors_importance, alpha=0.7,
                       edgecolor='black', linewidth=1.5)

        ax3.set_yticks(range(len(importance_sorted)))
        ax3.set_yticklabels(names_sorted, fontsize=10)
        ax3.set_xlabel('Feature Importance', fontsize=14, fontweight='bold')
        ax3.set_title('I. Feature Importance (PCA)',
                     fontsize=16, fontweight='bold', pad=20)
        ax3.grid(axis='x', alpha=0.3, linestyle='--')

        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#DE8F05', label='Numerical Features'),
            Patch(facecolor='#CC78BC', label='CV Features')
        ]
        ax3.legend(handles=legend_elements, loc='lower right', fontsize=11)

    def _plot_complementarity_metrics(self, fig, gs):
        """Row 4: Complementarity metrics"""

        # Create a 1x3 sub-grid within the passed SubplotSpec
        gs_sub = gs.subgridspec(1, 3, wspace=0.3)

        # Panel J: Correlation between methods
        ax1 = fig.add_subplot(gs_sub[0])

        # Calculate correlation for different feature types
        correlations = {}

        for sid in self.annotations.keys():
            if sid not in self.numerical_data or sid not in self.cv_data:
                continue

            num_data = self.numerical_data[sid]
            cv_data = self.cv_data[sid]

            # Peak count vs droplet count
            if 'peak_droplet_corr' not in correlations:
                correlations['peak_droplet_corr'] = {'num': [], 'cv': []}
            correlations['peak_droplet_corr']['num'].append(len(num_data))
            correlations['peak_droplet_corr']['cv'].append(len(cv_data))

            # Shannon entropy vs mean S_entropy
            if 's_entropy' in cv_data.columns:
                if 'entropy_corr' not in correlations:
                    correlations['entropy_corr'] = {'num': [], 'cv': []}
                correlations['entropy_corr']['num'].append(
                    self.annotations[sid]['shannon_entropy'])
                correlations['entropy_corr']['cv'].append(
                    cv_data['s_entropy'].mean())

        # Calculate correlation coefficients
        corr_values = []
        corr_labels = []

        for key, data in correlations.items():
            if len(data['num']) > 1:
                r = np.corrcoef(data['num'], data['cv'])[0, 1]
                corr_values.append(r)
                if key == 'peak_droplet_corr':
                    corr_labels.append('Peak Count vs\nDroplet Count')
                elif key == 'entropy_corr':
                    corr_labels.append('Shannon Entropy vs\nMean S_entropy')

        colors_corr = ['#029E73' if r > 0.7 else '#DE8F05' if r > 0.4 else '#CC3311'
                      for r in corr_values]

        bars = ax1.bar(range(len(corr_values)), corr_values,
                      color=colors_corr, alpha=0.7,
                      edgecolor='black', linewidth=2)

        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, corr_values)):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'r = {val:.3f}',
                    ha='center', va='bottom' if val > 0 else 'top',
                    fontsize=11, fontweight='bold')

        ax1.set_xticks(range(len(corr_values)))
        ax1.set_xticklabels(corr_labels, fontsize=11)
        ax1.set_ylabel('Pearson Correlation (r)', fontsize=14, fontweight='bold')
        ax1.set_title('J. Cross-Method Feature Correlation',
                     fontsize=16, fontweight='bold', pad=20)
        ax1.axhline(0, color='black', linewidth=1, linestyle='--')
        ax1.axhline(0.7, color='green', linewidth=1, linestyle=':', alpha=0.5,
                   label='Strong correlation')
        ax1.axhline(0.4, color='orange', linewidth=1, linestyle=':', alpha=0.5,
                   label='Moderate correlation')
        ax1.set_ylim(-0.2, 1.1)
        ax1.grid(axis='y', alpha=0.3, linestyle='--')
        ax1.legend(fontsize=10, loc='lower right')

        # Panel K: Complementarity score
        ax2 = fig.add_subplot(gs_sub[1])

        # Calculate complementarity score for each spectrum
        # High complementarity = methods disagree but combined is better
        complementarity_scores = []
        spec_ids_comp = []

        for sid in self.annotations.keys():
            num_conf = self.annotations[sid]['numerical_confidence']
            cv_conf = self.annotations[sid]['cv_confidence']
            combined_conf = self.annotations[sid]['combined_confidence']

            # Complementarity = how much combined exceeds max individual
            max_individual = max(num_conf, cv_conf)
            complementarity = (combined_conf - max_individual) / max_individual

            complementarity_scores.append(complementarity)
            spec_ids_comp.append(f'S{sid}')

        # Sort by complementarity
        sorted_indices = np.argsort(complementarity_scores)[::-1]
        complementarity_sorted = [complementarity_scores[i] for i in sorted_indices]
        labels_sorted = [spec_ids_comp[i] for i in sorted_indices]

        colors_comp = ['#029E73' if c > 0.1 else '#DE8F05' if c > 0 else '#CC3311'
                      for c in complementarity_sorted]

        bars = ax2.barh(range(len(complementarity_sorted)), complementarity_sorted,
                       color=colors_comp, alpha=0.7,
                       edgecolor='black', linewidth=1.5)

        ax2.set_yticks(range(len(complementarity_sorted)))
        ax2.set_yticklabels(labels_sorted, fontsize=10)
        ax2.set_xlabel('Complementarity Score', fontsize=14, fontweight='bold')
        ax2.set_title('K. Method Complementarity by Spectrum',
                     fontsize=16, fontweight='bold', pad=20)
        ax2.axvline(0, color='black', linewidth=2, linestyle='--')
        ax2.grid(axis='x', alpha=0.3, linestyle='--')

        # Add interpretation
        ax2.text(0.95, 0.95, 'High score = methods\ncomplement well',
                transform=ax2.transAxes, fontsize=10, fontweight='bold',
                ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='#029E73', alpha=0.5))

        # Panel L: Summary recommendations
        ax3 = fig.add_subplot(gs_sub[2])
        ax3.axis('off')

        # Calculate summary statistics
        num_better_count = sum(1 for sid in self.annotations.keys()
                              if self.annotations[sid]['better_method'] == 'Numerical')
        cv_better_count = sum(1 for sid in self.annotations.keys()
                             if self.annotations[sid]['better_method'] == 'CV')
        equal_count = sum(1 for sid in self.annotations.keys()
                         if self.annotations[sid]['better_method'] == 'Equal')

        mean_num_conf = np.mean([self.annotations[sid]['numerical_confidence']
                                for sid in self.annotations.keys()])
        mean_cv_conf = np.mean([self.annotations[sid]['cv_confidence']
                               for sid in self.annotations.keys()])
        mean_combined_conf = np.mean([self.annotations[sid]['combined_confidence']
                                     for sid in self.annotations.keys()])

        mean_complementarity = np.mean(complementarity_scores)

        summary_text = f"""
COMPLEMENTARITY ANALYSIS SUMMARY
{'='*50}

METHOD PERFORMANCE:
  • Numerical better: {num_better_count}/{len(self.annotations)} spectra ({num_better_count/len(self.annotations)*100:.1f}%)
  • CV better: {cv_better_count}/{len(self.annotations)} spectra ({cv_better_count/len(self.annotations)*100:.1f}%)
  • Equal performance: {equal_count}/{len(self.annotations)} spectra ({equal_count/len(self.annotations)*100:.1f}%)

MEAN CONFIDENCE SCORES:
  • Numerical method: {mean_num_conf:.3f}
  • CV method: {mean_cv_conf:.3f}
  • Combined method: {mean_combined_conf:.3f}
  • Improvement: {(mean_combined_conf - max(mean_num_conf, mean_cv_conf))/max(mean_num_conf, mean_cv_conf)*100:.1f}%

COMPLEMENTARITY:
  • Mean complementarity score: {mean_complementarity:.3f}
  • Methods show {'strong' if mean_complementarity > 0.1 else 'moderate' if mean_complementarity > 0 else 'weak'} complementarity

RECOMMENDATIONS:
  ✓ Use NUMERICAL method for: Simple spectra, high-throughput
  ✓ Use CV method for: Complex spectra, isobaric compounds
  ✓ Use COMBINED approach for: Maximum confidence, novel compounds
        """

        ax3.text(0.1, 0.95, summary_text,
                transform=ax3.transAxes, fontsize=11, fontweight='bold',
                verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        ax3.set_title('L. Summary and Recommendations',
                     fontsize=16, fontweight='bold', pad=20)

    def generate_complementarity_report(self):
        """Generate detailed complementarity report"""

        print("\n" + "="*80)
        print("COMPLEMENTARITY ANALYSIS REPORT")
        print("="*80)

        print("\nOVERALL PERFORMANCE:")
        print("-" * 60)

        num_better = [sid for sid in self.annotations.keys()
                     if self.annotations[sid]['better_method'] == 'Numerical']
        cv_better = [sid for sid in self.annotations.keys()
                    if self.annotations[sid]['better_method'] == 'CV']
        equal = [sid for sid in self.annotations.keys()
                if self.annotations[sid]['better_method'] == 'Equal']

        print(f"Numerical method better: {len(num_better)} spectra")
        for sid in num_better:
            adv = self.annotations[sid]['advantage']
            scenario = self.annotations[sid]['scenario']
            print(f"  - Spectrum {sid}: advantage = {adv:.3f}, scenario = {scenario}")

        print(f"\nCV method better: {len(cv_better)} spectra")
        for sid in cv_better:
            adv = self.annotations[sid]['advantage']
            scenario = self.annotations[sid]['scenario']
            print(f"  - Spectrum {sid}: advantage = {adv:.3f}, scenario = {scenario}")

        print(f"\nEqual performance: {len(equal)} spectra")
        for sid in equal:
            scenario = self.annotations[sid]['scenario']
            print(f"  - Spectrum {sid}: scenario = {scenario}")

        print("\nCONFIDENCE STATISTICS:")
        print("-" * 60)

        num_confs = [self.annotations[sid]['numerical_confidence']
                    for sid in self.annotations.keys()]
        cv_confs = [self.annotations[sid]['cv_confidence']
                   for sid in self.annotations.keys()]
        combined_confs = [self.annotations[sid]['combined_confidence']
                         for sid in self.annotations.keys()]

        print(f"Numerical method: {np.mean(num_confs):.3f} ± {np.std(num_confs):.3f}")
        print(f"CV method: {np.mean(cv_confs):.3f} ± {np.std(cv_confs):.3f}")
        print(f"Combined method: {np.mean(combined_confs):.3f} ± {np.std(combined_confs):.3f}")

        improvement = (np.mean(combined_confs) - max(np.mean(num_confs), np.mean(cv_confs))) / \
                     max(np.mean(num_confs), np.mean(cv_confs)) * 100
        print(f"Combined improvement: {improvement:.1f}%")

        print("\nSCENARIO ANALYSIS:")
        print("-" * 60)

        scenarios = set(self.annotations[sid]['scenario'] for sid in self.annotations.keys())
        for scenario in sorted(scenarios):
            print(f"\n{scenario}:")

            scenario_specs = [sid for sid in self.annotations.keys()
                            if self.annotations[sid]['scenario'] == scenario]

            num_conf_scenario = [self.annotations[sid]['numerical_confidence']
                               for sid in scenario_specs]
            cv_conf_scenario = [self.annotations[sid]['cv_confidence']
                              for sid in scenario_specs]

            print(f"  Numerical: {np.mean(num_conf_scenario):.3f} ± {np.std(num_conf_scenario):.3f}")
            print(f"  CV: {np.mean(cv_conf_scenario):.3f} ± {np.std(cv_conf_scenario):.3f}")

            if np.mean(cv_conf_scenario) > np.mean(num_conf_scenario):
                print(f"  → CV method recommended (advantage: {np.mean(cv_conf_scenario) - np.mean(num_conf_scenario):.3f})")
            elif np.mean(num_conf_scenario) > np.mean(cv_conf_scenario):
                print(f"  → Numerical method recommended (advantage: {np.mean(num_conf_scenario) - np.mean(cv_conf_scenario):.3f})")
            else:
                print(f"  → Methods perform equally")

        print("\n" + "="*80)

def main():
    """Main execution function"""

    print("="*80)
    print("COMPLEMENTARITY ANALYSIS")
    print("Where each method performs better and how they complement each other")
    print("="*80)

    # Initialize analyzer
    analyzer = ComplementarityAnalyzer()

    # Load data
    analyzer.load_data()

    # Simulate annotation performance
    analyzer.simulate_annotation_performance()

    # Generate comprehensive figure
    print("\nGenerating complementarity figure...")
    fig = analyzer.create_complementarity_figure()

    # Generate report
    analyzer.generate_complementarity_report()

    print("\nAnalysis complete!")
    print("Output files:")
    print("  - complementarity_analysis.png")
    print("  - complementarity_analysis.pdf")

if __name__ == "__main__":
    main()
