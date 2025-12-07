#!/usr/bin/env python3
"""
molecular_language_atlas.py

Creates comprehensive visual atlas of molecular language structure.
Minimal text, maximum information density.

Usage:
    python molecular_language_atlas.py

Author: Kundai Farai Sachikonye (with AI assistance)
Date: November 2025
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle, Rectangle, FancyBboxPatch, Wedge
from matplotlib.collections import PatchCollection, LineCollection
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path

# Setup paths
SCRIPT_DIR = Path(__file__).resolve().parent
PRECURSOR_ROOT = SCRIPT_DIR.parent.parent
RESULTS_DIR = PRECURSOR_ROOT / "results" / "tests" / "molecular_language"
SEQ_RESULTS_DIR = PRECURSOR_ROOT / "results" / "tests" / "sequence"

# Set publication-quality defaults
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 8
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['xtick.major.width'] = 1.0
plt.rcParams['ytick.major.width'] = 1.0

# Custom color palettes
ENTROPY_CMAP = sns.color_palette("rocket", as_cmap=True)
KNOWLEDGE_CMAP = sns.color_palette("mako", as_cmap=True)
TIME_CMAP = sns.color_palette("viridis", as_cmap=True)


class MolecularLanguageAtlas:
    """
    Creates comprehensive visual atlas of molecular language.
    """

    def __init__(self, output_dir=None):
        if output_dir is None:
            output_dir = PRECURSOR_ROOT / 'results' / 'visualizations' / 'molecular_language'
        self.output_dir = Path(output_dir) if isinstance(output_dir, str) else output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load data
        print(f"  Loading from: {RESULTS_DIR}")
        self.aa_alphabet = pd.read_csv(RESULTS_DIR / 'amino_acid_alphabet.csv')
        self.fragmentation = pd.read_csv(RESULTS_DIR / 'fragmentation_grammar.csv')
        self.sequences = pd.read_csv(RESULTS_DIR / 'sequence_sentropy_paths.csv')

        print("✓ Data loaded")
        print(f"  - Amino acids: {len(self.aa_alphabet)}")
        print(f"  - Fragments: {len(self.fragmentation)}")
        print(f"  - Sequence positions: {len(self.sequences)}")

    def create_master_atlas(self):
        """
        Create master atlas figure (single comprehensive visualization).
        """
        print("\nCreating Molecular Language Atlas...")

        # Create figure with custom layout
        fig = plt.figure(figsize=(20, 12))

        # Define grid
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3,
                             left=0.05, right=0.98, top=0.95, bottom=0.05)

        # Panel A: Amino Acid S-Entropy Space (3D → 2D projection)
        ax_aa = fig.add_subplot(gs[0, 0])
        self._plot_amino_acid_space(ax_aa)
        self._add_panel_label(ax_aa, 'A')

        # Panel B: Amino Acid Periodic Table (organized by properties)
        ax_periodic = fig.add_subplot(gs[0, 1])
        self._plot_amino_acid_periodic_table(ax_periodic)
        self._add_panel_label(ax_periodic, 'B')

        # Panel C: Fragmentation Grammar Network
        ax_grammar = fig.add_subplot(gs[0, 2:])
        self._plot_fragmentation_grammar(ax_grammar)
        self._add_panel_label(ax_grammar, 'C')

        # Panel D: Sequence S-Entropy Trajectories
        ax_traj = fig.add_subplot(gs[1, :2])
        self._plot_sequence_trajectories(ax_traj)
        self._add_panel_label(ax_traj, 'D')

        # Panel E: Complexity Landscape
        ax_complex = fig.add_subplot(gs[1, 2:])
        self._plot_complexity_landscape(ax_complex)
        self._add_panel_label(ax_complex, 'E')

        # Panel F: Amino Acid Similarity Matrix
        ax_sim = fig.add_subplot(gs[2, 0])
        self._plot_similarity_matrix(ax_sim)
        self._add_panel_label(ax_sim, 'F')

        # Panel G: Fragment Type Distribution (polar plot)
        ax_frag = fig.add_subplot(gs[2, 1], projection='polar')
        self._plot_fragment_distribution(ax_frag)
        self._add_panel_label(ax_frag, 'G')

        # Panel H: S-Entropy Correlation Network
        ax_corr = fig.add_subplot(gs[2, 2:])
        self._plot_sentropy_correlation_network(ax_corr)
        self._add_panel_label(ax_corr, 'H')

        # Save
        output_path = self.output_dir / 'molecular_language_atlas.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved: {output_path}")
        plt.close()

    def _plot_amino_acid_space(self, ax):
        """
        Panel A: Amino acids in S-entropy space (3D → 2D via t-SNE).
        Uses symbols and colors instead of text.
        """
        # Extract S-entropy coordinates
        coords = self.aa_alphabet[['s_knowledge', 's_time', 's_entropy']].values

        # t-SNE projection
        if len(coords) > 3:
            from sklearn.manifold import TSNE
            tsne = TSNE(n_components=2, perplexity=5, random_state=42)
            coords_2d = tsne.fit_transform(coords)
        else:
            coords_2d = coords[:, :2]

        # Color by hydrophobicity
        colors = self.aa_alphabet['hydrophobicity'].values

        # Size by mass
        sizes = (self.aa_alphabet['mass'].values / self.aa_alphabet['mass'].max()) * 500

        # Shape by charge
        charges = self.aa_alphabet['charge'].values

        # Plot with different markers for different charges
        for charge in np.unique(charges):
            mask = charges == charge

            if charge > 0:
                marker = '^'  # Triangle up for positive
            elif charge < 0:
                marker = 'v'  # Triangle down for negative
            else:
                marker = 'o'  # Circle for neutral

            scatter = ax.scatter(coords_2d[mask, 0], coords_2d[mask, 1],
                               c=colors[mask], s=sizes[mask],
                               marker=marker, alpha=0.7,
                               cmap='RdYlBu_r', vmin=-5, vmax=5,
                               edgecolors='black', linewidths=1.5)

            # Add one-letter code as annotation (minimal text)
            for i, (x, y) in enumerate(coords_2d[mask]):
                aa_idx = np.where(mask)[0][i]
                aa_code = self.aa_alphabet.iloc[aa_idx]['symbol']
                ax.text(x, y, aa_code, ha='center', va='center',
                       fontsize=6, fontweight='bold', color='white')

        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Hydrophobicity', fontsize=7)
        cbar.ax.tick_params(labelsize=6)

        # Legend (symbols only)
        legend_elements = [
            plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='gray',
                      markersize=8, label='+ charge'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
                      markersize=8, label='neutral'),
            plt.Line2D([0], [0], marker='v', color='w', markerfacecolor='gray',
                      markersize=8, label='− charge')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=6, framealpha=0.9)

        ax.set_xlabel('S-Entropy Dimension 1', fontsize=8)
        ax.set_ylabel('S-Entropy Dimension 2', fontsize=8)
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.3)

    def _plot_amino_acid_periodic_table(self, ax):
        """
        Panel B: Amino acid periodic table organized by S-entropy properties.
        Visual grid with color-coded cells.
        """
        # Sort by s_knowledge and s_time
        aa_sorted = self.aa_alphabet.sort_values(['s_knowledge', 's_time'])

        # Create grid layout (5x4 for 20 amino acids)
        n_rows, n_cols = 5, 4

        for idx, (_, aa) in enumerate(aa_sorted.iterrows()):
            row = idx // n_cols
            col = idx % n_cols

            # Cell position
            x, y = col, n_rows - row - 1

            # Color by hydrophobicity
            hydro = aa['hydrophobicity']
            color = plt.cm.RdYlBu_r((hydro + 5) / 10)  # Normalize to [0, 1]

            # Draw cell
            rect = FancyBboxPatch((x, y), 1, 1, boxstyle="round,pad=0.05",
                                 facecolor=color, edgecolor='black', linewidth=1.5)
            ax.add_patch(rect)

            # Add amino acid symbol (large)
            ax.text(x + 0.5, y + 0.6, aa['symbol'],
                   ha='center', va='center', fontsize=14, fontweight='bold',
                   color='white' if abs(hydro) > 2 else 'black')

            # Add mass (small, bottom)
            ax.text(x + 0.5, y + 0.2, f"{aa['mass']:.0f}",
                   ha='center', va='center', fontsize=6,
                   color='white' if abs(hydro) > 2 else 'black', alpha=0.8)

            # Add charge indicator (top right corner)
            if aa['charge'] > 0:
                ax.text(x + 0.85, y + 0.85, '+', ha='center', va='center',
                       fontsize=10, fontweight='bold', color='red')
            elif aa['charge'] < 0:
                ax.text(x + 0.85, y + 0.85, '−', ha='center', va='center',
                       fontsize=10, fontweight='bold', color='blue')

        ax.set_xlim(0, n_cols)
        ax.set_ylim(0, n_rows)
        ax.set_aspect('equal')
        ax.axis('off')

        # Add colorbar legend
        sm = plt.cm.ScalarMappable(cmap='RdYlBu_r',
                                   norm=plt.Normalize(vmin=-5, vmax=5))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04, orientation='horizontal')
        cbar.set_label('Hydrophobicity', fontsize=7)
        cbar.ax.tick_params(labelsize=6)

    def _plot_fragmentation_grammar(self, ax):
        """
        Panel C: Fragmentation grammar as network diagram.
        Nodes = fragment types, edges = transitions.
        """
        # Get unique parent sequences
        sequences = self.fragmentation['parent_sequence'].unique()

        # For each sequence, plot fragmentation pattern
        y_positions = np.linspace(0, 1, len(sequences))

        for seq_idx, seq in enumerate(sequences):
            seq_frags = self.fragmentation[self.fragmentation['parent_sequence'] == seq]

            y = y_positions[seq_idx]

            # Plot b-ions and y-ions separately
            b_ions = seq_frags[seq_frags['ion_type'] == 'b']
            y_ions = seq_frags[seq_frags['ion_type'] == 'y']

            # b-ions (blue, top)
            if len(b_ions) > 0:
                x_b = b_ions['position'].values / len(seq)
                sizes_b = (b_ions['mz'].values / b_ions['mz'].max()) * 200

                ax.scatter(x_b, [y + 0.1] * len(x_b), s=sizes_b,
                          color='steelblue', alpha=0.7, marker='^',
                          edgecolors='black', linewidths=0.5)

                # Connect with lines
                if len(x_b) > 1:
                    ax.plot(x_b, [y + 0.1] * len(x_b), 'steelblue',
                           alpha=0.3, linewidth=1)

            # y-ions (red, bottom)
            if len(y_ions) > 0:
                x_y = y_ions['position'].values / len(seq)
                sizes_y = (y_ions['mz'].values / y_ions['mz'].max()) * 200

                ax.scatter(x_y, [y - 0.1] * len(x_y), s=sizes_y,
                          color='crimson', alpha=0.7, marker='v',
                          edgecolors='black', linewidths=0.5)

                # Connect with lines
                if len(x_y) > 1:
                    ax.plot(x_y, [y - 0.1] * len(x_y), 'crimson',
                           alpha=0.3, linewidth=1)

            # Add sequence label (minimal)
            ax.text(-0.05, y, seq, ha='right', va='center',
                   fontsize=7, fontfamily='monospace')

        # Legend
        legend_elements = [
            plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='steelblue',
                      markersize=8, label='b-ions'),
            plt.Line2D([0], [0], marker='v', color='w', markerfacecolor='crimson',
                      markersize=8, label='y-ions')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=7, framealpha=0.9)

        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.2, 1.2)
        ax.set_xlabel('Relative Position in Sequence', fontsize=8)
        ax.set_ylabel('Peptide', fontsize=8)
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.3, axis='x')

    def _plot_sequence_trajectories(self, ax):
        """
        Panel D: S-entropy trajectories through sequences.
        Each sequence is a path in S-entropy space.
        """
        sequences = self.sequences['sequence'].unique()

        # Color palette
        colors = sns.color_palette('husl', len(sequences))

        for seq_idx, seq in enumerate(sequences):
            seq_data = self.sequences[self.sequences['sequence'] == seq].sort_values('position')

            positions = seq_data['position'].values
            s_knowledge = seq_data['s_knowledge'].values
            s_time = seq_data['s_time'].values
            s_entropy = seq_data['s_entropy'].values

            # Plot trajectory in 3D projected to 2D
            # Use s_knowledge vs s_time as axes, color by s_entropy

            # Plot line
            points = np.array([s_knowledge, s_time]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)

            lc = LineCollection(segments, cmap='viridis', alpha=0.7, linewidth=2)
            lc.set_array(s_entropy)
            lc.set_clim(0, 1)
            ax.add_collection(lc)

            # Plot points
            scatter = ax.scatter(s_knowledge, s_time, c=s_entropy,
                               cmap='viridis', s=100, alpha=0.8,
                               edgecolors='black', linewidths=1,
                               vmin=0, vmax=1, zorder=10)

            # Add amino acid labels
            for i, (x, y, aa) in enumerate(zip(s_knowledge, s_time,
                                               seq_data['amino_acid'].values)):
                ax.text(x, y, aa, ha='center', va='center',
                       fontsize=6, fontweight='bold', color='white', zorder=11)

            # Add sequence label at start
            ax.text(s_knowledge[0], s_time[0] - 0.05, seq,
                   ha='center', va='top', fontsize=7,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor=colors[seq_idx], alpha=0.7))

        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('S-Entropy', fontsize=8)
        cbar.ax.tick_params(labelsize=6)

        ax.set_xlabel('S-Knowledge', fontsize=8)
        ax.set_ylabel('S-Time', fontsize=8)
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.3)

    def _plot_complexity_landscape(self, ax):
        """
        Panel E: Sequence complexity landscape.
        Heatmap of complexity vs entropy.
        """
        # Get unique sequences
        seq_summary = self.sequences.groupby('sequence').first()

        # Create 2D histogram
        x = seq_summary['sequence_entropy'].values
        y = seq_summary['sequence_complexity'].values

        # Hexbin plot
        hexbin = ax.hexbin(x, y, gridsize=15, cmap='YlOrRd', mincnt=1,
                          edgecolors='black', linewidths=0.5)

        # Add sequence labels at their positions
        for seq, row in seq_summary.iterrows():
            ax.text(row['sequence_entropy'], row['sequence_complexity'], seq,
                   ha='center', va='center', fontsize=7, fontweight='bold',
                   color='white', bbox=dict(boxstyle='round,pad=0.2',
                                           facecolor='black', alpha=0.5))

        # Colorbar
        cbar = plt.colorbar(hexbin, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Density', fontsize=8)
        cbar.ax.tick_params(labelsize=6)

        ax.set_xlabel('Sequence Entropy', fontsize=8)
        ax.set_ylabel('Sequence Complexity', fontsize=8)
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.3)

    def _plot_similarity_matrix(self, ax):
        """
        Panel F: Amino acid similarity matrix based on S-entropy distance.
        """
        # Compute pairwise distances in S-entropy space
        coords = self.aa_alphabet[['s_knowledge', 's_time', 's_entropy']].values
        dist_matrix = squareform(pdist(coords, metric='euclidean'))

        # Convert to similarity (inverse distance)
        similarity = 1 / (1 + dist_matrix)

        # Plot heatmap
        im = ax.imshow(similarity, cmap='magma', aspect='auto')

        # Add amino acid labels
        aa_codes = self.aa_alphabet['symbol'].values
        ax.set_xticks(range(len(aa_codes)))
        ax.set_yticks(range(len(aa_codes)))
        ax.set_xticklabels(aa_codes, fontsize=6)
        ax.set_yticklabels(aa_codes, fontsize=6)

        # Rotate x labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

        # Colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Similarity', fontsize=7)
        cbar.ax.tick_params(labelsize=6)

        ax.set_xlabel('Amino Acid', fontsize=8)
        ax.set_ylabel('Amino Acid', fontsize=8)

    def _plot_fragment_distribution(self, ax):
        """
        Panel G: Fragment type distribution (radial plot).
        """
        # Count fragment types
        frag_counts = self.fragmentation.groupby('ion_type').size()

        # Radial bar plot
        theta = np.linspace(0, 2*np.pi, len(frag_counts), endpoint=False)
        width = 2*np.pi / len(frag_counts) * 0.8

        colors_map = {'b': 'steelblue', 'y': 'crimson', 'a': 'green'}
        colors = [colors_map.get(ion, 'gray') for ion in frag_counts.index]

        bars = ax.bar(theta, frag_counts.values, width=width, bottom=0,
                     color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

        # Add labels
        for angle, count, ion_type in zip(theta, frag_counts.values, frag_counts.index):
            ax.text(angle, count + count*0.1, ion_type,
                   ha='center', va='bottom', fontsize=10, fontweight='bold')

        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        ax.set_ylim(0, max(frag_counts.values) * 1.3)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['polar'].set_visible(False)
        ax.grid(True, alpha=0.3)

    def _plot_sentropy_correlation_network(self, ax):
        """
        Panel H: Correlation network between S-entropy dimensions.
        """
        # Compute correlations between S-entropy dimensions across all data
        sentropy_cols = ['s_knowledge', 's_time', 's_entropy']

        # Combine amino acid and sequence data
        aa_sentropy = self.aa_alphabet[sentropy_cols]
        seq_sentropy = self.sequences[sentropy_cols]

        all_sentropy = pd.concat([aa_sentropy, seq_sentropy], ignore_index=True)

        # Correlation matrix
        corr = all_sentropy.corr()

        # Plot as network
        n = len(sentropy_cols)
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)

        # Node positions (circular layout)
        node_x = np.cos(angles)
        node_y = np.sin(angles)

        # Draw edges (correlations)
        for i in range(n):
            for j in range(i+1, n):
                corr_val = abs(corr.iloc[i, j])

                if corr_val > 0.1:  # Only show significant correlations
                    ax.plot([node_x[i], node_x[j]], [node_y[i], node_y[j]],
                           'gray', alpha=corr_val, linewidth=corr_val*5)

        # Draw nodes
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        for i, (x, y, label, color) in enumerate(zip(node_x, node_y, sentropy_cols, colors)):
            circle = Circle((x, y), 0.15, facecolor=color, edgecolor='black',
                          linewidth=2, alpha=0.8, zorder=10)
            ax.add_patch(circle)

            # Add label
            label_short = label.replace('s_', 'S-').title()
            ax.text(x, y, label_short, ha='center', va='center',
                   fontsize=7, fontweight='bold', color='white', zorder=11)

        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect('equal')
        ax.axis('off')

    def _add_panel_label(self, ax, label):
        """Add panel label (A, B, C, etc.) to subplot. Handles 2D, 3D, and polar axes."""
        # Check if this is a 3D axis
        if hasattr(ax, 'get_zlim'):
            # For 3D axes, use figure text instead
            bbox = ax.get_position()
            fig = ax.get_figure()
            fig.text(bbox.x0, bbox.y1 + 0.02, label,
                    fontsize=16, fontweight='bold', va='bottom', ha='left')
        elif hasattr(ax, 'set_theta_zero_location'):
            # For polar axes, use figure text
            bbox = ax.get_position()
            fig = ax.get_figure()
            fig.text(bbox.x0, bbox.y1 + 0.02, label,
                    fontsize=16, fontweight='bold', va='bottom', ha='left')
        else:
            ax.text(-0.1, 1.05, label, transform=ax.transAxes,
                   fontsize=16, fontweight='bold', va='top', ha='right')


# Additional specialized visualizations

class FragmentationGrammarVisualizer:
    """
    Specialized visualizations for fragmentation grammar.
    """

    def __init__(self, fragmentation_df, output_dir=None):
        self.fragmentation = fragmentation_df
        if output_dir is None:
            output_dir = PRECURSOR_ROOT / 'results' / 'visualizations' / 'molecular_language'
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def create_grammar_tree(self):
        """
        Create tree diagram of fragmentation grammar.
        """
        print("\nCreating fragmentation grammar tree...")

        fig, ax = plt.subplots(figsize=(14, 10))

        sequences = self.fragmentation['parent_sequence'].unique()

        for seq_idx, seq in enumerate(sequences):
            seq_frags = self.fragmentation[self.fragmentation['parent_sequence'] == seq]

            # Root node (parent sequence)
            root_y = seq_idx * 3
            ax.text(0, root_y, seq, ha='center', va='center',
                   fontsize=12, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue',
                            edgecolor='black', linewidth=2))

            # Fragment nodes
            for frag_idx, (_, frag) in enumerate(seq_frags.iterrows()):
                # Position based on fragment type and position
                if frag['ion_type'] == 'b':
                    x = frag['position'] * 2
                    y = root_y + 1
                    color = 'steelblue'
                elif frag['ion_type'] == 'y':
                    x = frag['position'] * 2
                    y = root_y - 1
                    color = 'crimson'
                else:
                    x = frag['position'] * 2
                    y = root_y
                    color = 'green'

                # Draw edge from root to fragment
                ax.plot([0, x], [root_y, y], 'gray', alpha=0.3, linewidth=1)

                # Fragment node
                circle = Circle((x, y), 0.3, facecolor=color, edgecolor='black',
                              linewidth=1.5, alpha=0.7)
                ax.add_patch(circle)

                # Label
                label = f"{frag['ion_type']}{frag['position']}"
                ax.text(x, y, label, ha='center', va='center',
                       fontsize=7, fontweight='bold', color='white')

                # m/z annotation
                ax.text(x, y - 0.5, f"{frag['mz']:.1f}",
                       ha='center', va='top', fontsize=6, alpha=0.7)

        ax.set_xlim(-2, 15)
        ax.set_ylim(-2, len(sequences) * 3 + 1)
        ax.set_aspect('equal')
        ax.axis('off')

        output_path = self.output_dir / 'fragmentation_grammar_tree.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved: {output_path}")
        plt.close()


class SequenceEntropyVisualizer:
    """
    Specialized visualizations for sequence S-entropy paths.
    """

    def __init__(self, sequences_df, output_dir=None):
        self.sequences = sequences_df
        if output_dir is None:
            output_dir = PRECURSOR_ROOT / 'results' / 'visualizations' / 'molecular_language'
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def create_entropy_flow_diagram(self):
        """
        Create Sankey-like flow diagram of S-entropy through sequences.
        """
        print("\nCreating S-entropy flow diagram...")

        fig, ax = plt.subplots(figsize=(16, 8))

        sequences = self.sequences['sequence'].unique()

        for seq_idx, seq in enumerate(sequences):
            seq_data = self.sequences[self.sequences['sequence'] == seq].sort_values('position')

            y_base = seq_idx * 2

            # Draw flow for each position
            for i in range(len(seq_data) - 1):
                curr = seq_data.iloc[i]
                next_pos = seq_data.iloc[i + 1]

                x1, x2 = curr['position'], next_pos['position']

                # Width proportional to entropy
                width1 = curr['s_entropy'] * 0.5 + 0.1
                width2 = next_pos['s_entropy'] * 0.5 + 0.1

                # Color by knowledge
                color = plt.cm.viridis(curr['s_knowledge'])

                # Draw trapezoid (flow segment)
                vertices = [
                    (x1, y_base - width1/2),
                    (x1, y_base + width1/2),
                    (x2, y_base + width2/2),
                    (x2, y_base - width2/2)
                ]

                polygon = plt.Polygon(vertices, facecolor=color, edgecolor='black',
                                    linewidth=0.5, alpha=0.7)
                ax.add_patch(polygon)

                # Add amino acid label
                ax.text((x1 + x2) / 2, y_base, curr['amino_acid'],
                       ha='center', va='center', fontsize=8, fontweight='bold',
                       color='white')

            # Sequence label
            ax.text(-0.5, y_base, seq, ha='right', va='center',
                   fontsize=10, fontweight='bold')

        ax.set_xlim(-1, max(self.sequences['position']) + 1)
        ax.set_ylim(-1, len(sequences) * 2)
        ax.set_xlabel('Position in Sequence', fontsize=12)
        ax.axis('off')

        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(0, 1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, fraction=0.02, pad=0.02)
        cbar.set_label('S-Knowledge', fontsize=10)

        output_path = self.output_dir / 'sentropy_flow_diagram.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved: {output_path}")
        plt.close()

    def create_3d_trajectory_plot(self):
        """
        Create 3D trajectory plot in S-entropy space.
        """
        print("\nCreating 3D S-entropy trajectory...")

        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        sequences = self.sequences['sequence'].unique()
        colors = sns.color_palette('husl', len(sequences))

        for seq_idx, seq in enumerate(sequences):
            seq_data = self.sequences[self.sequences['sequence'] == seq].sort_values('position')

            x = seq_data['s_knowledge'].values
            y = seq_data['s_time'].values
            z = seq_data['s_entropy'].values

            # Plot trajectory
            ax.plot(x, y, z, color=colors[seq_idx], linewidth=2, alpha=0.7, label=seq)

            # Plot points
            ax.scatter(x, y, z, color=colors[seq_idx], s=100, alpha=0.8,
                      edgecolors='black', linewidths=1)

            # Add amino acid labels
            for i, (xi, yi, zi, aa) in enumerate(zip(x, y, z, seq_data['amino_acid'].values)):
                ax.text(xi, yi, zi, aa, fontsize=7, fontweight='bold')

        ax.set_xlabel('S-Knowledge', fontsize=10)
        ax.set_ylabel('S-Time', fontsize=10)
        ax.set_zlabel('S-Entropy', fontsize=10)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

        output_path = self.output_dir / 'sentropy_3d_trajectory.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved: {output_path}")
        plt.close()


def main():
    """
    Main function - creates all visualizations.
    """
    print("="*80)
    print("MOLECULAR LANGUAGE VISUALIZATION SUITE")
    print("="*80)

    # Create master atlas
    atlas = MolecularLanguageAtlas()
    atlas.create_master_atlas()

    # Create specialized visualizations
    grammar_viz = FragmentationGrammarVisualizer(
        pd.read_csv(RESULTS_DIR / 'fragmentation_grammar.csv')
    )
    grammar_viz.create_grammar_tree()

    sequence_viz = SequenceEntropyVisualizer(
        pd.read_csv(RESULTS_DIR / 'sequence_sentropy_paths.csv')
    )
    sequence_viz.create_entropy_flow_diagram()
    sequence_viz.create_3d_trajectory_plot()

    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE")
    print("="*80)
    print("Generated files:")
    print("  - molecular_language_atlas.png (master figure)")
    print("  - fragmentation_grammar_tree.png")
    print("  - sentropy_flow_diagram.png")
    print("  - sentropy_3d_trajectory.png")
    print("="*80)


if __name__ == '__main__':
    main()
