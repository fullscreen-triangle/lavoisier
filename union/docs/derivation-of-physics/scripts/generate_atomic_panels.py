"""
Generate visualization panels for Section 10:
- Atomic Structure from Partition Coordinates
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec

plt.style.use('default')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = '#f8f8f8'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

# Element data
ELEMENTS = [
    'H', 'He',
    'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
    'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
    'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr'
]

def generate_atomic_structure_panel():
    """Visualize atomic structure from partition coordinates."""
    fig = plt.figure(figsize=(16, 14))
    gs = gridspec.GridSpec(4, 4, figure=fig, hspace=0.4, wspace=0.35)
    
    # Panel 1: Periodic table (first 36 elements)
    ax1 = fig.add_subplot(gs[0:2, 0:2])
    
    # Periodic table layout
    pt_layout = [
        [(1, 'H'), None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, (2, 'He')],
        [(3, 'Li'), (4, 'Be'), None, None, None, None, None, None, None, None, None, None, (5, 'B'), (6, 'C'), (7, 'N'), (8, 'O'), (9, 'F'), (10, 'Ne')],
        [(11, 'Na'), (12, 'Mg'), None, None, None, None, None, None, None, None, None, None, (13, 'Al'), (14, 'Si'), (15, 'P'), (16, 'S'), (17, 'Cl'), (18, 'Ar')],
        [(19, 'K'), (20, 'Ca'), (21, 'Sc'), (22, 'Ti'), (23, 'V'), (24, 'Cr'), (25, 'Mn'), (26, 'Fe'), (27, 'Co'), (28, 'Ni'), (29, 'Cu'), (30, 'Zn'), (31, 'Ga'), (32, 'Ge'), (33, 'As'), (34, 'Se'), (35, 'Br'), (36, 'Kr')]
    ]
    
    # Colors by block
    colors = {
        's': '#FF6B6B',  # Alkali, alkaline earth
        'p': '#4ECDC4',  # Main group
        'd': '#FFE66D',  # Transition metals
    }
    
    for row_idx, row in enumerate(pt_layout):
        for col_idx, elem in enumerate(row):
            if elem is not None:
                Z, symbol = elem
                # Determine block
                if col_idx < 2:
                    block = 's'
                elif col_idx >= 12:
                    block = 'p'
                else:
                    block = 'd'
                
                rect = FancyBboxPatch((col_idx, 3-row_idx), 0.9, 0.9, 
                                      boxstyle="round,pad=0.02",
                                      facecolor=colors[block], edgecolor='black', linewidth=1)
                ax1.add_patch(rect)
                ax1.text(col_idx + 0.45, 3-row_idx + 0.65, symbol, 
                        ha='center', va='center', fontsize=9, fontweight='bold')
                ax1.text(col_idx + 0.45, 3-row_idx + 0.25, str(Z),
                        ha='center', va='center', fontsize=7, color='gray')
    
    ax1.set_xlim(-0.5, 18.5)
    ax1.set_ylim(-0.5, 4.5)
    ax1.set_aspect('equal')
    ax1.axis('off')
    ax1.set_title('Periodic Table from Partition Geometry\n(Z = partition count)', fontsize=12)
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors['s'], edgecolor='black', label='s-block'),
                       Patch(facecolor=colors['d'], edgecolor='black', label='d-block'),
                       Patch(facecolor=colors['p'], edgecolor='black', label='p-block')]
    ax1.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=9)
    
    # Panel 2: Shell filling order
    ax2 = fig.add_subplot(gs[0, 2])
    
    # Diagonal filling rule
    orbitals = ['1s', '2s', '2p', '3s', '3p', '4s', '3d', '4p', '5s', '4d', '5p', '6s']
    n_plus_l = [1, 2, 3, 3, 4, 4, 5, 5, 5, 6, 6, 6]
    electrons = [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2]
    cumulative = np.cumsum(electrons)
    
    colors = plt.cm.viridis(np.array(n_plus_l) / 7)
    
    y_pos = np.arange(len(orbitals))
    bars = ax2.barh(y_pos, electrons, color=colors, edgecolor='black')
    
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(orbitals)
    ax2.set_xlabel('Electrons')
    ax2.set_title('Shell Filling Order\n(n + l rule)')
    ax2.invert_yaxis()
    
    # Add cumulative
    for i, (e, c) in enumerate(zip(electrons, cumulative)):
        ax2.text(e + 0.5, i, f'Σ={c}', va='center', fontsize=8)
    
    # Panel 3: Period lengths
    ax3 = fig.add_subplot(gs[0, 3])
    
    periods = [1, 2, 3, 4, 5, 6, 7]
    lengths = [2, 8, 8, 18, 18, 32, 32]
    
    colors = plt.cm.plasma(np.linspace(0.2, 0.9, len(periods)))
    bars = ax3.bar(periods, lengths, color=colors, edgecolor='black', linewidth=2)
    
    ax3.set_xlabel('Period')
    ax3.set_ylabel('Number of Elements')
    ax3.set_title('Period Lengths\n(2, 8, 8, 18, 18, 32, 32)')
    ax3.set_xticks(periods)
    
    # Add formula annotation
    ax3.annotate('2(1² + 1² + 2² + 2² + ...)', xy=(4, 20), fontsize=9,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Panel 4: Group properties (alkali metals)
    ax4 = fig.add_subplot(gs[1, 2])
    
    alkali = ['Li', 'Na', 'K', 'Rb', 'Cs']
    Z_alkali = [3, 11, 19, 37, 55]
    ionization = [5.39, 5.14, 4.34, 4.18, 3.89]  # eV
    
    ax4.plot(Z_alkali, ionization, 'bo-', markersize=10, linewidth=2)
    for z, ion, name in zip(Z_alkali, ionization, alkali):
        ax4.annotate(name, xy=(z, ion), xytext=(z+2, ion+0.1), fontsize=10)
    
    ax4.set_xlabel('Atomic Number Z')
    ax4.set_ylabel('Ionization Energy (eV)')
    ax4.set_title('Group 1 (Alkali Metals)\n(Same outer l=0)')
    ax4.set_xlim(0, 60)
    
    # Panel 5: Transition metals
    ax5 = fig.add_subplot(gs[1, 3])
    
    # 3d transition metals
    transition = ['Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn']
    Z_trans = list(range(21, 31))
    d_electrons = [1, 2, 3, 5, 5, 6, 7, 8, 10, 10]  # Note Cr, Cu anomalies
    
    bars = ax5.bar(Z_trans, d_electrons, color='gold', edgecolor='black')
    ax5.set_xticks(Z_trans)
    ax5.set_xticklabels(transition, rotation=45, fontsize=8)
    ax5.set_xlabel('Element')
    ax5.set_ylabel('3d Electrons')
    ax5.set_title('Transition Metals\n(3d filling)')
    
    # Mark anomalies
    ax5.annotate('Anomaly', xy=(24, 5), xytext=(24, 7),
                arrowprops=dict(arrowstyle='->', color='red'), fontsize=8, color='red')
    ax5.annotate('Anomaly', xy=(29, 10), xytext=(29, 11),
                arrowprops=dict(arrowstyle='->', color='red'), fontsize=8, color='red')
    
    # Panel 6: Hydrogen spectrum (Lyman, Balmer, Paschen)
    ax6 = fig.add_subplot(gs[2, 0])
    
    # Energy levels
    n_levels = np.arange(1, 7)
    E_n = -13.6 / n_levels**2
    
    for n, E in zip(n_levels, E_n):
        ax6.hlines(E, 0, 1, linewidth=3, color=plt.cm.viridis(n/7))
        ax6.text(1.05, E, f'n={n}', fontsize=10, va='center')
    
    # Transitions
    series = [
        ('Lyman', 1, 'purple'),
        ('Balmer', 2, 'blue'),
        ('Paschen', 3, 'red')
    ]
    
    for name, n_final, color in series:
        for n_i in range(n_final + 1, 6):
            E_i = -13.6 / n_i**2
            E_f = -13.6 / n_final**2
            x_offset = 0.2 + 0.2 * (series.index((name, n_final, color)))
            ax6.annotate('', xy=(x_offset, E_f), xytext=(x_offset, E_i),
                        arrowprops=dict(arrowstyle='->', color=color, lw=1.5, alpha=0.6))
    
    ax6.set_xlim(-0.1, 1.8)
    ax6.set_ylim(-15, 1)
    ax6.set_ylabel('Energy (eV)')
    ax6.set_title('Hydrogen Spectrum\n(Partition Transitions)')
    ax6.set_xticks([])
    
    # Legend
    for i, (name, _, color) in enumerate(series):
        ax6.text(0.2 + 0.2*i, -14, name, fontsize=8, color=color, rotation=90)
    
    # Panel 7: Ionization energy trend
    ax7 = fig.add_subplot(gs[2, 1])
    
    # First 36 elements
    Z = np.arange(1, 37)
    # Approximate ionization energies (simplified)
    IE = np.array([13.6, 24.6, 5.4, 9.3, 8.3, 11.3, 14.5, 13.6, 17.4, 21.6,
                   5.1, 7.6, 6.0, 8.2, 10.5, 10.4, 13.0, 15.8,
                   4.3, 6.1, 6.5, 6.8, 6.7, 6.8, 7.4, 7.9, 7.9, 7.6, 7.7, 9.4,
                   6.0, 7.9, 9.8, 9.8, 11.8, 14.0])
    
    # Color by period
    period_colors = [1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 
                    3, 3, 3, 3, 3, 3, 3, 3,
                    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]
    
    scatter = ax7.scatter(Z, IE, c=period_colors, cmap='Set1', s=50, edgecolor='black')
    ax7.plot(Z, IE, 'k-', alpha=0.3, linewidth=1)
    
    # Mark noble gases
    noble = [2, 10, 18, 36]
    for ng in noble:
        ax7.annotate(ELEMENTS[ng-1], xy=(ng, IE[ng-1]), xytext=(ng, IE[ng-1]+2),
                    fontsize=8, ha='center')
    
    ax7.set_xlabel('Atomic Number Z')
    ax7.set_ylabel('First Ionization Energy (eV)')
    ax7.set_title('Ionization Energy Trend\n(Periodic Pattern)')
    
    # Panel 8: Electron configuration notation
    ax8 = fig.add_subplot(gs[2, 2])
    
    configs = [
        ('H', '1s¹'),
        ('He', '1s²'),
        ('C', '1s² 2s² 2p²'),
        ('O', '1s² 2s² 2p⁴'),
        ('Fe', '[Ar] 3d⁶ 4s²'),
        ('Cu', '[Ar] 3d¹⁰ 4s¹')
    ]
    
    y_pos = np.arange(len(configs))
    ax8.set_xlim(0, 10)
    ax8.set_ylim(-0.5, len(configs)-0.5)
    
    for i, (elem, config) in enumerate(configs):
        ax8.text(0.5, i, elem, fontsize=12, fontweight='bold', va='center')
        ax8.text(2, i, config, fontsize=10, va='center', family='monospace')
        ax8.text(8, i, f'(n,l,m,s)', fontsize=8, va='center', color='gray')
    
    ax8.axis('off')
    ax8.set_title('Electron Configurations\n(Partition Coordinates)')
    
    # Panel 9: Atomic radius trend
    ax9 = fig.add_subplot(gs[2, 3])
    
    # Approximate atomic radii (pm)
    radii = np.array([53, 31, 167, 112, 87, 77, 75, 73, 71, 69,
                      190, 145, 118, 111, 98, 88, 79, 71,
                      243, 194, 184, 176, 171, 166, 161, 156, 152, 149, 145, 142,
                      136, 125, 114, 103, 94, 88])
    
    ax9.scatter(Z, radii, c=period_colors, cmap='Set1', s=50, edgecolor='black')
    ax9.plot(Z, radii, 'k-', alpha=0.3, linewidth=1)
    
    ax9.set_xlabel('Atomic Number Z')
    ax9.set_ylabel('Atomic Radius (pm)')
    ax9.set_title('Atomic Radius Trend\n(r ∝ n²/Z_eff)')
    
    # Panel 10: Electronegativity
    ax10 = fig.add_subplot(gs[3, 0])
    
    # Pauling electronegativity
    EN = np.array([2.2, 0, 1.0, 1.6, 2.0, 2.5, 3.0, 3.4, 4.0, 0,
                   0.9, 1.3, 1.6, 1.9, 2.2, 2.6, 3.2, 0,
                   0.8, 1.0, 1.4, 1.5, 1.6, 1.7, 1.6, 1.8, 1.9, 1.9, 1.9, 1.7,
                   1.8, 2.0, 2.2, 2.6, 3.0, 0])
    
    mask = EN > 0
    ax10.scatter(Z[mask], EN[mask], c=np.array(period_colors)[mask], 
                cmap='Set1', s=50, edgecolor='black')
    ax10.plot(Z[mask], EN[mask], 'k-', alpha=0.3, linewidth=1)
    
    ax10.set_xlabel('Atomic Number Z')
    ax10.set_ylabel('Electronegativity (Pauling)')
    ax10.set_title('Electronegativity Trend\n(Partition Boundary Affinity)')
    
    # Panel 11: Spectral lines
    ax11 = fig.add_subplot(gs[3, 1])
    
    # Balmer series wavelengths
    n_upper = np.arange(3, 8)
    R = 1.097e7  # Rydberg constant
    wavelengths = 1 / (R * (1/4 - 1/n_upper**2)) * 1e9  # nm
    
    for n, wl in zip(n_upper, wavelengths):
        color = plt.cm.rainbow((wl - 400) / 300)
        ax11.axvline(x=wl, color=color, linewidth=4, alpha=0.8)
        ax11.text(wl, 0.8, f'n={n}', fontsize=8, rotation=90, va='bottom', ha='center')
    
    ax11.set_xlim(380, 700)
    ax11.set_ylim(0, 1)
    ax11.set_xlabel('Wavelength (nm)')
    ax11.set_title('Balmer Series\n(Δl = ±1 Selection)')
    ax11.set_yticks([])
    
    # Add visible spectrum background
    for wl in range(380, 700, 5):
        color = plt.cm.rainbow((wl - 380) / 320)
        ax11.axvspan(wl, wl+5, alpha=0.1, color=color)
    
    # Panel 12: Complete derivation chain
    ax12 = fig.add_subplot(gs[3, 2:])
    
    chain = [
        'Bounded\nPhase Space',
        'Poincaré\nRecurrence',
        'Oscillatory\nDynamics',
        'Categorical\nStates',
        '(n,l,m,s)\nCoordinates',
        '2n² Capacity\n(n+l) Order',
        'Periodic\nTable'
    ]
    
    n_steps = len(chain)
    x_pos = np.linspace(0, 10, n_steps)
    
    for i, (x, text) in enumerate(zip(x_pos, chain)):
        # Box
        color = plt.cm.viridis(i / n_steps)
        rect = FancyBboxPatch((x - 0.6, 0.2), 1.2, 0.6,
                              boxstyle="round,pad=0.05",
                              facecolor=color, edgecolor='black', linewidth=2)
        ax12.add_patch(rect)
        ax12.text(x, 0.5, text, ha='center', va='center', fontsize=9,
                 fontweight='bold', color='white' if i > 2 else 'black')
        
        # Arrow
        if i < n_steps - 1:
            ax12.annotate('', xy=(x_pos[i+1] - 0.7, 0.5), xytext=(x + 0.7, 0.5),
                        arrowprops=dict(arrowstyle='->', color='black', lw=2))
    
    ax12.set_xlim(-1, 11)
    ax12.set_ylim(0, 1)
    ax12.axis('off')
    ax12.set_title('Complete Derivation Chain\n(First Principles → Chemistry)', fontsize=12)
    
    plt.suptitle('Atomic Structure from Partition Coordinates', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig('../figures/atomic_structure_panel.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig('../figures/atomic_structure_panel.pdf', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Generated: atomic_structure_panel.png/pdf")

if __name__ == "__main__":
    import os
    os.makedirs('../figures', exist_ok=True)
    generate_atomic_structure_panel()

