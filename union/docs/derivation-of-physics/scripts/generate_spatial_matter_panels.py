"""
Generate visualization panels for Sections 6-7:
- Spatial Structure from Partition Geometry
- Matter, Energy, and Mode Occupation
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Wedge, FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap, Normalize
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
from scipy.special import sph_harm
import math

plt.style.use('default')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = '#f8f8f8'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

def generate_spatial_matter_panel():
    """Visualize spatial emergence and matter/energy."""
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.4, wspace=0.35)
    
    # Panel 1: 3D emergence from (l, m) coordinates
    ax1 = fig.add_subplot(gs[0, 0], projection='3d')
    
    # Draw coordinate axes from l, m mapping
    # Show how l, m define spherical coordinates
    u = np.linspace(0, 2*np.pi, 50)
    v = np.linspace(0, np.pi, 25)
    
    # Spherical shell
    r = 1.0
    x = r * np.outer(np.cos(u), np.sin(v))
    y = r * np.outer(np.sin(u), np.sin(v))
    z = r * np.outer(np.ones(np.size(u)), np.cos(v))
    
    ax1.plot_surface(x, y, z, alpha=0.3, color='cyan', edgecolor='none')
    
    # Draw coordinate lines
    for theta in np.linspace(0, np.pi, 5):
        phi = np.linspace(0, 2*np.pi, 50)
        ax1.plot(r*np.sin(theta)*np.cos(phi), r*np.sin(theta)*np.sin(phi), 
                r*np.cos(theta)*np.ones_like(phi), 'b-', alpha=0.5, linewidth=1)
    
    for phi in np.linspace(0, 2*np.pi, 8):
        theta = np.linspace(0, np.pi, 25)
        ax1.plot(r*np.sin(theta)*np.cos(phi), r*np.sin(theta)*np.sin(phi),
                r*np.cos(theta), 'r-', alpha=0.5, linewidth=1)
    
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')
    ax1.set_title('3D from Angular (l, m)\nSpherical Coordinates')
    
    # Panel 2: Radial scaling r ∝ n²
    ax2 = fig.add_subplot(gs[0, 1])
    
    n_vals = np.arange(1, 8)
    r_vals = n_vals**2  # r ∝ n²
    
    # Draw concentric circles
    theta = np.linspace(0, 2*np.pi, 100)
    for n, r in zip(n_vals, r_vals):
        x = r/50 * np.cos(theta)
        y = r/50 * np.sin(theta)
        ax2.plot(x, y, linewidth=2, label=f'n={n}, r∝{r}')
        ax2.text(r/50 * 1.1, 0, f'n={n}', fontsize=8, va='center')
    
    ax2.set_xlim(-1.2, 1.2)
    ax2.set_ylim(-1.2, 1.2)
    ax2.set_aspect('equal')
    ax2.set_xlabel('x (scaled)')
    ax2.set_ylabel('y (scaled)')
    ax2.set_title('Radial Extension\nr ∝ n² (Bohr scaling)')
    
    # Panel 3: Dimensionality uniqueness
    ax3 = fig.add_subplot(gs[0, 2])
    
    # Show D=1,2,3,4 and their angular momentum structure
    dims = [1, 2, 3, 4]
    angular_qnums = [0, 1, 2, 3]  # Number of independent angular quantum numbers
    colors = ['gray', 'blue', 'green', 'orange']
    
    bars = ax3.bar(dims, angular_qnums, color=colors, edgecolor='black', linewidth=2)
    
    # Highlight D=3
    bars[2].set_edgecolor('red')
    bars[2].set_linewidth(4)
    
    ax3.set_xlabel('Spatial Dimension D')
    ax3.set_ylabel('Angular Quantum Numbers')
    ax3.set_title('Dimensionality from\nPartition Constraints')
    ax3.set_xticks(dims)
    
    # Annotation
    ax3.annotate('(l, m) structure\n→ D = 3 unique', xy=(3, 2), xytext=(3.5, 2.5),
                arrowprops=dict(arrowstyle='->', color='red'), fontsize=10, color='red')
    
    # Panel 4: Locality from radial separation
    ax4 = fig.add_subplot(gs[0, 3])
    
    # Overlap integral decay
    delta_n = np.linspace(0, 5, 100)
    overlap = np.exp(-delta_n / 0.5)
    
    ax4.semilogy(delta_n, overlap, 'b-', linewidth=2.5)
    ax4.fill_between(delta_n, overlap, alpha=0.2)
    ax4.axhline(y=0.01, color='red', linestyle='--', label='1% threshold')
    
    ax4.set_xlabel('Radial Separation |n₁ - n₂|')
    ax4.set_ylabel('Overlap ⟨n₁|n₂⟩')
    ax4.set_title('Locality Principle\n(Exponential decay)')
    ax4.legend()
    ax4.set_xlim(0, 5)
    
    # Panel 5: Occupied vs unoccupied modes
    ax5 = fig.add_subplot(gs[1, 0])
    
    # Mode occupation grid
    n_modes = 100
    np.random.seed(42)
    occupied = np.random.random(n_modes) < 0.05  # 5% occupied
    
    grid_size = 10
    occupation_grid = occupied.reshape(grid_size, grid_size)
    
    cmap = LinearSegmentedColormap.from_list('occupation', ['#f0f0f0', '#2196F3'])
    ax5.imshow(occupation_grid, cmap=cmap, aspect='equal')
    
    # Count
    n_occ = occupied.sum()
    ax5.set_title(f'Mode Occupation\n{n_occ}/100 occupied (5%)')
    ax5.set_xlabel('Mode index i')
    ax5.set_ylabel('Mode index j')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#f0f0f0', edgecolor='black', label='Unoccupied (dark)'),
                       Patch(facecolor='#2196F3', edgecolor='black', label='Occupied (matter)')]
    ax5.legend(handles=legend_elements, loc='upper right', fontsize=8)
    
    # Panel 6: Exclusion principle
    ax6 = fig.add_subplot(gs[1, 1])
    
    # Draw orbital boxes
    orbitals = ['1s', '2s', '2p', '3s']
    capacities = [2, 2, 6, 2]
    occupations = [2, 2, 4, 0]  # Carbon-like
    
    y_pos = 0
    for orb, cap, occ in zip(orbitals, capacities, occupations):
        # Draw boxes
        for i in range(cap//2):
            box = Rectangle((i*0.5, y_pos), 0.4, 0.8, 
                           facecolor='white', edgecolor='black', linewidth=2)
            ax6.add_patch(box)
            
            # Add electrons (arrows for spin)
            if 2*i < occ:
                ax6.annotate('', xy=(i*0.5+0.2, y_pos+0.7), xytext=(i*0.5+0.2, y_pos+0.3),
                            arrowprops=dict(arrowstyle='->', color='blue', lw=2))
            if 2*i+1 < occ:
                ax6.annotate('', xy=(i*0.5+0.2, y_pos+0.3), xytext=(i*0.5+0.2, y_pos+0.7),
                            arrowprops=dict(arrowstyle='->', color='red', lw=2))
        
        ax6.text(-0.3, y_pos+0.4, orb, ha='right', va='center', fontsize=11, fontweight='bold')
        y_pos += 1.2
    
    ax6.set_xlim(-0.5, 2)
    ax6.set_ylim(-0.2, 5)
    ax6.axis('off')
    ax6.set_title('Exclusion Principle\n(Coordinate Uniqueness)')
    
    # Add annotation
    ax6.text(1.5, 2, '↑ spin +½\n↓ spin -½\nMax 2 per orbital', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Panel 7: Mass-frequency relationship
    ax7 = fig.add_subplot(gs[1, 2])
    
    omega = np.linspace(0.1, 10, 100)
    hbar = 1.0
    c = 1.0
    mass = hbar * omega / c**2
    
    ax7.plot(omega, mass, 'b-', linewidth=2.5)
    ax7.fill_between(omega, 0, mass, alpha=0.2)
    
    # Mark particle masses
    particles = {'electron': 0.511, 'muon': 105.7, 'proton': 938.3}
    for name, m in particles.items():
        if m/100 < 10:
            ax7.axhline(y=m/100, color='red', linestyle='--', alpha=0.5)
            ax7.text(9, m/100, name, fontsize=9, va='bottom')
    
    ax7.set_xlabel('Oscillation Frequency ω')
    ax7.set_ylabel('Mass m = ℏω/c²')
    ax7.set_title('Mass-Frequency Identity\nm = ℏω/c²')
    ax7.set_xlim(0, 10)
    
    # Panel 8: 95%/5% dark sector ratio
    ax8 = fig.add_subplot(gs[1, 3])
    
    sizes = [5, 27, 68]  # Matter, Dark Matter, Dark Energy
    labels = ['Visible\nMatter\n5%', 'Dark\nMatter\n27%', 'Dark\nEnergy\n68%']
    colors = ['#2196F3', '#9C27B0', '#424242']
    explode = (0.1, 0.02, 0.02)
    
    wedges, texts, autotexts = ax8.pie(sizes, labels=labels, colors=colors, 
                                        explode=explode, autopct='',
                                        startangle=90, textprops={'fontsize': 10})
    
    ax8.set_title('Cosmic Mode Occupation\n(Occupied vs Unoccupied)')
    
    # Panel 9: Wave-particle duality
    ax9 = fig.add_subplot(gs[2, 0])
    
    x = np.linspace(0, 10, 500)
    
    # Wave aspect
    wave = np.sin(2*np.pi*x) * np.exp(-((x-5)**2)/4)
    ax9.fill_between(x, wave, alpha=0.3, color='blue', label='Wave (mode)')
    ax9.plot(x, wave, 'b-', linewidth=2)
    
    # Particle aspect (localized)
    particle_x = 5
    ax9.axvline(x=particle_x, color='red', linestyle='--', linewidth=2, label='Particle (occupation)')
    ax9.plot(particle_x, 0, 'ro', markersize=15)
    
    ax9.set_xlabel('Position x')
    ax9.set_ylabel('Amplitude')
    ax9.set_title('Wave-Particle Duality\n(Mode vs Occupation)')
    ax9.legend(loc='upper right')
    ax9.set_xlim(0, 10)
    
    # Panel 10: Energy conservation
    ax10 = fig.add_subplot(gs[2, 1])
    
    t = np.linspace(0, 10, 200)
    
    # Three energy components that sum to constant
    E_kinetic = 2 + np.sin(2*np.pi*t/3)**2
    E_potential = 2 + np.cos(2*np.pi*t/3)**2
    E_total = E_kinetic + E_potential
    
    ax10.plot(t, E_kinetic, 'b-', linewidth=2, label='Kinetic E_k')
    ax10.plot(t, E_potential, 'r-', linewidth=2, label='Potential E_p')
    ax10.plot(t, E_total, 'k-', linewidth=3, label='Total E')
    
    ax10.fill_between(t, 0, E_kinetic, alpha=0.2, color='blue')
    ax10.fill_between(t, E_kinetic, E_kinetic+E_potential, alpha=0.2, color='red')
    
    ax10.set_xlabel('Time t')
    ax10.set_ylabel('Energy')
    ax10.set_title('Energy Conservation\ndE/dt = 0')
    ax10.legend(loc='upper right')
    ax10.set_xlim(0, 10)
    ax10.set_ylim(0, 6)
    
    # Panel 11: Mode occupation statistics
    ax11 = fig.add_subplot(gs[2, 2])
    
    # Fermi-Dirac vs Bose-Einstein
    E = np.linspace(0, 5, 100)
    mu = 2.0  # Chemical potential
    kT = 0.5
    
    # Fermi-Dirac (fermions)
    f_FD = 1 / (np.exp((E - mu)/kT) + 1)
    
    # Bose-Einstein (bosons)
    f_BE = np.zeros_like(E)
    mask = E > mu + 0.1
    f_BE[mask] = 1 / (np.exp((E[mask] - mu)/kT) - 1)
    f_BE = np.clip(f_BE, 0, 5)
    
    ax11.plot(E, f_FD, 'b-', linewidth=2.5, label='Fermi-Dirac (s=±½)')
    ax11.plot(E, f_BE, 'r-', linewidth=2.5, label='Bose-Einstein (s=0,1,...)')
    ax11.axvline(x=mu, color='gray', linestyle='--', label=f'μ = {mu}')
    
    ax11.set_xlabel('Energy E')
    ax11.set_ylabel('Occupation f(E)')
    ax11.set_title('Mode Occupation Statistics\n(Fermions vs Bosons)')
    ax11.legend(loc='upper right', fontsize=9)
    ax11.set_xlim(0, 5)
    ax11.set_ylim(0, 3)
    
    # Panel 12: Vacuum energy and dark sector
    ax12 = fig.add_subplot(gs[2, 3])
    
    # Modes by energy
    omega = np.linspace(0, 10, 100)
    density_of_states = omega**2  # 3D density of states
    
    # Zero-point energy
    E_zp = 0.5 * omega  # ½ℏω per mode
    vacuum_energy = density_of_states * E_zp
    
    ax12.fill_between(omega, 0, vacuum_energy, alpha=0.3, color='purple', label='Vacuum energy')
    ax12.plot(omega, vacuum_energy, 'purple', linewidth=2)
    
    # Occupied modes (small portion)
    occupied_modes = density_of_states * 0.05 * E_zp
    ax12.fill_between(omega, 0, occupied_modes, alpha=0.5, color='blue', label='Occupied (visible)')
    
    ax12.set_xlabel('Frequency ω')
    ax12.set_ylabel('Energy Density')
    ax12.set_title('Vacuum Energy\n(Unoccupied Mode Contribution)')
    ax12.legend(loc='upper left')
    ax12.set_xlim(0, 10)
    
    plt.suptitle('Spatial Structure, Matter, and Energy', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('../figures/spatial_matter_panel.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig('../figures/spatial_matter_panel.pdf', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Generated: spatial_matter_panel.png/pdf")

if __name__ == "__main__":
    import os
    os.makedirs('../figures', exist_ok=True)
    generate_spatial_matter_panel()

