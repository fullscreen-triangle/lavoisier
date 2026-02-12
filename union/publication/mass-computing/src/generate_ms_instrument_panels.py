#!/usr/bin/env python3
"""
Generate mass spectrometer instrument panels:
1. FT-ICR (Fourier Transform Ion Cyclotron Resonance)
2. Orbitrap

Each panel contains 4 charts showing the physics of ion motion and detection.
Style matches the existing TOF panel.
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap
import os

# Physical constants
E_CHARGE = 1.602176634e-19  # Elementary charge (C)
AMU = 1.66053906660e-27  # Atomic mass unit (kg)

def generate_fticr_panel(save_path):
    """
    Panel: FT-ICR (Fourier Transform Ion Cyclotron Resonance) Mass Spectrometer

    Key physics:
    - Cyclotron frequency: ω_c = qB/m
    - Ions orbit in magnetic field
    - Frequency measurement gives m/z
    - Ultra-high resolution
    """
    fig = plt.figure(figsize=(14, 12))
    fig.patch.set_facecolor('white')

    # Title
    fig.suptitle('Panel: FT-ICR (Fourier Transform Ion Cyclotron Resonance) Mass Spectrometer',
                 fontsize=14, fontweight='bold', y=0.98)

    # === Chart A: 3D Ion Cyclotron Orbits ===
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    ax1.set_title('A) 3D Ion Cyclotron Orbits\nLighter ions orbit faster', fontsize=11)

    # Cyclotron motion for different m/z
    t = np.linspace(0, 10 * np.pi, 1000)

    # Different m/z values
    mz_values = [100, 500, 1000, 2000]
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']

    # Magnetic field B = 7 Tesla (typical)
    B = 7.0

    for i, mz in enumerate(mz_values):
        # Cyclotron frequency ω_c = qB/m = eB/(m/z × amu)
        omega_c = E_CHARGE * B / (mz * AMU)
        # Normalize for visualization
        omega_norm = omega_c / (E_CHARGE * B / (100 * AMU))  # Relative to m/z=100

        # Orbital radius (arbitrary units for visualization)
        r = 0.03  # All same radius initially

        # Cyclotron orbit
        x = r * np.cos(omega_norm * t)
        y = r * np.sin(omega_norm * t)
        z = t / (10 * np.pi) * 0.001 * mz  # Slight axial drift proportional to m/z

        ax1.plot(x, y, z, color=colors[i], linewidth=1.5, label=f'm/z = {mz}', alpha=0.8)

    # Draw magnetic field direction
    ax1.quiver(0, 0, 0, 0, 0, 0.1, color='black', arrow_length_ratio=0.3, linewidth=2)
    ax1.text(0, 0, 0.12, 'B', fontsize=12, ha='center', fontweight='bold')

    ax1.set_xlabel('X Position (m)', fontsize=9)
    ax1.set_ylabel('Y Position (m)', fontsize=9)
    ax1.set_zlabel('Axial Position (m)', fontsize=9)
    ax1.legend(loc='upper left', fontsize=8)

    # === Chart B: Cyclotron Frequency vs m/z (3D surface) ===
    ax2 = fig.add_subplot(2, 2, 2, projection='3d')
    ax2.set_title('B) Cyclotron Frequency Relationship\nω_c = qB/m', fontsize=11)

    # Create surface: frequency as function of m/z and B field
    mz_range = np.linspace(100, 2000, 50)
    B_range = np.linspace(3, 15, 50)  # Tesla
    MZ, B_field = np.meshgrid(mz_range, B_range)

    # Cyclotron frequency (in MHz for visualization)
    # ω_c = eB/m, f_c = eB/(2πm)
    freq_MHz = (E_CHARGE * B_field) / (2 * np.pi * MZ * AMU) / 1e6

    surf = ax2.plot_surface(MZ, B_field, freq_MHz, cmap='viridis', alpha=0.85,
                            linewidth=0.1, antialiased=True)

    ax2.set_xlabel('m/z', fontsize=9)
    ax2.set_ylabel('B (Tesla)', fontsize=9)
    ax2.set_zlabel('Frequency (MHz)', fontsize=9)

    cbar = plt.colorbar(surf, ax=ax2, shrink=0.6, pad=0.1)
    cbar.set_label('f_c (MHz)', fontsize=9)

    # === Chart C: Image Current & FT (3D) ===
    ax3 = fig.add_subplot(2, 2, 3, projection='3d')
    ax3.set_title('C) Image Current Detection\nFourier Transform to spectrum', fontsize=11)

    # Time domain signal (image current from orbiting ions)
    t_signal = np.linspace(0, 1, 2000)  # 1 second acquisition

    # Mixed signal from 3 m/z values
    mz_components = [500, 750, 1000]
    frequencies = [(E_CHARGE * 7.0) / (2 * np.pi * mz * AMU) / 1e6 for mz in mz_components]
    amplitudes = [1.0, 0.6, 0.4]

    # Generate time-domain signal
    signal = np.zeros_like(t_signal)
    for freq, amp in zip(frequencies, amplitudes):
        # Add damped oscillation (ions lose coherence)
        decay = np.exp(-t_signal * 2)
        signal += amp * decay * np.cos(2 * np.pi * freq * 1000 * t_signal)

    # Add noise
    signal += 0.1 * np.random.randn(len(signal))

    # Create 3D representation: time vs frequency domain
    # FFT
    fft_result = np.abs(np.fft.fft(signal))[:len(signal)//2]
    freq_axis = np.fft.fftfreq(len(signal), t_signal[1]-t_signal[0])[:len(signal)//2]

    # Subsample for visualization
    t_sub = t_signal[::20]
    signal_sub = signal[::20]

    # Plot time domain
    ax3.plot(t_sub * 1000, np.zeros_like(t_sub), signal_sub, color='#377eb8', linewidth=1)

    # Plot frequency domain (as bars)
    freq_sub = freq_axis[::10]
    fft_sub = fft_result[::10] / max(fft_result) * 2
    for f, h in zip(freq_sub[:50], fft_sub[:50]):
        if h > 0.1:
            ax3.bar3d(0, f, 0, 50, 0.5, h, color='#e41a1c', alpha=0.7)

    ax3.set_xlabel('Time (ms)', fontsize=9)
    ax3.set_ylabel('Frequency (kHz)', fontsize=9)
    ax3.set_zlabel('Amplitude', fontsize=9)

    # === Chart D: Resolution vs m/z (3D surface) ===
    ax4 = fig.add_subplot(2, 2, 4, projection='3d')
    ax4.set_title('D) Mass Resolution Landscape\nR = m/Δm', fontsize=11)

    # Resolution depends on observation time and m/z
    mz_range = np.linspace(100, 2000, 50)
    t_obs = np.linspace(0.1, 2, 50)  # Observation time in seconds
    MZ, T_obs = np.meshgrid(mz_range, t_obs)

    # Resolution: R ∝ f × T_obs, and f ∝ 1/m
    # R = (B × T_obs) / m (simplified)
    B_tesla = 7.0
    Resolution = (E_CHARGE * B_tesla * T_obs) / (MZ * AMU) / 1e6
    # Normalize to typical FT-ICR resolution range
    Resolution = Resolution / Resolution.max() * 1e6  # Up to 1 million

    surf4 = ax4.plot_surface(MZ, T_obs, Resolution / 1e6, cmap='plasma', alpha=0.85,
                             linewidth=0.1, antialiased=True)

    ax4.set_xlabel('m/z', fontsize=9)
    ax4.set_ylabel('Observation Time (s)', fontsize=9)
    ax4.set_zlabel('Resolution (×10⁶)', fontsize=9)

    cbar4 = plt.colorbar(surf4, ax=ax4, shrink=0.6, pad=0.1)
    cbar4.set_label('R (millions)', fontsize=9)

    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    plt.savefig(save_path, dpi=300, facecolor='white', edgecolor='none', bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

def generate_orbitrap_panel(save_path):
    """
    Panel: Orbitrap Mass Spectrometer

    Key physics:
    - Axial oscillation frequency: ω = √(k/m) where k = electrostatic field constant
    - Ions orbit around central electrode
    - Frequency-based mass measurement
    - High resolution without superconducting magnet
    """
    fig = plt.figure(figsize=(14, 12))
    fig.patch.set_facecolor('white')

    # Title
    fig.suptitle('Panel: Orbitrap Mass Spectrometer',
                 fontsize=14, fontweight='bold', y=0.98)

    # === Chart A: 3D Ion Trajectories in Orbitrap ===
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    ax1.set_title('A) 3D Ion Trajectories in Orbitrap\nSpiral motion around central electrode', fontsize=11)

    # Orbitrap geometry: ions spiral around central spindle electrode
    t = np.linspace(0, 20 * np.pi, 2000)

    # Different m/z values
    mz_values = [200, 500, 1000, 1500]
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']

    # Field constant k (determines axial frequency)
    k_field = 1e10  # Arbitrary units for visualization

    for i, mz in enumerate(mz_values):
        # Axial frequency: ω_z = √(k × z/m)
        omega_z = np.sqrt(k_field / mz) / 1e4  # Normalized

        # Radial frequency (approximately 2× axial)
        omega_r = 0.3 * omega_z

        # Orbital radius (decreasing with m/z due to injection)
        r_orbit = 0.02 + 0.005 * (1000/mz)

        # Trajectory: rotating + axial oscillation
        x = r_orbit * np.cos(omega_r * t)
        y = r_orbit * np.sin(omega_r * t)
        z = 0.03 * np.cos(omega_z * t)  # Axial oscillation

        ax1.plot(x, y, z, color=colors[i], linewidth=1, label=f'm/z = {mz}', alpha=0.8)

    # Draw central electrode (spindle)
    theta_elec = np.linspace(0, 2*np.pi, 50)
    z_elec = np.linspace(-0.04, 0.04, 20)
    THETA, Z_E = np.meshgrid(theta_elec, z_elec)
    R_elec = 0.008 * (1 + 0.3 * (Z_E/0.04)**2)  # Spindle shape
    X_elec = R_elec * np.cos(THETA)
    Y_elec = R_elec * np.sin(THETA)
    ax1.plot_surface(X_elec, Y_elec, Z_E, color='gray', alpha=0.3)

    ax1.set_xlabel('X (mm)', fontsize=9)
    ax1.set_ylabel('Y (mm)', fontsize=9)
    ax1.set_zlabel('Z (mm)', fontsize=9)
    ax1.legend(loc='upper left', fontsize=8)

    # === Chart B: Axial Frequency vs m/z (3D surface) ===
    ax2 = fig.add_subplot(2, 2, 2, projection='3d')
    ax2.set_title('B) Axial Oscillation Frequency\nω = √(k/m)', fontsize=11)

    # Create surface: frequency as function of m/z and field strength
    mz_range = np.linspace(100, 2000, 50)
    k_range = np.linspace(0.5, 2, 50)  # Relative field strength
    MZ, K = np.meshgrid(mz_range, k_range)

    # Axial frequency (in kHz for visualization)
    # ω = √(k/m)
    k_base = 1e10
    freq_kHz = np.sqrt(k_base * K / (MZ * AMU)) / (2 * np.pi) / 1e3

    surf = ax2.plot_surface(MZ, K, freq_kHz, cmap='viridis', alpha=0.85,
                            linewidth=0.1, antialiased=True)

    ax2.set_xlabel('m/z', fontsize=9)
    ax2.set_ylabel('Field Strength (rel.)', fontsize=9)
    ax2.set_zlabel('Frequency (kHz)', fontsize=9)

    cbar = plt.colorbar(surf, ax=ax2, shrink=0.6, pad=0.1)
    cbar.set_label('f_axial (kHz)', fontsize=9)

    # === Chart C: Electrostatic Potential (3D) ===
    ax3 = fig.add_subplot(2, 2, 3, projection='3d')
    ax3.set_title('C) Orbitrap Electrostatic Potential\nQuadro-logarithmic field', fontsize=11)

    # Orbitrap potential: U(r,z) = (k/2)[z² - r²/2] + (k/2) × R² × ln(r/R)
    r = np.linspace(0.01, 0.05, 50)
    z = np.linspace(-0.04, 0.04, 50)
    R, Z = np.meshgrid(r, z)

    # Quadro-logarithmic potential
    R_m = 0.01  # Central electrode radius
    k_pot = 1e4
    U = (k_pot/2) * (Z**2 - R**2/2) + (k_pot/2) * R_m**2 * np.log(R/R_m)

    # Normalize for visualization
    U = (U - U.min()) / (U.max() - U.min())

    surf3 = ax3.plot_surface(R * 1000, Z * 1000, U, cmap='coolwarm', alpha=0.85,
                             linewidth=0.1, antialiased=True)

    # Add contour on bottom
    ax3.contour(R * 1000, Z * 1000, U, zdir='z', offset=0, cmap='coolwarm', alpha=0.5, levels=15)

    ax3.set_xlabel('r (mm)', fontsize=9)
    ax3.set_ylabel('z (mm)', fontsize=9)
    ax3.set_zlabel('Potential (norm.)', fontsize=9)

    cbar3 = plt.colorbar(surf3, ax=ax3, shrink=0.6, pad=0.1)
    cbar3.set_label('U (normalized)', fontsize=9)

    # === Chart D: Resolution vs m/z and Transient Time ===
    ax4 = fig.add_subplot(2, 2, 4, projection='3d')
    ax4.set_title('D) Orbitrap Resolution\nR ∝ f × T_transient', fontsize=11)

    # Resolution depends on frequency and observation time
    mz_range = np.linspace(200, 2000, 50)
    t_transient = np.linspace(32, 512, 50)  # Transient time in ms
    MZ, T_trans = np.meshgrid(mz_range, t_transient)

    # Resolution: R = f × T / (2 × Δf_min)
    # f ∝ 1/√m, so R ∝ T/√m
    k_base = 1e10
    freq = np.sqrt(k_base / (MZ * AMU)) / (2 * np.pi)
    Resolution = freq * T_trans / 1000 * 50  # Scaling factor

    # Normalize to typical Orbitrap range (50k - 500k)
    Resolution = Resolution / Resolution.max() * 500000

    surf4 = ax4.plot_surface(MZ, T_trans, Resolution / 1000, cmap='plasma', alpha=0.85,
                             linewidth=0.1, antialiased=True)

    # Add typical operating point
    ax4.scatter([400], [256], [240], color='red', s=100, marker='*',
               label='Typical: 240k @ m/z 400')

    ax4.set_xlabel('m/z', fontsize=9)
    ax4.set_ylabel('Transient Time (ms)', fontsize=9)
    ax4.set_zlabel('Resolution (×10³)', fontsize=9)
    ax4.legend(loc='upper right', fontsize=8)

    cbar4 = plt.colorbar(surf4, ax=ax4, shrink=0.6, pad=0.1)
    cbar4.set_label('R (thousands)', fontsize=9)

    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    plt.savefig(save_path, dpi=300, facecolor='white', edgecolor='none', bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

def main():
    """Generate FT-ICR and Orbitrap panels."""
    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'figures')
    os.makedirs(output_dir, exist_ok=True)

    # Also save to precursor location to match existing TOF panel
    precursor_dir = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..',
                                 'precursor', 'docs', 'union-of-two-crowns', 'figures')
    os.makedirs(precursor_dir, exist_ok=True)

    print("Generating Mass Spectrometer Instrument Panels...")
    print("=" * 60)

    # Generate FT-ICR panel
    fticr_path1 = os.path.join(output_dir, '04_fticr_mass_spectrometer.png')
    fticr_path2 = os.path.join(precursor_dir, '04_fticr_mass_spectrometer.png')
    generate_fticr_panel(fticr_path1)
    generate_fticr_panel(fticr_path2)

    # Generate Orbitrap panel
    orbitrap_path1 = os.path.join(output_dir, '05_orbitrap_mass_spectrometer.png')
    orbitrap_path2 = os.path.join(precursor_dir, '05_orbitrap_mass_spectrometer.png')
    generate_orbitrap_panel(orbitrap_path1)
    generate_orbitrap_panel(orbitrap_path2)

    print("=" * 60)
    print("All mass spectrometer panels generated successfully!")
    print(f"\nFigures saved to:")
    print(f"  - {output_dir}")
    print(f"  - {precursor_dir}")

if __name__ == '__main__':
    main()
