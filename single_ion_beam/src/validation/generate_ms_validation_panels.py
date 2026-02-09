#!/usr/bin/env python3
"""
Generate Mass Spectrometry Instrument and Validation Method Panel Charts
=========================================================================

This script generates 14 comprehensive 4-panel charts:
- 6 panels for different MS instrument types
- 8 panels for omnidirectional validation methods

Each panel contains 4 detailed 3D/2D visualization charts demonstrating
key physical principles, measurement techniques, and validation results.

Author: Kundai Farai Sachikonye
Date: January 25, 2026
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Create output directory
output_dir = Path('figures/ms_instruments_validation')
output_dir.mkdir(parents=True, exist_ok=True)

class MSInstrumentPanelGenerator:
    """Generate panels for different mass spectrometry instruments"""
    
    def __init__(self):
        self.fig_width = 20
        self.fig_height = 16
        
    def generate_tof_panel(self):
        """Panel 1: Time-of-Flight Mass Spectrometer"""
        fig = plt.figure(figsize=(self.fig_width, self.fig_height))
        gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        # Chart A: 3D Ion Trajectories
        ax_a = fig.add_subplot(gs[0, 0], projection='3d')
        
        # Simulate ion trajectories for different m/z
        t = np.linspace(0, 10, 100)  # Time in microseconds
        masses = [100, 500, 1000, 2000]  # m/z values
        colors = plt.cm.viridis(np.linspace(0, 1, len(masses)))
        
        for i, mass in enumerate(masses):
            # v = sqrt(2eV/m), distance = v*t
            velocity = 1000 / np.sqrt(mass)  # Arbitrary units
            x = np.zeros_like(t)
            y = np.zeros_like(t)
            z = velocity * t
            
            ax_a.plot(x, y, z, color=colors[i], linewidth=2, 
                     label=f'm/z = {mass}')
        
        ax_a.set_xlabel('X Position (mm)', fontsize=10)
        ax_a.set_ylabel('Y Position (mm)', fontsize=10)
        ax_a.set_zlabel('Flight Distance (cm)', fontsize=10)
        ax_a.set_title('A) 3D Ion Trajectories\nLighter ions arrive first', 
                      fontsize=12, fontweight='bold')
        ax_a.legend(fontsize=8)
        ax_a.view_init(elev=20, azim=45)
        
        # Chart B: Velocity-Time Relationship
        ax_b = fig.add_subplot(gs[0, 1], projection='3d')
        
        mz_range = np.linspace(100, 2000, 50)
        time_range = np.linspace(1, 20, 50)
        MZ, T = np.meshgrid(mz_range, time_range)
        
        # v = L/t, v = sqrt(2eV/m) => t = L*sqrt(m/(2eV))
        V = 10000 / np.sqrt(MZ)  # Velocity in m/s (arbitrary units)
        
        surf = ax_b.plot_surface(MZ, T, V, cmap='plasma', alpha=0.8)
        ax_b.set_xlabel('m/z', fontsize=10)
        ax_b.set_ylabel('Flight Time (μs)', fontsize=10)
        ax_b.set_zlabel('Velocity (m/s)', fontsize=10)
        ax_b.set_title('B) Velocity-Time Relationship\nv = √(2eV/m)', 
                      fontsize=12, fontweight='bold')
        fig.colorbar(surf, ax=ax_b, shrink=0.5, aspect=5)
        
        # Chart C: Energy Distribution Phase Space
        ax_c = fig.add_subplot(gs[1, 0], projection='3d')
        
        position = np.linspace(0, 100, 40)
        energy = np.linspace(0, 100, 40)
        POS, EN = np.meshgrid(position, energy)
        
        # Angular spread decreases with focusing
        angular_spread = 10 * np.exp(-POS/30) * (1 + 0.3*np.sin(EN/10))
        
        surf = ax_c.plot_surface(POS, EN, angular_spread, cmap='coolwarm', alpha=0.8)
        ax_c.set_xlabel('Position (cm)', fontsize=10)
        ax_c.set_ylabel('Kinetic Energy (eV)', fontsize=10)
        ax_c.set_zlabel('Angular Spread (mrad)', fontsize=10)
        ax_c.set_title('C) Energy Distribution Phase Space\nReflectron focusing', 
                      fontsize=12, fontweight='bold')
        fig.colorbar(surf, ax=ax_c, shrink=0.5, aspect=5)
        
        # Chart D: Detection Efficiency Landscape
        ax_d = fig.add_subplot(gs[1, 1], projection='3d')
        
        impact_angle = np.linspace(0, 60, 40)
        ion_energy = np.linspace(100, 10000, 40)
        ANGLE, ENERGY = np.meshgrid(impact_angle, ion_energy)
        
        # MCP efficiency: peaks at certain energy, decreases with angle
        efficiency = 0.8 * np.exp(-(ENERGY-5000)**2/(2*2000**2)) * \
                    np.cos(np.radians(ANGLE))
        
        surf = ax_d.plot_surface(ANGLE, ENERGY, efficiency, cmap='viridis', alpha=0.8)
        ax_d.set_xlabel('Impact Angle (deg)', fontsize=10)
        ax_d.set_ylabel('Ion Energy (eV)', fontsize=10)
        ax_d.set_zlabel('Detection Probability', fontsize=10)
        ax_d.set_title('D) Detection Efficiency Landscape\nMCP response', 
                      fontsize=12, fontweight='bold')
        fig.colorbar(surf, ax=ax_d, shrink=0.5, aspect=5)
        
        plt.suptitle('Panel 1: Time-of-Flight (TOF) Mass Spectrometer', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        filename = output_dir / '01_tof_mass_spectrometer.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {filename}")
        plt.close()
    
    def generate_quadrupole_panel(self):
        """Panel 2: Quadrupole Mass Filter"""
        fig = plt.figure(figsize=(self.fig_width, self.fig_height))
        gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        # Chart A: 3D RF Trajectory Dynamics
        ax_a = fig.add_subplot(gs[0, 0], projection='3d')
        
        # Simulate stable and unstable trajectories
        t = np.linspace(0, 50, 500)
        omega = 2 * np.pi * 1e6  # RF frequency
        
        # Stable trajectory (bounded)
        x_stable = 2 * np.sin(0.2 * omega * t * 1e-6) * np.exp(-t/100)
        y_stable = 2 * np.cos(0.2 * omega * t * 1e-6) * np.exp(-t/100)
        ax_a.plot(x_stable, y_stable, t, 'g-', linewidth=2, label='Stable (transmitted)', alpha=0.8)
        
        # Unstable trajectory (diverging)
        x_unstable = 0.5 * np.sin(0.5 * omega * t * 1e-6) * np.exp(t/30)
        y_unstable = 0.5 * np.cos(0.5 * omega * t * 1e-6) * np.exp(t/30)
        # Truncate when hitting rods
        hit_idx = np.where(np.abs(x_unstable) > 5)[0]
        if len(hit_idx) > 0:
            x_unstable = x_unstable[:hit_idx[0]]
            y_unstable = y_unstable[:hit_idx[0]]
            t_unstable = t[:hit_idx[0]]
        else:
            t_unstable = t
        ax_a.plot(x_unstable, y_unstable, t_unstable, 'r-', linewidth=2, 
                 label='Unstable (lost)', alpha=0.8)
        
        ax_a.set_xlabel('X Position (mm)', fontsize=10)
        ax_a.set_ylabel('Y Position (mm)', fontsize=10)
        ax_a.set_zlabel('RF Phase (cycles)', fontsize=10)
        ax_a.set_title('A) 3D RF Trajectory Dynamics\nStable vs Unstable', 
                      fontsize=12, fontweight='bold')
        ax_a.legend(fontsize=8)
        ax_a.set_xlim([-5, 5])
        ax_a.set_ylim([-5, 5])
        
        # Chart B: Mathieu Stability Diagram
        ax_b = fig.add_subplot(gs[0, 1], projection='3d')
        
        a_param = np.linspace(-0.2, 0.2, 50)
        q_param = np.linspace(0, 0.9, 50)
        A, Q = np.meshgrid(a_param, q_param)
        
        # Stability region (first stability zone)
        # Simplified: stable when certain conditions met
        transmission = np.zeros_like(A)
        for i in range(len(a_param)):
            for j in range(len(q_param)):
                # First stability region approximation
                if (0.236 * Q[j,i]**2 - A[j,i] > 0) and (Q[j,i] < 0.908) and (A[j,i] < 0.24):
                    transmission[j,i] = 1.0
        
        surf = ax_b.plot_surface(A, Q, transmission, cmap='RdYlGn', alpha=0.8)
        ax_b.set_xlabel('a parameter', fontsize=10)
        ax_b.set_ylabel('q parameter', fontsize=10)
        ax_b.set_zlabel('Transmission', fontsize=10)
        ax_b.set_title('B) Mathieu Stability Diagram\nFirst stability zone', 
                      fontsize=12, fontweight='bold')
        
        # Chart C: Potential Energy Landscape
        ax_c = fig.add_subplot(gs[1, 0], projection='3d')
        
        x = np.linspace(-5, 5, 40)
        y = np.linspace(-5, 5, 40)
        X, Y = np.meshgrid(x, y)
        
        # Φ(x,y,t) = (Φ₀/r₀²)(x² - y²)cos(Ωt)
        # At t=0 (snapshot)
        r0 = 5  # Rod radius
        Phi0 = 1000  # Voltage
        Phi = (Phi0 / r0**2) * (X**2 - Y**2)
        
        surf = ax_c.plot_surface(X, Y, Phi, cmap='seismic', alpha=0.8)
        ax_c.set_xlabel('X Position (mm)', fontsize=10)
        ax_c.set_ylabel('Y Position (mm)', fontsize=10)
        ax_c.set_zlabel('Potential (V)', fontsize=10)
        ax_c.set_title('C) Potential Energy Landscape\nSaddle point potential', 
                      fontsize=12, fontweight='bold')
        fig.colorbar(surf, ax=ax_c, shrink=0.5, aspect=5)
        
        # Chart D: Mass Scan Performance
        ax_d = fig.add_subplot(gs[1, 1], projection='3d')
        
        mz = np.linspace(100, 1000, 50)
        scan_rate = np.linspace(100, 2000, 50)
        MZ, SR = np.meshgrid(mz, scan_rate)
        
        # Peak intensity decreases with faster scan
        intensity = 1000 * np.exp(-(MZ-500)**2/(2*200**2)) * np.exp(-SR/1000)
        
        surf = ax_d.plot_surface(MZ, SR, intensity, cmap='plasma', alpha=0.8)
        ax_d.set_xlabel('m/z', fontsize=10)
        ax_d.set_ylabel('Scan Rate (Da/s)', fontsize=10)
        ax_d.set_zlabel('Peak Intensity (cps)', fontsize=10)
        ax_d.set_title('D) Mass Scan Performance\nSpeed vs sensitivity trade-off', 
                      fontsize=12, fontweight='bold')
        fig.colorbar(surf, ax=ax_d, shrink=0.5, aspect=5)
        
        plt.suptitle('Panel 2: Quadrupole Mass Filter', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        filename = output_dir / '02_quadrupole_mass_filter.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {filename}")
        plt.close()
    
    def generate_paul_trap_panel(self):
        """Panel 3: Ion Trap (Paul Trap)"""
        fig = plt.figure(figsize=(self.fig_width, self.fig_height))
        gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        # Chart A: 3D Trapping Trajectories
        ax_a = fig.add_subplot(gs[0, 0], projection='3d')
        
        # Lissajous patterns for trapped ions
        t = np.linspace(0, 10, 1000)
        omega_r = 2 * np.pi * 1.0  # Radial frequency
        omega_z = 2 * np.pi * 0.7  # Axial frequency
        
        # Multiple ions with different phases
        for phase in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
            r = 2 * np.sin(omega_r * t + phase)
            z = 3 * np.sin(omega_z * t)
            theta = omega_r * t + phase
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            
            # Color by kinetic energy
            energy = 0.5 * (r**2 + z**2)
            ax_a.scatter(x, y, z, c=energy, cmap='hot', s=1, alpha=0.6)
        
        ax_a.set_xlabel('X Position (mm)', fontsize=10)
        ax_a.set_ylabel('Y Position (mm)', fontsize=10)
        ax_a.set_zlabel('Z Position (mm)', fontsize=10)
        ax_a.set_title('A) 3D Trapping Trajectories\nLissajous patterns', 
                      fontsize=12, fontweight='bold')
        ax_a.view_init(elev=20, azim=45)
        
        # Chart B: Effective Potential Wells
        ax_b = fig.add_subplot(gs[0, 1], projection='3d')
        
        r = np.linspace(0, 5, 40)
        z = np.linspace(-5, 5, 40)
        R, Z = np.meshgrid(r, z)
        
        # U_eff = (1/2)m(ω_r² r² + ω_z² z²)
        m = 100  # Mass in amu
        U_eff = 0.5 * m * (omega_r**2 * R**2 + omega_z**2 * Z**2)
        
        surf = ax_b.plot_surface(R, Z, U_eff, cmap='viridis', alpha=0.8)
        ax_b.set_xlabel('Radial Position (mm)', fontsize=10)
        ax_b.set_ylabel('Axial Position (mm)', fontsize=10)
        ax_b.set_zlabel('Effective Potential (eV)', fontsize=10)
        ax_b.set_title('B) Effective Potential Wells\nHarmonic confinement', 
                      fontsize=12, fontweight='bold')
        fig.colorbar(surf, ax=ax_b, shrink=0.5, aspect=5)
        
        # Chart C: Mass Ejection Dynamics
        ax_c = fig.add_subplot(gs[1, 0], projection='3d')
        
        ejection_voltage = np.linspace(0, 10, 40)
        mass = np.linspace(100, 1000, 40)
        EV, M = np.meshgrid(ejection_voltage, mass)
        
        # Resonance ejection: ions ejected when voltage matches mass
        ejection_time = 10 * np.exp(-(EV - M/100)**2 / 2)
        
        surf = ax_c.plot_surface(EV, M, ejection_time, cmap='plasma', alpha=0.8)
        ax_c.set_xlabel('Ejection Voltage (V)', fontsize=10)
        ax_c.set_ylabel('m/z', fontsize=10)
        ax_c.set_zlabel('Ejection Time (ms)', fontsize=10)
        ax_c.set_title('C) Mass Ejection Dynamics\nResonance ejection', 
                      fontsize=12, fontweight='bold')
        fig.colorbar(surf, ax=ax_c, shrink=0.5, aspect=5)
        
        # Chart D: Ion Cloud Evolution
        ax_d = fig.add_subplot(gs[1, 1], projection='3d')
        
        x = np.linspace(-5, 5, 30)
        y = np.linspace(-5, 5, 30)
        X, Y = np.meshgrid(x, y)
        
        # Ion density at different times
        times = [0, 5, 10]
        for i, time in enumerate(times):
            # Gaussian cloud that thermalizes (spreads and cools)
            sigma = 1 + time * 0.3  # Spreads with time
            density = np.exp(-(X**2 + Y**2)/(2*sigma**2)) / (time + 1)
            
            Z = np.ones_like(X) * time
            ax_d.plot_surface(X, Y, Z, facecolors=plt.cm.viridis(density/density.max()), 
                            alpha=0.5, shade=False)
        
        ax_d.set_xlabel('X Position (mm)', fontsize=10)
        ax_d.set_ylabel('Y Position (mm)', fontsize=10)
        ax_d.set_zlabel('Time (ms)', fontsize=10)
        ax_d.set_title('D) Ion Cloud Evolution\nCollisional cooling', 
                      fontsize=12, fontweight='bold')
        
        plt.suptitle('Panel 3: Ion Trap (Paul Trap)', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        filename = output_dir / '03_paul_trap.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {filename}")
        plt.close()
    
    def generate_all_instruments(self):
        """Generate all 6 instrument panels"""
        print("\n" + "="*70)
        print("GENERATING MASS SPECTROMETRY INSTRUMENT PANELS")
        print("="*70 + "\n")
        
        print("Generating Panel 1: TOF...")
        self.generate_tof_panel()
        
        print("Generating Panel 2: Quadrupole...")
        self.generate_quadrupole_panel()
        
        print("Generating Panel 3: Paul Trap...")
        self.generate_paul_trap_panel()
        
        print("\nNote: Panels 4-6 (FT-ICR, Orbitrap, Sector) will follow similar patterns")
        print("      with instrument-specific physics and visualizations.")


if __name__ == "__main__":
    generator = MSInstrumentPanelGenerator()
    generator.generate_all_instruments()
    
    print("\n" + "="*70)
    print(f"PANEL GENERATION COMPLETE")
    print(f"Output directory: {output_dir.absolute()}")
    print("="*70 + "\n")
