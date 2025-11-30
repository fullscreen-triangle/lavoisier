# 3D trajectory with state annotations
def plot_categorical_progression(data_dict):
    fig = go.Figure()

    # Plot trajectory
    for state_num, (coords_list, coords_array) in data_dict.items():
        s_k = coords_array[:, 0]
        s_t = coords_array[:, 1]
        s_e = coords_array[:, 2]

        fig.add_trace(go.Scatter3d(
            x=s_k, y=s_t, z=s_e,
            mode='lines+markers',
            name=f'C{state_num}',
            line=dict(width=5),
            marker=dict(size=8)
        ))

        # Annotate start/end
        fig.add_trace(go.Scatter3d(
            x=[s_k[0]], y=[s_t[0]], z=[s_e[0]],
            mode='markers+text',
            text=[f'C{state_num} start'],
            marker=dict(size=15, symbol='diamond')
        ))

    fig.update_layout(
        scene=dict(
            xaxis_title='S-knowledge',
            yaxis_title='S-time',
            zaxis_title='S-entropy'
        ),
        title='Categorical State Progression: C₀ → C₁ → C₂ → C₃'
    )
    return fig


# Overlay Waters + Thermo data
def plot_platform_independence(waters_data, thermo_data):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # For each projection
    for ax, (x_idx, y_idx, xlabel, ylabel) in zip(
        axes,
        [(0, 2, 'S-knowledge', 'S-entropy'),
         (1, 2, 'S-time', 'S-entropy'),
         (0, 1, 'S-knowledge', 'S-time')]
    ):
        # Waters data (blue)
        for ion_num, (_, coords) in waters_data.items():
            ax.plot(coords[:, x_idx], coords[:, y_idx],
                   'b-', alpha=0.7, linewidth=2)

        # Thermo data (red, dashed)
        for ion_num, (_, coords) in thermo_data.items():
            ax.plot(coords[:, x_idx], coords[:, y_idx],
                   'r--', alpha=0.7, linewidth=2)

        ax.set_xlabel(xlabel, fontsize=14, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
        ax.legend(['Waters Q-TOF', 'Thermo Orbitrap'])
        ax.grid(True, alpha=0.3)

    # Calculate and annotate CV
    cv = calculate_cv(waters_data, thermo_data)
    fig.suptitle(f'Platform Independence: CV = {cv:.2f}%',
                fontsize=16, fontweight='bold')

    return fig
# Log-linear plot showing I ∝ exp(-|E|/⟨E⟩)
def plot_intensity_entropy(fragments_data):
    fig, ax = plt.subplots(figsize=(10, 8))

    # Extract edge density and intensity
    edge_density = [calc_edge_density(frag) for frag in fragments_data]
    intensity = [frag['intensity'] for frag in fragments_data]

    # Log scale
    ax.semilogy(edge_density, intensity, 'o',
               markersize=10, alpha=0.6)

    # Fit exponential
    from scipy.optimize import curve_fit
    def exp_model(x, a, b):
        return a * np.exp(-b * x)

    popt, _ = curve_fit(exp_model, edge_density, intensity)
    x_fit = np.linspace(min(edge_density), max(edge_density), 100)
    y_fit = exp_model(x_fit, *popt)
    ax.plot(x_fit, y_fit, 'r-', linewidth=3,
           label=f'I = {popt[0]:.2e} exp(-{popt[1]:.3f}|E|)')

    ax.set_xlabel('Phase-lock Edge Density |E|',
                 fontsize=14, fontweight='bold')
    ax.set_ylabel('Fragment Intensity',
                 fontsize=14, fontweight='bold')
    ax.set_title('Intensity as Termination Probability',
                fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    return fig
