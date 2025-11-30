import numpy as np
import plotly.graph_objects as go

def plot_sentropy_trajectories_3d(data_dict):
    """
    Plot 3D trajectories in S-entropy space

    Parameters:
    -----------
    data_dict : dict
        Dictionary with ion numbers as keys and
        (coordinates_list, numpy_array) as values
    """
    fig = go.Figure()

    # Color palette for different ions
    colors = ['red', 'blue', 'green', 'orange', 'purple',
              'cyan', 'magenta', 'yellow', 'brown', 'pink']

    for ion_num, (coords_list, coords_array) in data_dict.items():
        # Extract coordinates
        s_knowledge = coords_array[:, 0]
        s_time = coords_array[:, 1]
        s_entropy = coords_array[:, 2]

        # Plot trajectory
        fig.add_trace(go.Scatter3d(
            x=s_knowledge,
            y=s_time,
            z=s_entropy,
            mode='lines+markers',
            name=f'Ion {ion_num}',
            line=dict(
                color=colors[ion_num % len(colors)],
                width=3
            ),
            marker=dict(
                size=3,
                color=colors[ion_num % len(colors)],
                opacity=0.6
            )
        ))

    # Layout
    fig.update_layout(
        title='S-entropy Trajectories in 3D Space',
        scene=dict(
            xaxis_title='S-knowledge',
            yaxis_title='S-time',
            zaxis_title='S-entropy',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.2)
            )
        ),
        width=1000,
        height=800,
        showlegend=True
    )

    return fig

# Use it
fig = plot_sentropy_trajectories_3d(your_data_dict)
fig.show()


import matplotlib.pyplot as plt

def plot_sentropy_projections(data_dict):
    """
    Plot 2D projections of S-entropy trajectories
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    colors = plt.cm.tab10(np.linspace(0, 1, len(data_dict)))

    for idx, (ion_num, (coords_list, coords_array)) in enumerate(data_dict.items()):
        s_knowledge = coords_array[:, 0]
        s_time = coords_array[:, 1]
        s_entropy = coords_array[:, 2]

        # s_knowledge vs s_entropy (MOST IMPORTANT)
        axes[0].plot(s_knowledge, s_entropy,
                    color=colors[idx],
                    label=f'Ion {ion_num}',
                    linewidth=2, alpha=0.7)
        axes[0].scatter(s_knowledge[0], s_entropy[0],
                       color=colors[idx], s=100, marker='o',
                       edgecolor='black', linewidth=2, zorder=5)
        axes[0].scatter(s_knowledge[-1], s_entropy[-1],
                       color=colors[idx], s=100, marker='s',
                       edgecolor='black', linewidth=2, zorder=5)

        # s_time vs s_entropy
        axes[1].plot(s_time, s_entropy,
                    color=colors[idx],
                    linewidth=2, alpha=0.7)
        axes[1].scatter(s_time[0], s_entropy[0],
                       color=colors[idx], s=100, marker='o',
                       edgecolor='black', linewidth=2, zorder=5)
        axes[1].scatter(s_time[-1], s_entropy[-1],
                       color=colors[idx], s=100, marker='s',
                       edgecolor='black', linewidth=2, zorder=5)

        # s_knowledge vs s_time
        axes[2].plot(s_knowledge, s_time,
                    color=colors[idx],
                    linewidth=2, alpha=0.7)
        axes[2].scatter(s_knowledge[0], s_time[0],
                       color=colors[idx], s=100, marker='o',
                       edgecolor='black', linewidth=2, zorder=5)
        axes[2].scatter(s_knowledge[-1], s_time[-1],
                       color=colors[idx], s=100, marker='s',
                       edgecolor='black', linewidth=2, zorder=5)

    # Labels
    axes[0].set_xlabel('S-knowledge', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('S-entropy', fontsize=14, fontweight='bold')
    axes[0].set_title('Knowledge-Entropy Projection', fontsize=16, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('S-time', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('S-entropy', fontsize=14, fontweight='bold')
    axes[1].set_title('Time-Entropy Projection', fontsize=16, fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    axes[2].set_xlabel('S-knowledge', fontsize=14, fontweight='bold')
    axes[2].set_ylabel('S-time', fontsize=14, fontweight='bold')
    axes[2].set_title('Knowledge-Time Projection', fontsize=16, fontweight='bold')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig

# Use it
fig = plot_sentropy_projections(your_data_dict)
plt.savefig('sentropy_projections.png', dpi=300, bbox_inches='tight')
plt.show()
from scipy.stats import gaussian_kde

def plot_sentropy_density(data_dict):
    """
    Plot density heatmap of S-entropy trajectories
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Combine all trajectories
    all_knowledge = []
    all_time = []
    all_entropy = []

    for ion_num, (coords_list, coords_array) in data_dict.items():
        all_knowledge.extend(coords_array[:, 0])
        all_time.extend(coords_array[:, 1])
        all_entropy.extend(coords_array[:, 2])

    all_knowledge = np.array(all_knowledge)
    all_time = np.array(all_time)
    all_entropy = np.array(all_entropy)

    # Knowledge-Entropy density
    xy = np.vstack([all_knowledge, all_entropy])
    z = gaussian_kde(xy)(xy)
    axes[0].scatter(all_knowledge, all_entropy, c=z, s=10, cmap='viridis')
    axes[0].set_xlabel('S-knowledge', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('S-entropy', fontsize=14, fontweight='bold')
    axes[0].set_title('Knowledge-Entropy Density', fontsize=16, fontweight='bold')

    # Time-Entropy density
    xy = np.vstack([all_time, all_entropy])
    z = gaussian_kde(xy)(xy)
    axes[1].scatter(all_time, all_entropy, c=z, s=10, cmap='viridis')
    axes[1].set_xlabel('S-time', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('S-entropy', fontsize=14, fontweight='bold')
    axes[1].set_title('Time-Entropy Density', fontsize=16, fontweight='bold')

    # Knowledge-Time density
    xy = np.vstack([all_knowledge, all_time])
    z = gaussian_kde(xy)(xy)
    im = axes[2].scatter(all_knowledge, all_time, c=z, s=10, cmap='viridis')
    axes[2].set_xlabel('S-knowledge', fontsize=14, fontweight='bold')
    axes[2].set_ylabel('S-time', fontsize=14, fontweight='bold')
    axes[2].set_title('Knowledge-Time Density', fontsize=16, fontweight='bold')

    plt.colorbar(im, ax=axes[2], label='Density')
    plt.tight_layout()
    return fig

# Use it
fig = plot_sentropy_density(your_data_dict)
plt.savefig('sentropy_density.png', dpi=300, bbox_inches='tight')
plt.show()
