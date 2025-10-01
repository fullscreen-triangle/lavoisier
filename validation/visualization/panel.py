import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyBboxPatch
import seaborn as sns

# Set style for all plots
plt.style.use('default')
sns.set_palette("husl")

def create_panel_figure(nrows, ncols, figsize=(20, 15)):
    """Create a panel figure with organized subplots"""
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if nrows == 1 and ncols == 1:
        axes = [axes]
    elif nrows == 1 or ncols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    return fig, axes

# PANEL 1: OSCILLATORY REALITY FOUNDATIONS
def plot_oscillatory_foundations():
    """Panel showing core oscillatory reality concepts"""
    fig, axes = create_panel_figure(2, 2, figsize=(16, 12))
    
    # Plot 1A: 95%/5% Reality Split
    sizes = [95, 5]
    labels = ['Continuous\nOscillatory (95%)', 'Discrete\nApproximations (5%)']
    colors = ['#2E86AB', '#A23B72']
    axes[0].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    axes[0].set_title('A. Reality Information Distribution', fontweight='bold', fontsize=14)
    
    # Plot 1B: Self-Sustaining Loop
    angles = np.linspace(0, 2*np.pi, 4, endpoint=False)
    domains = ['Math', 'Physics', 'Observation', 'Consciousness']
    for i, (angle, domain) in enumerate(zip(angles, domains)):
        x, y = 2*np.cos(angle), 2*np.sin(angle)
        circle = Circle((x, y), 0.5, color=colors[i%2], alpha=0.7)
        axes[1].add_patch(circle)
        axes[1].text(x, y, domain, ha='center', va='center', fontweight='bold')
    
    # Add arrows between domains
    for i in range(4):
        start_angle = angles[i]
        end_angle = angles[(i+1)%4]
        x1, y1 = 1.5*np.cos(start_angle), 1.5*np.sin(start_angle)
        x2, y2 = 1.5*np.cos(end_angle), 1.5*np.sin(end_angle)
        axes[1].annotate('', xy=(x2, y2), xytext=(x1, y1),
                        arrowprops=dict(arrowstyle='->', lw=2))
    
    axes[1].set_xlim(-3, 3)
    axes[1].set_ylim(-3, 3)
    axes[1].set_aspect('equal')
    axes[1].set_title('B. Self-Sustaining Reality Loop', fontweight='bold', fontsize=14)
    axes[1].axis('off')
    
    # Plot 1C: Mathematical Necessity Chain
    steps = ['Self-Consistent\nStructure', 'Must Contain\n"I Exist"', 'Statement Must\nBe True', 'Requires Dynamic\nMaintenance']
    y_positions = [3, 2, 1, 0]
    
    for i, (step, y) in enumerate(zip(steps, y_positions)):
        box = FancyBboxPatch((-0.8, y-0.2), 1.6, 0.4, boxstyle="round,pad=0.05", 
                           facecolor=colors[i%2], alpha=0.7)
        axes[2].add_patch(box)
        axes[2].text(0, y, step, ha='center', va='center', fontweight='bold')
        
        if i < len(steps)-1:
            axes[2].annotate('', xy=(0, y_positions[i+1]+0.2), xytext=(0, y-0.2),
                           arrowprops=dict(arrowstyle='->', lw=2))
    
    axes[2].set_xlim(-1.5, 1.5)
    axes[2].set_ylim(-0.5, 3.5)
    axes[2].set_title('C. Mathematical Necessity Chain', fontweight='bold', fontsize=14)
    axes[2].axis('off')
    
    # Plot 1D: Oscillatory vs Discrete Signals
    t = np.linspace(0, 4*np.pi, 1000)
    continuous = np.sin(t) + 0.3*np.sin(3*t) + 0.1*np.sin(7*t)
    t_discrete = np.linspace(0, 4*np.pi, 20)
    discrete = np.sin(t_discrete) + 0.3*np.sin(3*t_discrete) + 0.1*np.sin(7*t_discrete)
    
    axes[3].plot(t, continuous, color=colors[0], linewidth=2, label='Continuous (95%)')
    axes[3].scatter(t_discrete, discrete, color=colors[1], s=50, label='Discrete (5%)')
    axes[3].set_xlabel('Time')
    axes[3].set_ylabel('Amplitude')
    axes[3].set_title('D. Continuous vs Discrete Information', fontweight='bold', fontsize=14)
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

# PANEL 2: S-ENTROPY COORDINATE NAVIGATION
def plot_sentropy_navigation():
    """Panel showing S-entropy coordinate navigation concepts"""
    fig, axes = create_panel_figure(2, 2, figsize=(16, 12))
    
    # Plot 2A: Complexity Comparison
    n_values = np.logspace(1, 5, 50)
    traditional = n_values ** 2
    sentropy = np.ones_like(n_values) * 100
    
    axes[0].loglog(n_values, traditional, 'r-', linewidth=3, label='Traditional O(N²)')
    axes[0].loglog(n_values, sentropy, 'g-', linewidth=3, label='S-Entropy O(1)')
    axes[0].set_xlabel('Dataset Size (N)')
    axes[0].set_ylabel('Processing Time')
    axes[0].set_title('A. Computational Complexity Comparison', fontweight='bold', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2B: Navigation Paths
    # Traditional random walk
    np.random.seed(42)
    traditional_path = np.cumsum(np.random.randn(50, 2) * 0.2, axis=0)
    
    # Direct S-entropy path
    start = np.array([0, 0])
    end = np.array([3, 2])
    direct_path = np.array([start + t*(end-start) for t in np.linspace(0, 1, 10)])
    
    axes[1].plot(traditional_path[:, 0], traditional_path[:, 1], 'r-', linewidth=2, 
                marker='o', markersize=4, label='Traditional Search')
    axes[1].plot(direct_path[:, 0], direct_path[:, 1], 'g-', linewidth=3, 
                marker='s', markersize=6, label='S-Entropy Direct')
    axes[1].scatter([end[0]], [end[1]], color='gold', s=200, marker='*', 
                   label='Target Molecule', zorder=5)
    axes[1].set_xlabel('Molecular Space X')
    axes[1].set_ylabel('Molecular Space Y')
    axes[1].set_title('B. Navigation Path Comparison', fontweight='bold', fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Plot 2C: Information Access Coverage
    molecular_types = ['Amino\nAcids', 'Nucleotides', 'Carbs', 'Lipids', 'Metabolites']
    traditional_coverage = [45, 38, 52, 41, 48]
    sentropy_coverage = [98, 97, 99, 96, 98]
    
    x = np.arange(len(molecular_types))
    width = 0.35
    axes[2].bar(x - width/2, traditional_coverage, width, label='Traditional (5%)', 
               color='red', alpha=0.7)
    axes[2].bar(x + width/2, sentropy_coverage, width, label='S-Entropy (95%)', 
               color='green', alpha=0.7)
    axes[2].set_xlabel('Molecular Class')
    axes[2].set_ylabel('Coverage (%)')
    axes[2].set_title('C. Molecular Information Coverage', fontweight='bold', fontsize=14)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(molecular_types)
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    # Plot 2D: Coordinate Transformation
    theta = np.linspace(0, 2*np.pi, 100)
    r = 1 + 0.3*np.sin(5*theta)
    x_orig = r * np.cos(theta)
    y_orig = r * np.sin(theta)
    
    # Transform to S-entropy coordinates
    x_transform = x_orig * np.cos(theta/2) - y_orig * np.sin(theta/2)
    y_transform = x_orig * np.sin(theta/2) + y_orig * np.cos(theta/2)
    
    axes[3].plot(x_orig, y_orig, 'b-', linewidth=2, label='Original Coordinates')
    axes[3].plot(x_transform, y_transform, 'g-', linewidth=2, label='S-Entropy Coordinates')
    axes[3].set_xlabel('Coordinate X')
    axes[3].set_ylabel('Coordinate Y')
    axes[3].set_title('D. Coordinate Transformation', fontweight='bold', fontsize=14)
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)
    axes[3].set_aspect('equal')
    
    plt.tight_layout()
    return fig

# PANEL 3: VALIDATION RESULTS
def plot_validation_results():
    """Panel showing experimental validation results"""
    fig, axes = create_panel_figure(2, 2, figsize=(16, 12))
    
    # Plot 3A: Accuracy Comparison
    methods = ['Traditional\nNumerical', 'Computer\nVision', 'S-Stellas\nPure']
    pl_neg = [78.5, 82.3, 98.9]
    tg_pos = [76.2, 84.1, 99.1]
    
    x = np.arange(len(methods))
    width = 0.35
    axes[0].bar(x - width/2, pl_neg, width, label='PL_Neg Dataset', alpha=0.8)
    axes[0].bar(x + width/2, tg_pos, width, label='TG_Pos Dataset', alpha=0.8)
    axes[0].set_xlabel('Method')
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_title('A. Molecular Identification Accuracy', fontweight='bold', fontsize=14)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(methods)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Add accuracy values on bars
    for i, (p, t) in enumerate(zip(pl_neg, tg_pos)):
        axes[0].text(i - width/2, p + 1, f'{p:.1f}%', ha='center', fontweight='bold')
        axes[0].text(i + width/2, t + 1, f'{t:.1f}%', ha='center', fontweight='bold')
    
    # Plot 3B: Processing Speed
    methods_speed = ['Traditional', 'S-Stellas']
    speeds = [1, 1000]  # relative speed
    
    axes[1].bar(methods_speed, speeds, color=['red', 'green'], alpha=0.7)
    axes[1].set_yscale('log')
    axes[1].set_ylabel('Relative Processing Speed')
    axes[1].set_title('B. Processing Speed Comparison', fontweight='bold', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    
    for i, speed in enumerate(speeds):
        axes[1].text(i, speed * 1.5, f'{speed}x', ha='center', fontweight='bold', fontsize=12)
    
    # Plot 3C: Enhancement Analysis
    base_methods = ['Numerical', 'Vision']
    enhancement = [14.3, 14.7]  # percentage improvement with S-Stellas
    
    axes[2].bar(base_methods, enhancement, color=['orange', 'blue'], alpha=0.7)
    axes[2].set_ylabel('Accuracy Enhancement (%)')
    axes[2].set_title('C. S-Stellas Enhancement Effect', fontweight='bold', fontsize=14)
    axes[2].grid(True, alpha=0.3)
    
    for i, enh in enumerate(enhancement):
        axes[2].text(i, enh + 0.5, f'+{enh:.1f}%', ha='center', fontweight='bold')
    
    # Plot 3D: Cross-Dataset Validation
    cross_validation = [72.3, 79.1, 96.7]  # accuracy retention across datasets
    
    axes[3].bar(methods, cross_validation, color=['red', 'blue', 'green'], alpha=0.7)
    axes[3].set_ylabel('Cross-Dataset Accuracy (%)')
    axes[3].set_title('D. Cross-Dataset Validation', fontweight='bold', fontsize=14)
    axes[3].set_xticks(x)
    axes[3].set_xticklabels(methods)
    axes[3].grid(True, alpha=0.3)
    
    for i, acc in enumerate(cross_validation):
        axes[3].text(i, acc + 1, f'{acc:.1f}%', ha='center', fontweight='bold')
    
    plt.tight_layout()
    return fig

# PANEL 4: BIOLOGICAL MAXWELL DEMONS
def plot_maxwell_demons():
    """Panel showing biological Maxwell demon performance"""
    fig, axes = create_panel_figure(2, 2, figsize=(16, 12))
    
    # Plot 4A: Recognition Network Architecture
    # Simplified network visualization
    layers = ['Input', 'Attention', 'Memory', 'Recognition', 'Output']
    layer_sizes = [8, 6, 4, 3, 1]
    
    for i, (layer, size) in enumerate(zip(layers, layer_sizes)):
        x = i + 1
        y_positions = np.linspace(1, 5, size)
        for y in y_positions:
            circle = Circle((x, y), 0.15, color=plt.cm.viridis(i/len(layers)), alpha=0.8)
            axes[0].add_patch(circle)
        
        axes[0].text(x, 0.5, layer, ha='center', va='center', fontweight='bold', rotation=45)
    
    axes[0].set_xlim(0.5, 5.5)
    axes[0].set_ylim(0, 6)
    axes[0].set_title('A. BMD Network Architecture', fontweight='bold', fontsize=14)
    axes[0].axis('off')
    
    # Plot 4B: Performance Metrics
    metrics = ['Speed', 'Accuracy', 'Memory\nEfficiency', 'Adaptability']
    traditional = [1, 75, 1, 60]
    bmd = [1000, 99.9, 1000, 95]
    
    x = np.arange(len(metrics))
    width = 0.35
    axes[1].bar(x - width/2, traditional, width, label='Traditional', color='red', alpha=0.7)
    axes[1].bar(x + width/2, bmd, width, label='BMD Network', color='green', alpha=0.7)
    axes[1].set_yscale('log')
    axes[1].set_ylabel('Performance (Relative)')
    axes[1].set_title('B. Performance Comparison', fontweight='bold', fontsize=14)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(metrics)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Plot 4C: Complexity Handling
    complexity_levels = np.linspace(1, 10, 50)
    traditional_perf = 90 * np.exp(-complexity_levels/3)
    bmd_perf = 99 * (1 - 0.1 * np.exp(-complexity_levels))
    
    axes[2].plot(complexity_levels, traditional_perf, 'r-', linewidth=3, label='Traditional')
    axes[2].plot(complexity_levels, bmd_perf, 'g-', linewidth=3, label='BMD Network')
    axes[2].set_xlabel('Problem Complexity')
    axes[2].set_ylabel('Performance (%)')
    axes[2].set_title('C. Complexity Handling', fontweight='bold', fontsize=14)
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    # Plot 4D: O(1) Complexity Achievement
    dataset_sizes = [100, 1000, 10000, 100000, 1000000]
    traditional_time = [x**2 for x in dataset_sizes]
    bmd_time = [100] * len(dataset_sizes)
    
    axes[3].loglog(dataset_sizes, traditional_time, 'r-', linewidth=3, 
                  marker='o', label='Traditional O(N²)')
    axes[3].loglog(dataset_sizes, bmd_time, 'g-', linewidth=3, 
                  marker='s', label='BMD O(1)')
    axes[3].set_xlabel('Dataset Size')
    axes[3].set_ylabel('Processing Time')
    axes[3].set_title('D. O(1) Complexity Achievement', fontweight='bold', fontsize=14)
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

# SIMPLE EXECUTION FUNCTION
def generate_all_panels():
    """Generate all four main panels as separate image files"""
    
    print("Generating Panel 1: Oscillatory Reality Foundations...")
    fig1 = plot_oscillatory_foundations()
    fig1.savefig('panel1_oscillatory_foundations.png', dpi=300, bbox_inches='tight')
    plt.close(fig1)
    
    print("Generating Panel 2: S-Entropy Coordinate Navigation...")
    fig2 = plot_sentropy_navigation()
    fig2.savefig('panel2_sentropy_navigation.png', dpi=300, bbox_inches='tight')
    plt.close(fig2)
    
    print("Generating Panel 3: Validation Results...")
    fig3 = plot_validation_results()
    fig3.savefig('panel3_validation_results.png', dpi=300, bbox_inches='tight')
    plt.close(fig3)
    
    print("Generating Panel 4: Biological Maxwell Demons...")
    fig4 = plot_maxwell_demons()
    fig4.savefig('panel4_maxwell_demons.png', dpi=300, bbox_inches='tight')
    plt.close(fig4)
    
    print("\nAll panels generated successfully!")
    print("Files created:")
    print("- panel1_oscillatory_foundations.png")
    print("- panel2_sentropy_navigation.png") 
    print("- panel3_validation_results.png")
    print("- panel4_maxwell_demons.png")

# CLEAR INSTRUCTIONS
def print_instructions():
    """Print clear instructions for using the visualization code"""
    print("""
=== LAVOISIER FRAMEWORK VISUALIZATION INSTRUCTIONS ===

QUICK START:
1. Run: generate_all_panels()
2. This creates 4 panel images showing your complete framework

PANEL DESCRIPTIONS:

Panel 1 - Oscillatory Reality Foundations:
A. Reality split (95% continuous vs 5% discrete)
B. Self-sustaining loop connecting math, physics, observation, consciousness  
C. Mathematical necessity chain proving oscillatory existence
D. Continuous vs discrete signal comparison

Panel 2 - S-Entropy Coordinate Navigation:
A. Complexity comparison (O(N²) vs O(1))
B. Navigation paths (random walk vs direct access)
C. Molecular information coverage by class
D. Coordinate transformation visualization

Panel 3 - Validation Results:
A. Accuracy comparison across methods and datasets
B. Processing speed improvements (1000x faster)
C. Enhancement effects of S-Stellas integration
D. Cross-dataset validation performance

Panel 4 - Biological Maxwell Demons:
A. Network architecture (attention→memory→recognition)
B. Performance metrics comparison
C. Complexity handling capabilities
D. O(1) complexity achievement demonstration

CUSTOMIZATION:
- Modify colors by changing the color arrays in each function
- Adjust figure sizes by changing figsize parameters
- Add/remove subplots by modifying the create_panel_figure calls
- Change data values to match your exact experimental results

USAGE:
generate_all_panels()  # Creates all 4 panels
print_instructions()   # Shows this help text
    """)

if __name__ == "__main__":
    print_instructions()
    generate_all_panels()
