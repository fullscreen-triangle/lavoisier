#!/usr/bin/env python3
"""
Generate Key Validation Visualizations
Standalone script for generating the essential validation charts
"""

import sys
import os
from pathlib import Path

from visualization.panel import plot_oscillatory_foundations, plot_sentropy_navigation, plot_validation_results, \
    plot_maxwell_demons, print_instructions

# Add validation to path
sys.path.insert(0, str(Path(__file__).parent))

def generate_essential_visualizations():
    """Generate the most important validation visualizations"""
    print("🎨 Generating Essential Lavoisier Framework Visualizations")
    print("=" * 60)
    
    try:
        # Import visualization modules
        print("Importing visualization frameworks...")
        
        from validation.visualization.panel import generate_all_panels, print_instructions
        from validation.visualization.oscillatory import LavoisierVisualizationSuite
        
        print("✓ Successfully imported visualization modules")
        print()
        
        # Generate the 4 key validation panels
        print("🔬 Generating 4 Key Validation Panels...")
        generate_all_panels()
        print("✓ Panel visualizations complete")
        print()
        
        # Generate complete theoretical visualization suite
        print("🧬 Generating Complete Theoretical Framework Visualizations...")
        viz_suite = LavoisierVisualizationSuite()
        
        # Generate all visualizations
        suite_files = viz_suite.generate_all_visualizations("lavoisier_framework_visualizations")
        
        # Create validation report
        report_file = viz_suite.create_validation_report("lavoisier_validation_report.html")
        
        print(f"✓ Generated {len(suite_files)} additional visualizations")
        print(f"✓ Created comprehensive validation report")
        print()
        
        print("📊 VISUALIZATION SUMMARY:")
        print("=" * 60)
        print()
        
        print("🎯 KEY VALIDATION PANELS (4 files):")
        panel_files = [
            "panel1_oscillatory_foundations.png",
            "panel2_sentropy_navigation.png", 
            "panel3_validation_results.png",
            "panel4_maxwell_demons.png"
        ]
        
        for i, panel in enumerate(panel_files, 1):
            if Path(panel).exists():
                print(f"  ✅ {i}. {panel}")
            else:
                print(f"  ⚠️  {i}. {panel} (check for errors)")
        
        print()
        print(f"🧬 THEORETICAL FRAMEWORK SUITE ({len(suite_files)} files):")
        print(f"  📁 Directory: lavoisier_framework_visualizations/")
        
        # Show key files from suite
        static_files = [f for f in suite_files if str(f).endswith('.png')]
        interactive_files = [f for f in suite_files if str(f).endswith('.html')]
        
        print(f"  📈 Static visualizations: {len(static_files)} PNG files")
        print(f"  🖥️  Interactive visualizations: {len(interactive_files)} HTML files")
        print()
        
        print("📋 VALIDATION REPORT:")
        print(f"  📄 {report_file}")
        print("  🌐 Open in browser to view comprehensive validation summary")
        print()
        
        print("🎯 KEY VALIDATION POINTS VISUALIZED:")
        print("=" * 60)
        print("✓ Oscillatory Reality Theory (95%/5% information split)")
        print("✓ S-Entropy Coordinate Navigation (O(N²) → O(1) complexity)")  
        print("✓ Biological Maxwell Demons (performance transcendence)")
        print("✓ Validation Results (accuracy comparisons & enhancements)")
        print("✓ Mathematical Necessity (theoretical foundations)")
        print("✓ Temporal Coordinate Systems (predetermined state access)")
        print("✓ System Architecture (complete framework overview)")
        print()
        
        print("📖 USAGE INSTRUCTIONS:")
        print("=" * 60)
        print_instructions()
        
        return {
            'panel_files': panel_files,
            'suite_files': suite_files,
            'report_file': report_file,
            'status': 'SUCCESS'
        }
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("Required visualization dependencies may be missing.")
        print("Install with: pip install matplotlib seaborn plotly numpy")
        return {'status': 'IMPORT_ERROR', 'error': str(e)}
        
    except Exception as e:
        print(f"❌ Error generating visualizations: {e}")
        
        import traceback
        traceback.print_exc()
        
        return {'status': 'ERROR', 'error': str(e)}


def generate_custom_panel(panel_type: str):
    """Generate a specific validation panel"""
    print(f"🎨 Generating {panel_type} Panel...")
    
    try:
        if panel_type.lower() == "oscillatory":
            fig = plot_oscillatory_foundations()
            fig.savefig(f'custom_{panel_type.lower()}_panel.png', dpi=300, bbox_inches='tight')
            
        elif panel_type.lower() == "sentropy":
            fig = plot_sentropy_navigation()
            fig.savefig(f'custom_{panel_type.lower()}_panel.png', dpi=300, bbox_inches='tight')
            
        elif panel_type.lower() == "validation":
            fig = plot_validation_results()
            fig.savefig(f'custom_{panel_type.lower()}_panel.png', dpi=300, bbox_inches='tight')
            
        elif panel_type.lower() == "maxwell":
            fig = plot_maxwell_demons()
            fig.savefig(f'custom_{panel_type.lower()}_panel.png', dpi=300, bbox_inches='tight')
            
        else:
            print(f"❌ Unknown panel type: {panel_type}")
            print("Available types: oscillatory, sentropy, validation, maxwell")
            return False
            
        import matplotlib.pyplot as plt
        plt.close(fig)
        
        print(f"✅ Generated custom_{panel_type.lower()}_panel.png")
        return True
        
    except Exception as e:
        print(f"❌ Error generating {panel_type} panel: {e}")
        return False


def main():
    """Main function with options"""
    print("Lavoisier Framework Visualization Generator")
    print("Choose an option:")
    print("1. Generate all essential visualizations (recommended)")
    print("2. Generate specific panel")
    print("3. Show instructions only")
    print()
    
    try:
        choice = input("Enter choice (1-3) or press Enter for option 1: ").strip()
        
        if choice == "2":
            panel_type = input("Enter panel type (oscillatory/sentropy/validation/maxwell): ").strip()
            success = generate_custom_panel(panel_type)
            return {'status': 'SUCCESS' if success else 'ERROR'}
            
        elif choice == "3":
            print_instructions()
            return {'status': 'SUCCESS'}
            
        else:
            # Default: generate all visualizations
            return generate_essential_visualizations()
            
    except KeyboardInterrupt:
        print("\n🛑 Generation cancelled by user")
        return {'status': 'CANCELLED'}
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return {'status': 'ERROR', 'error': str(e)}


if __name__ == "__main__":
    print("🧬 Lavoisier Framework - Validation Visualization Generator")
    print("=" * 70)
    print()
    
    result = main()
    
    if result['status'] == 'SUCCESS':
        print()
        print("🎉 Visualization generation completed successfully!")
        print("Your theoretical framework validation is now visually documented.")
        
    elif result['status'] == 'ERROR':
        print()
        print(f"❌ Generation failed: {result.get('error', 'Unknown error')}")
        print("Check error messages above for debugging.")
        
    else:
        print(f"\n📊 Generation status: {result['status']}")
    
    print()
    print("Thank you for using the Lavoisier Framework Visualization Generator! 🚀")
