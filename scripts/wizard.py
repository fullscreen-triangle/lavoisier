"""
Workflow wizard for guiding users through common analysis procedures
"""
from typing import Dict, List, Any, Optional, Callable
import os
from pathlib import Path
import logging

from .components import WizardInterface, console, display_actionable_error
from .styles import STYLES

logger = logging.getLogger(__name__)

class AnalysisWizard:
    """Wizard for guiding users through common analysis workflows"""
    
    def __init__(self, config_dir: Optional[Path] = None):
        """
        Initialize the analysis wizard
        
        Args:
            config_dir: Directory to store configuration profiles
        """
        self.config_dir = config_dir or Path.home() / ".lavoisier" / "configs"
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Dictionary of available workflows
        self.workflows = {
            "basic_analysis": self._basic_analysis_workflow,
            "batch_processing": self._batch_processing_workflow,
            "comparative_analysis": self._comparative_analysis_workflow,
            "custom_visualization": self._custom_visualization_workflow
        }
    
    def list_workflows(self) -> List[Dict[str, str]]:
        """
        List available workflows with descriptions
        
        Returns:
            List of workflow info dictionaries
        """
        return [
            {
                "id": "basic_analysis",
                "name": "Basic MS Analysis",
                "description": "Standard processing of a single MS file with default parameters"
            },
            {
                "id": "batch_processing",
                "name": "Batch Processing",
                "description": "Process multiple MS files with the same parameters"
            },
            {
                "id": "comparative_analysis",
                "name": "Comparative Analysis",
                "description": "Compare results from multiple analysis methods"
            },
            {
                "id": "custom_visualization",
                "name": "Custom Visualization",
                "description": "Create customized visualizations of analysis results"
            }
        ]
    
    def run_workflow(self, workflow_id: str) -> Dict[str, Any]:
        """
        Run a specific workflow
        
        Args:
            workflow_id: ID of the workflow to run
            
        Returns:
            Collected configuration parameters
        """
        if workflow_id not in self.workflows:
            available_workflows = ", ".join(self.workflows.keys())
            display_actionable_error(
                f"Unknown workflow: {workflow_id}",
                [f"Choose one of the available workflows: {available_workflows}",
                 "Use 'lavoisier wizard list' to see all available workflows"]
            )
            return {}
        
        return self.workflows[workflow_id]()
    
    def load_profile(self, profile_name: str) -> Dict[str, Any]:
        """
        Load a saved configuration profile
        
        Args:
            profile_name: Name of the profile to load
            
        Returns:
            Configuration parameters from the profile
        """
        profile_path = self.config_dir / f"{profile_name}.json"
        if not profile_path.exists():
            logger.error(f"Profile does not exist: {profile_name}")
            return {}
        
        try:
            import json
            with open(profile_path, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading profile {profile_name}: {str(e)}")
            return {}
    
    def save_profile(self, profile_name: str, config: Dict[str, Any]) -> bool:
        """
        Save configuration as a profile
        
        Args:
            profile_name: Name for the profile
            config: Configuration parameters to save
            
        Returns:
            True if successful, False otherwise
        """
        profile_path = self.config_dir / f"{profile_name}.json"
        
        try:
            import json
            with open(profile_path, "w") as f:
                json.dump(config, f, indent=2)
            logger.info(f"Saved profile: {profile_name}")
            return True
        except Exception as e:
            logger.error(f"Error saving profile {profile_name}: {str(e)}")
            return False
    
    def list_profiles(self) -> List[str]:
        """
        List available configuration profiles
        
        Returns:
            List of profile names
        """
        return [f.stem for f in self.config_dir.glob("*.json")]
    
    def _basic_analysis_workflow(self) -> Dict[str, Any]:
        """
        Basic MS analysis workflow
        
        Returns:
            Configuration parameters for basic analysis
        """
        wizard = WizardInterface("Basic MS Analysis Workflow")
        
        # Add wizard steps
        wizard.add_step(
            "input_file",
            "Enter the path to your MS data file (mzML format)",
            validator=lambda path: os.path.exists(path) and path.endswith((".mzML", ".mzXML", ".CDF", ".mzData"))
        )
        
        wizard.add_step(
            "output_directory",
            "Enter the output directory for results",
            default="./results",
            validator=lambda path: os.path.exists(os.path.dirname(path)) if path != "./results" else True
        )
        
        wizard.add_step(
            "analysis_type",
            "Select the type of analysis to perform",
            options=["MS1 peak detection", "Full MS1+MS2 analysis", "Targeted compound search"]
        )
        
        wizard.add_step(
            "intensity_threshold",
            "Enter the intensity threshold for peak detection",
            default="1000.0",
            validator=lambda val: val.replace(".", "", 1).isdigit()
        )
        
        wizard.add_step(
            "mass_tolerance",
            "Enter the mass tolerance in Da",
            default="0.01",
            validator=lambda val: val.replace(".", "", 1).isdigit()
        )
        
        wizard.add_step(
            "visualization",
            "Would you like to generate visualizations?",
            options=["Yes", "No"]
        )
        
        wizard.add_step(
            "save_profile",
            "Would you like to save these settings as a profile for future use?",
            options=["Yes", "No"]
        )
        
        # Run the wizard
        config = wizard.run()
        
        # Handle profile saving
        if config.get("save_profile") == "Yes":
            profile_name = console.input("[info]Enter a name for this profile:[/info] ")
            if profile_name:
                self.save_profile(profile_name, config)
                console.print(f"[success]Profile saved as: {profile_name}[/success]")
        
        return config
    
    def _batch_processing_workflow(self) -> Dict[str, Any]:
        """
        Batch processing workflow
        
        Returns:
            Configuration parameters for batch processing
        """
        wizard = WizardInterface("Batch Processing Workflow")
        
        # Add wizard steps
        wizard.add_step(
            "input_directory",
            "Enter the directory containing MS data files",
            validator=lambda path: os.path.isdir(path)
        )
        
        wizard.add_step(
            "file_pattern",
            "Enter a file pattern to match (e.g., *.mzML)",
            default="*.mzML",
            validator=lambda pattern: "*" in pattern and "." in pattern
        )
        
        wizard.add_step(
            "output_directory",
            "Enter the output directory for results",
            default="./batch_results",
            validator=lambda path: os.path.exists(os.path.dirname(path)) if path != "./batch_results" else True
        )
        
        wizard.add_step(
            "parallel_jobs",
            "Enter the number of parallel jobs to run (0 for auto)",
            default="0",
            validator=lambda val: val.isdigit()
        )
        
        wizard.add_step(
            "analysis_type",
            "Select the type of analysis to perform",
            options=["MS1 peak detection", "Full MS1+MS2 analysis", "Targeted compound search"]
        )
        
        wizard.add_step(
            "intensity_threshold",
            "Enter the intensity threshold for peak detection",
            default="1000.0",
            validator=lambda val: val.replace(".", "", 1).isdigit()
        )
        
        wizard.add_step(
            "mass_tolerance",
            "Enter the mass tolerance in Da",
            default="0.01",
            validator=lambda val: val.replace(".", "", 1).isdigit()
        )
        
        wizard.add_step(
            "generate_report",
            "Would you like to generate a summary report for all files?",
            options=["Yes", "No"]
        )
        
        wizard.add_step(
            "save_profile",
            "Would you like to save these settings as a profile for future use?",
            options=["Yes", "No"]
        )
        
        # Run the wizard
        config = wizard.run()
        
        # Handle profile saving
        if config.get("save_profile") == "Yes":
            profile_name = console.input("[info]Enter a name for this profile:[/info] ")
            if profile_name:
                self.save_profile(profile_name, config)
                console.print(f"[success]Profile saved as: {profile_name}[/success]")
        
        return config
    
    def _comparative_analysis_workflow(self) -> Dict[str, Any]:
        """
        Comparative analysis workflow
        
        Returns:
            Configuration parameters for comparative analysis
        """
        wizard = WizardInterface("Comparative Analysis Workflow")
        
        # Add wizard steps
        wizard.add_step(
            "input_file",
            "Enter the path to your MS data file",
            validator=lambda path: os.path.exists(path) and path.endswith((".mzML", ".mzXML", ".CDF", ".mzData"))
        )
        
        wizard.add_step(
            "output_directory",
            "Enter the output directory for results",
            default="./comparison_results",
            validator=lambda path: os.path.exists(os.path.dirname(path)) if path != "./comparison_results" else True
        )
        
        wizard.add_step(
            "methods",
            "Select methods to compare (comma-separated)",
            options=["1: Standard peak detection", "2: ML-enhanced detection", 
                    "3: LLM-assisted annotation", "4: Visual pipeline annotation"]
        )
        
        wizard.add_step(
            "comparison_metrics",
            "Select comparison metrics",
            options=["Peak count", "Annotation coverage", "Processing time", "All metrics"]
        )
        
        wizard.add_step(
            "intensity_threshold",
            "Enter the intensity threshold for peak detection",
            default="1000.0",
            validator=lambda val: val.replace(".", "", 1).isdigit()
        )
        
        wizard.add_step(
            "visualization_type",
            "Select visualization type for comparison",
            options=["Bar chart", "Heat map", "Combined view", "None"]
        )
        
        wizard.add_step(
            "save_profile",
            "Would you like to save these settings as a profile for future use?",
            options=["Yes", "No"]
        )
        
        # Run the wizard
        config = wizard.run()
        
        # Handle profile saving
        if config.get("save_profile") == "Yes":
            profile_name = console.input("[info]Enter a name for this profile:[/info] ")
            if profile_name:
                self.save_profile(profile_name, config)
                console.print(f"[success]Profile saved as: {profile_name}[/success]")
        
        return config
    
    def _custom_visualization_workflow(self) -> Dict[str, Any]:
        """
        Custom visualization workflow
        
        Returns:
            Configuration parameters for visualization
        """
        wizard = WizardInterface("Custom Visualization Workflow")
        
        # Add wizard steps
        wizard.add_step(
            "input_file",
            "Enter the path to your results file or raw MS data",
            validator=lambda path: os.path.exists(path)
        )
        
        wizard.add_step(
            "visualization_type",
            "Select the type of visualization to create",
            options=["MS1 spectra", "MS2 spectra", "Chromatogram", "3D visualization", 
                    "Comparative plot", "Time series"]
        )
        
        wizard.add_step(
            "output_format",
            "Select output format",
            options=["PNG", "PDF", "SVG", "Interactive HTML", "Video"]
        )
        
        wizard.add_step(
            "color_scheme",
            "Select color scheme",
            options=["Default", "Viridis", "Plasma", "Inferno", "Grayscale", "High contrast"]
        )
        
        wizard.add_step(
            "width",
            "Enter the width in pixels",
            default="1200",
            validator=lambda val: val.isdigit() and 100 <= int(val) <= 3000
        )
        
        wizard.add_step(
            "height",
            "Enter the height in pixels",
            default="800",
            validator=lambda val: val.isdigit() and 100 <= int(val) <= 3000
        )
        
        wizard.add_step(
            "advanced_options",
            "Would you like to configure advanced visualization options?",
            options=["Yes", "No"]
        )
        
        # Conditionally add advanced options
        config = wizard.run()
        
        if config.get("advanced_options") == "Yes":
            advanced_wizard = WizardInterface("Advanced Visualization Options")
            
            advanced_wizard.add_step(
                "dpi",
                "Enter the DPI for raster outputs",
                default="300",
                validator=lambda val: val.isdigit() and 72 <= int(val) <= 1200
            )
            
            advanced_wizard.add_step(
                "font_size",
                "Enter the base font size",
                default="12",
                validator=lambda val: val.isdigit() and 8 <= int(val) <= 24
            )
            
            advanced_wizard.add_step(
                "show_grid",
                "Show grid lines?",
                options=["Yes", "No"]
            )
            
            advanced_wizard.add_step(
                "interactive_features",
                "Select interactive features (for HTML output)",
                options=["Zoom and pan", "Tooltips", "Linked views", "All features", "None"]
            )
            
            # Merge the advanced config with the base config
            config.update(advanced_wizard.run())
        
        # Handle profile saving
        save_profile = console.input("[info]Would you like to save these settings as a profile? (y/n):[/info] ")
        if save_profile.lower() in ["y", "yes"]:
            profile_name = console.input("[info]Enter a name for this profile:[/info] ")
            if profile_name:
                self.save_profile(profile_name, config)
                console.print(f"[success]Profile saved as: {profile_name}[/success]")
        
        return config
