"""
Reusable UI components for the CLI interface
"""
from typing import List, Dict, Any, Optional, Union, Tuple
import time
from datetime import timedelta

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.markdown import Markdown
from rich.syntax import Syntax
from rich.tree import Tree
from rich import box

from .styles import STYLES, COLORS

console = Console(theme=STYLES)

class HelpMessages:
    """Helper class for displaying consistent help messages"""
    
    @staticmethod
    def command_help(command: str, description: str, examples: List[Dict[str, str]]) -> None:
        """Display help for a specific command with examples"""
        console.print(f"\n[title]{command}[/title]")
        console.print(f"\n{description}\n")
        
        if examples:
            console.print("[header]Examples:[/header]")
            for example in examples:
                console.print(f"  [muted]# {example['description']}[/muted]")
                console.print(f"  [command]{example['command']}[/command]\n")
    
    @staticmethod
    def parameter_help(parameters: List[Dict[str, str]]) -> None:
        """Display help for command parameters"""
        table = Table(box=box.SIMPLE)
        table.add_column("Parameter", style=STYLES["parameter"])
        table.add_column("Description")
        table.add_column("Default", style=STYLES["muted"])
        
        for param in parameters:
            default = param.get("default", "")
            if default:
                default = f"[value]{default}[/value]"
            table.add_row(param["name"], param["description"], default)
        
        console.print(table)
    
    @staticmethod
    def error_with_suggestion(message: str, suggestion: str) -> None:
        """Display an error message with an actionable suggestion"""
        error_panel = Panel(
            Text.from_markup(f"{message}\n\n[info]Suggestion:[/info] {suggestion}"),
            title="Error",
            border_style=STYLES["error"],
            expand=False
        )
        console.print(error_panel)


class ProgressReporter:
    """Progress reporting with estimated time remaining"""
    
    def __init__(self, total: int, description: str = "Processing"):
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.text]{task.description}"),
            BarColumn(complete_style=STYLES["progress.bar"]),
            TextColumn("[progress.text]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console
        )
        self.task_id = self.progress.add_task(description, total=total)
        self.start_time = time.time()
        self.total = total
    
    def start(self):
        """Start the progress reporter"""
        self.progress.start()
        return self
    
    def update(self, advance: int = 1, description: Optional[str] = None):
        """Update the progress"""
        update_args = {"advance": advance}
        if description:
            update_args["description"] = description
        self.progress.update(self.task_id, **update_args)
    
    def get_eta(self) -> str:
        """Get the estimated time remaining as a string"""
        completed = self.progress.tasks[0].completed
        if completed == 0:
            return "Calculating..."
        
        elapsed = time.time() - self.start_time
        remaining = (elapsed / completed) * (self.total - completed)
        return str(timedelta(seconds=int(remaining)))
    
    def stop(self):
        """Stop the progress reporter"""
        self.progress.stop()


class ResultVisualizer:
    """Visualize analysis results in the terminal"""
    
    @staticmethod
    def show_spectrum_text(spectrum_data: Dict[str, Any], title: str = "MS Spectrum"):
        """Display a simplified spectrum representation in text"""
        if not spectrum_data or "mz" not in spectrum_data or "intensity" not in spectrum_data:
            console.print("[error]Invalid spectrum data provided[/error]")
            return
        
        # Extract data
        mz_values = spectrum_data["mz"]
        intensities = spectrum_data["intensity"]
        
        # Create a simple text-based visualization
        max_intensity = max(intensities)
        normalized = [int((i / max_intensity) * 20) for i in intensities]
        
        table = Table(title=title, box=box.SIMPLE)
        table.add_column("m/z", style=STYLES["info"])
        table.add_column("Intensity", style=STYLES["value"])
        table.add_column("Visualization", style=STYLES["highlight"])
        
        # Show top 10 peaks for simplicity
        top_indices = sorted(range(len(intensities)), key=lambda i: intensities[i], reverse=True)[:10]
        
        for idx in top_indices:
            mz = f"{mz_values[idx]:.4f}"
            intensity = f"{intensities[idx]:.0f}"
            bar = "█" * normalized[idx]
            table.add_row(mz, intensity, bar)
        
        console.print(table)
    
    @staticmethod
    def show_comparative_results(results: List[Dict[str, Any]], title: str = "Comparative Analysis"):
        """Display comparative results for different analysis methods"""
        if not results:
            console.print("[error]No results provided for comparison[/error]")
            return
        
        panel = Panel(
            title=title,
            border_style=STYLES["info"],
            expand=False
        )
        
        table = Table(box=box.SIMPLE)
        table.add_column("Method", style=STYLES["header"])
        
        # Determine common metrics across all results
        metrics = set()
        for result in results:
            metrics.update(result.keys())
        metrics.discard("method")  # Remove the method key
        
        for metric in sorted(metrics):
            table.add_column(metric.capitalize(), style=STYLES["value"])
        
        for result in results:
            row = [result.get("method", "Unknown")]
            for metric in sorted(metrics):
                if metric in result and metric != "method":
                    row.append(str(result[metric]))
                else:
                    row.append("-")
            table.add_row(*row)
        
        panel.renderable = table
        console.print(panel)


class WizardInterface:
    """Interactive wizard for guiding users through common workflows"""
    
    def __init__(self, title: str):
        self.title = title
        self.steps = []
        self.current_step = 0
        self.results = {}
    
    def add_step(self, title: str, prompt: str, options: Optional[List[str]] = None,
                validator: Optional[callable] = None, default: Any = None):
        """Add a step to the wizard"""
        self.steps.append({
            "title": title,
            "prompt": prompt,
            "options": options,
            "validator": validator,
            "default": default
        })
        return self
    
    def run(self) -> Dict[str, Any]:
        """Run the wizard and return the collected results"""
        console.print(f"\n[title]{self.title}[/title]\n")
        
        for i, step in enumerate(self.steps):
            self.current_step = i
            
            # Display step header
            console.print(f"[header]Step {i+1}/{len(self.steps)}: {step['title']}[/header]")
            console.print(f"\n{step['prompt']}\n")
            
            # Handle options if provided
            if step.get("options"):
                for j, option in enumerate(step["options"]):
                    console.print(f"  [value]{j+1}[/value]. {option}")
                
                while True:
                    try:
                        choice = console.input(f"\n[info]Enter choice (1-{len(step['options'])})[/info]: ")
                        idx = int(choice) - 1
                        if 0 <= idx < len(step["options"]):
                            self.results[step["title"]] = step["options"][idx]
                            break
                        else:
                            console.print(f"[error]Please enter a number between 1 and {len(step['options'])}[/error]")
                    except ValueError:
                        console.print("[error]Please enter a valid number[/error]")
            
            # Handle free text input
            else:
                default_text = f" [muted](default: {step['default']})[/muted]" if step.get("default") else ""
                while True:
                    value = console.input(f"[info]Enter value{default_text}:[/info] ")
                    
                    # Use default if empty
                    if not value and step.get("default") is not None:
                        value = step["default"]
                    
                    # Validate if needed
                    if step.get("validator") and not step["validator"](value):
                        console.print("[error]Invalid input. Please try again.[/error]")
                        continue
                    
                    self.results[step["title"]] = value
                    break
            
            console.print()  # Add spacing between steps
        
        console.print("[success]Wizard completed successfully![/success]\n")
        return self.results


def display_actionable_error(error_message: str, suggestions: List[str]) -> None:
    """Display an error message with actionable suggestions"""
    panel = Panel(
        Text.from_markup(f"{error_message}\n\n[header]Suggestions:[/header]\n" + 
                        "\n".join(f"• {suggestion}" for suggestion in suggestions)),
        title="Error",
        border_style=STYLES["error"],
        expand=False
    )
    console.print(panel)
