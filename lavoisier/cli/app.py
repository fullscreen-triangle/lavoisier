import os
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any

import typer
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
from rich.syntax import Syntax
from rich.tree import Tree
import platform

# Import GlobalConfig correctly from the core.config module
from lavoisier.core.config import GlobalConfig
from lavoisier.core.logging import setup_logging, console
from lavoisier.core.metacognition import create_orchestrator, PipelineType, AnalysisStatus

# Initialize Typer CLI app with more detailed help
app = typer.Typer(
    name="lavoisier",
    help="High-performance mass spectrometry analysis with metacognitive orchestration",
    add_completion=True,
    rich_markup_mode="rich",
    context_settings={"help_option_names": ["-h", "--help"]},
)



# Store configuration and orchestrator globally
config: Optional[GlobalConfig] = None
orchestrator = None

def _load_config(config_path: str) -> GlobalConfig:
    """Load configuration from file"""
    global config
    try:
        if config_path and os.path.exists(config_path):
            config = GlobalConfig.load(config_path)
        else:
            # Try to find config in default locations
            config_locations = [
                "./lavoisier_config.yaml",
                os.path.expanduser("~/.config/lavoisier/config.yaml"),
                "/etc/lavoisier/config.yaml",
            ]
            
            for loc in config_locations:
                if os.path.exists(loc):
                    config = GlobalConfig.load(loc)
                    break
            
            if config is None:
                # No config found, use default
                config = GlobalConfig()
                
        # Make paths absolute based on current directory
        config.update_paths(os.getcwd())
        return config
        
    except Exception as e:
        console.print(f"[bold red]Error loading configuration: {str(e)}[/bold red]")
        raise typer.Exit(code=1)


def _get_orchestrator():
    """Get or create the orchestrator instance"""
    global orchestrator, config
    if orchestrator is None:
        if config is None:
            config = _load_config(None)
        orchestrator = create_orchestrator(config)
    return orchestrator


def _print_intro():
    """Print introductory banner"""
    os_info = f"{platform.system()} {platform.release()}"
    python_info = f"Python {platform.python_version()}"
    
    console.print(Panel.fit(
        Markdown("# Lavoisier\n\nHigh-performance mass spectrometry analysis with metacognitive orchestration"),
        border_style="blue",
        padding=(1, 2),
        title=f"[bold cyan]{os_info} | {python_info}[/bold cyan]",
        subtitle="[bold green]v0.1.0[/bold green]"
    ))


def _print_examples():
    """Print usage examples"""
    console.print("\n[bold cyan]Examples:[/bold cyan]")
    
    examples = [
        ("Process a single file", "lavoisier process path/to/file.mzML -o ./results"),
        ("Process multiple files", "lavoisier process path/to/*.mzML -o ./results"),
        ("Analyze with LLM assistance", "lavoisier analyze path/to/file.mzML -o ./results"),
        ("Compare multiple files", "lavoisier compare path/to/file1.mzML path/to/file2.mzML -o ./results"),
        ("Check task status", "lavoisier task abc123"),
        ("List all tasks", "lavoisier list-tasks"),
        ("Cancel running task", "lavoisier cancel abc123"),
    ]
    
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Description", style="cyan")
    table.add_column("Command", style="green")
    
    for desc, cmd in examples:
        table.add_row(f"• {desc}:", f"$ {cmd}")
    
    console.print(table)


@app.callback()
def callback(
    ctx: typer.Context,
    config_path: Optional[str] = typer.Option(
        None, "--config", "-c", 
        help="Path to configuration file. If not provided, defaults will be used"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", 
        help="Enable verbose output (sets log level to DEBUG)"
    ),
    examples: bool = typer.Option(
        False, "--examples", 
        help="Show usage examples and exit"
    )
):
    """
    Lavoisier - High-performance mass spectrometry analysis with metacognitive orchestration
    
    This tool provides a suite of commands for processing, analyzing, and comparing mass 
    spectrometry data using advanced machine learning and LLM-assisted techniques.
    
    Use the --examples flag to see common usage patterns.
    """
    # Show examples if requested
    if examples:
        _print_intro()
        _print_examples()
        raise typer.Exit()
    
    # Skip for completion command
    if ctx.invoked_subcommand == "completion":
        return
    
    # Load configuration
    global config
    config = _load_config(config_path)
    
    # Adjust log level if verbose mode is enabled
    if verbose:
        config.logging.level = "DEBUG"
    
    # Set up logging
    setup_logging(config.logging)
    
    # Print intro only for main commands
    if ctx.invoked_subcommand in ["process", "analyze", "compare", "hybrid", "list-tasks"]:
        _print_intro()


@app.command("process")
def process_command(
    input_files: List[str] = typer.Argument(
        ..., 
        help="Input files to process (supports glob patterns)",
        show_default=False
    ),
    output_dir: str = typer.Option(
        None, "--output", "-o", 
        help="Output directory (defaults to config value)"
    ),
    pipeline: str = typer.Option(
        "numeric", "--pipeline", "-p",
        help="Pipeline to use: numeric (default) or visual"
    ),
    workers: int = typer.Option(
        None, "--workers", "-w",
        help="Number of worker processes (overrides config)"
    ),
    no_wait: bool = typer.Option(
        False, "--no-wait",
        help="Don't wait for processing to complete (run in background)"
    ),
    cache_level: Optional[str] = typer.Option(
        None, "--cache", "-C",
        help="Cache level: none, raw, processed, analyzed, or all"
    ),
    force: bool = typer.Option(
        False, "--force", "-f",
        help="Force processing even if cached results exist"
    )
):
    """
    Process MS files using the numeric or visual pipeline.
    
    This command extracts data from mass spectrometry files and performs 
    initial processing. Results will be saved to the specified output directory.
    
    [bold cyan]Examples:[/bold cyan]
    
    $ lavoisier process example.mzML -o ./results
    
    $ lavoisier process *.mzML -o ./results -p visual
    
    $ lavoisier process data/*.mzML -o ./results -w 4 --no-wait
    """
    orchestrator = _get_orchestrator()
    
    # Resolve input files (handle glob patterns)
    resolved_inputs = []
    for pattern in input_files:
        from glob import glob
        matches = glob(pattern)
        if matches:
            resolved_inputs.extend(matches)
        else:
            console.print(f"[yellow]Warning: No files match pattern '{pattern}'[/yellow]")
    
    if not resolved_inputs:
        console.print("[bold red]Error: No input files found[/bold red]")
        raise typer.Exit(code=1)
    
    console.print(f"[bold green]Found {len(resolved_inputs)} input files[/bold green]")
    
    # Determine output directory
    if output_dir is None:
        output_dir = config.paths.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Create parameters dict
    parameters = {}
    
    if workers is not None:
        parameters["n_workers"] = workers
    
    # Handle cache settings
    if cache_level is not None:
        if cache_level.lower() == "none":
            parameters["caching_enabled"] = False
        else:
            parameters["caching_enabled"] = True
            parameters["cache_level"] = cache_level.lower()
    
    if force:
        parameters["force_recompute"] = True
    
    # Determine pipeline type
    if pipeline.lower() == "numeric":
        pipeline_type = PipelineType.NUMERIC
    elif pipeline.lower() == "visual":
        pipeline_type = PipelineType.VISUAL
    else:
        console.print(f"[bold red]Error: Unknown pipeline type '{pipeline}'[/bold red]")
        console.print("[yellow]Supported types: numeric, visual[/yellow]")
        raise typer.Exit(code=1)
    
    # Create and start task
    task_id = orchestrator.create_task(
        pipeline_type=pipeline_type,
        input_files=resolved_inputs,
        output_dir=output_dir,
        parameters=parameters
    )
    
    console.print(f"[bold blue]Created task {task_id}[/bold blue]")
    
    # Start task
    success = orchestrator.start_task(task_id)
    
    if not success:
        console.print(f"[bold red]Failed to start task {task_id}[/bold red]")
        raise typer.Exit(code=1)
    
    console.print(f"[bold green]Started task {task_id}[/bold green]")
    
    # Wait for task to complete if requested
    if not no_wait:
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=console
        ) as progress:
            # Add overall task
            task = progress.add_task(f"Processing {len(resolved_inputs)} files", total=100)
            
            # Update progress until task is done
            status = orchestrator.get_task_status(task_id)
            last_progress = -1
            
            while status.status == AnalysisStatus.RUNNING:
                # Update progress
                if status.progress != last_progress:
                    progress.update(task, completed=status.progress)
                    last_progress = status.progress
                
                # Get updated status
                status = orchestrator.get_task_status(task_id)
                
                # Sleep briefly to avoid excessive polling
                import time
                time.sleep(0.5)
            
            # Final progress update
            progress.update(task, completed=100)
        
        # Show task result
        if status.status == AnalysisStatus.COMPLETED:
            console.print(f"[bold green]Task {task_id} completed successfully[/bold green]")
            console.print(f"Results saved to: {output_dir}")
            
            # Display stats if available
            if status.stats:
                table = Table(title="Processing Statistics")
                table.add_column("Metric", style="cyan")
                table.add_column("Value", style="green")
                
                for key, value in status.stats.items():
                    # Format value based on type
                    if isinstance(value, float):
                        value = f"{value:.2f}"
                    table.add_row(key, str(value))
                
                console.print(table)
        else:
            console.print(f"[bold red]Task {task_id} failed: {status.error}[/bold red]")
            raise typer.Exit(code=1)
    else:
        console.print(f"[bold yellow]Task {task_id} is running in the background[/bold yellow]")
        console.print(f"Check status with: lavoisier task {task_id}")


@app.command("analyze")
def analyze_command(
    input_files: List[str] = typer.Argument(
        ..., 
        help="Input files to analyze (supports glob patterns)",
        show_default=False
    ),
    output_dir: str = typer.Option(
        None, "--output", "-o", 
        help="Output directory (defaults to config value)"
    ),
    llm_assist: bool = typer.Option(
        True, "--llm-assist/--no-llm-assist",
        help="Use LLM to assist in analysis (default: enabled)"
    ),
    no_wait: bool = typer.Option(
        False, "--no-wait",
        help="Don't wait for processing to complete (run in background)"
    ),
    model: Optional[str] = typer.Option(
        None, "--model", "-m",
        help="LLM model to use for assistance (default: from config)"
    ),
    query: Optional[str] = typer.Option(
        None, "--query", "-q",
        help="Specific question to ask the LLM about the data"
    )
):
    """
    Analyze MS files with advanced techniques and LLM assistance.
    
    This command processes MS files and performs in-depth analysis, 
    optionally using LLMs to interpret results and generate insights.
    
    [bold cyan]Examples:[/bold cyan]
    
    $ lavoisier analyze example.mzML -o ./results
    
    $ lavoisier analyze *.mzML -o ./results --model gpt-4
    
    $ lavoisier analyze data/*.mzML -q "Identify unusual patterns in these samples"
    """
    orchestrator = _get_orchestrator()
    
    # Resolve input files (handle glob patterns)
    resolved_inputs = []
    for pattern in input_files:
        from glob import glob
        matches = glob(pattern)
        if matches:
            resolved_inputs.extend(matches)
        else:
            console.print(f"[yellow]Warning: No files match pattern '{pattern}'[/yellow]")
    
    if not resolved_inputs:
        console.print("[bold red]Error: No input files found[/bold red]")
        raise typer.Exit(code=1)
    
    console.print(f"[bold green]Found {len(resolved_inputs)} input files[/bold green]")
    
    # Determine output directory
    if output_dir is None:
        output_dir = config.paths.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Create parameters dict
    parameters = {
        "llm_assist": llm_assist
    }
    
    if model:
        parameters["llm_model"] = model
    
    if query:
        parameters["llm_query"] = query
    
    # Create and start task
    task_id = orchestrator.create_task(
        pipeline_type=PipelineType.ANALYSIS,
        input_files=resolved_inputs,
        output_dir=output_dir,
        parameters=parameters
    )
    
    console.print(f"[bold blue]Created task {task_id}[/bold blue]")
    
    # Start task
    success = orchestrator.start_task(task_id)
    
    if not success:
        console.print(f"[bold red]Failed to start task {task_id}[/bold red]")
        raise typer.Exit(code=1)
    
    console.print(f"[bold green]Started task {task_id}[/bold green]")
    
    # Wait for task to complete if requested
    if not no_wait:
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=console
        ) as progress:
            # Add overall task
            task = progress.add_task(f"Analyzing {len(resolved_inputs)} files", total=100)
            
            # Update progress until task is done
            status = orchestrator.get_task_status(task_id)
            last_progress = -1
            
            while status.status == AnalysisStatus.RUNNING:
                # Update progress
                if status.progress != last_progress:
                    progress.update(task, completed=status.progress)
                    last_progress = status.progress
                
                # Get updated status
                status = orchestrator.get_task_status(task_id)
                
                # Sleep briefly to avoid excessive polling
                import time
                time.sleep(0.5)
            
            # Final progress update
            progress.update(task, completed=100)
        
        # Show task result
        if status.status == AnalysisStatus.COMPLETED:
            console.print(f"[bold green]Task {task_id} completed successfully[/bold green]")
            console.print(f"Results saved to: {output_dir}")
            
            # Display analysis summary if LLM was used
            if llm_assist and status.summary:
                panel = Panel(
                    Markdown(status.summary),
                    title="[bold cyan]Analysis Summary[/bold cyan]",
                    border_style="cyan"
                )
                console.print(panel)
        else:
            console.print(f"[bold red]Task {task_id} failed: {status.error}[/bold red]")
            raise typer.Exit(code=1)
    else:
        console.print(f"[bold yellow]Task {task_id} is running in the background[/bold yellow]")
        console.print(f"Check status with: lavoisier task {task_id}")


@app.command("task")
def task_command(
    task_id: str = typer.Argument(..., help="ID of the task to check"),
    watch: bool = typer.Option(
        False, "--watch", "-w",
        help="Watch task progress in real-time until completion"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v",
        help="Show detailed task information"
    )
):
    """
    Check the status of a specific task.
    
    This command displays detailed information about a task, including its 
    status, progress, and results if completed.
    
    [bold cyan]Examples:[/bold cyan]
    
    $ lavoisier task abc123
    
    $ lavoisier task abc123 --watch
    
    $ lavoisier task abc123 --verbose
    """
    orchestrator = _get_orchestrator()
    
    if watch:
        # Watch task progress in real-time
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=console
        ) as progress:
            # Add task
            progress_task = progress.add_task(f"Task {task_id}", total=100)
            
            # Update progress until task is done
            status = orchestrator.get_task_status(task_id)
            last_progress = -1
            
            while status.status == AnalysisStatus.RUNNING:
                # Update progress
                if status.progress != last_progress:
                    progress_task_desc = f"Task {task_id}"
                    if status.current_step:
                        progress_task_desc += f" - {status.current_step}"
                    
                    progress.update(
                        progress_task, 
                        completed=status.progress,
                        description=progress_task_desc
                    )
                    last_progress = status.progress
                
                # Get updated status
                status = orchestrator.get_task_status(task_id)
                
                # Sleep briefly to avoid excessive polling
                import time
                time.sleep(0.5)
            
            # Final progress update
            progress.update(progress_task, completed=100)
        
        # Show final status
        _display_task_status(orchestrator, task_id, verbose)
    else:
        # Just show current status
        _display_task_status(orchestrator, task_id, verbose)


def _display_task_status(orchestrator, task_id: str, verbose: bool = False):
    """Display task status in a formatted way"""
    status = orchestrator.get_task_status(task_id)
    
    # Create rich tree for hierarchical display
    tree = Tree(f"[bold blue]Task {task_id}[/bold blue]")
    
    # Add basic status information
    status_color = {
        AnalysisStatus.PENDING: "yellow",
        AnalysisStatus.RUNNING: "cyan",
        AnalysisStatus.COMPLETED: "green",
        AnalysisStatus.FAILED: "red",
        AnalysisStatus.CANCELLED: "red"
    }.get(status.status, "white")
    
    tree.add(f"[bold {status_color}]Status: {status.status.name}[/bold {status_color}]")
    
    if status.progress >= 0:
        tree.add(f"Progress: {status.progress:.1f}%")
    
    if status.start_time:
        tree.add(f"Started: {status.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    if status.end_time:
        tree.add(f"Ended: {status.end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Calculate duration if both start and end time are available
        if status.start_time:
            duration = status.end_time - status.start_time
            tree.add(f"Duration: {duration}")
    
    # Add error information if failed
    if status.status == AnalysisStatus.FAILED and status.error:
        error_branch = tree.add("[bold red]Error[/bold red]")
        error_branch.add(status.error)
    
    # Add pipeline information
    pipeline_branch = tree.add("Pipeline")
    pipeline_branch.add(f"Type: {status.pipeline_type.name}")
    pipeline_branch.add(f"Files: {len(status.input_files)}")
    pipeline_branch.add(f"Output: {status.output_dir}")
    
    # Add parameters if verbose
    if verbose and status.parameters:
        params_branch = tree.add("Parameters")
        for key, value in status.parameters.items():
            params_branch.add(f"{key}: {value}")
    
    # Add stats if available and verbose
    if verbose and status.stats:
        stats_branch = tree.add("Statistics")
        for key, value in status.stats.items():
            # Format value based on type
            if isinstance(value, float):
                stats_branch.add(f"{key}: {value:.2f}")
            else:
                stats_branch.add(f"{key}: {value}")
    
    # Add summary if available
    if status.summary:
        summary_branch = tree.add("[bold cyan]Summary[/bold cyan]")
        for line in status.summary.split('\n'):
            summary_branch.add(line)
    
    # Print the tree
    console.print(tree)
    
    # If task failed, exit with error code
    if status.status == AnalysisStatus.FAILED:
        raise typer.Exit(code=1)


@app.command("list-tasks")
def list_tasks_command(
    limit: int = typer.Option(
        10, "--limit", "-n",
        help="Maximum number of tasks to display"
    ),
    all_tasks: bool = typer.Option(
        False, "--all", "-a",
        help="Show all tasks, not just recent ones"
    ),
    status: Optional[str] = typer.Option(
        None, "--status", "-s",
        help="Filter tasks by status (PENDING, RUNNING, COMPLETED, FAILED, CANCELLED)"
    )
):
    """
    List recent tasks and their status.
    
    This command displays a table of recent tasks, showing their status, 
    progress, and basic information.
    
    [bold cyan]Examples:[/bold cyan]
    
    $ lavoisier list-tasks
    
    $ lavoisier list-tasks --all
    
    $ lavoisier list-tasks --status RUNNING
    """
    orchestrator = _get_orchestrator()
    
    # Get tasks
    tasks = orchestrator.list_tasks()
    
    # Filter by status if requested
    if status:
        try:
            filter_status = AnalysisStatus[status.upper()]
            tasks = [t for t in tasks if t.status == filter_status]
        except KeyError:
            console.print(f"[bold red]Invalid status: {status}[/bold red]")
            console.print("[yellow]Valid statuses: PENDING, RUNNING, COMPLETED, FAILED, CANCELLED[/yellow]")
            raise typer.Exit(code=1)
    
    # Sort by start time (newest first)
    tasks.sort(key=lambda t: t.start_time if t.start_time else 0, reverse=True)
    
    # Limit number of tasks if not showing all
    if not all_tasks:
        tasks = tasks[:limit]
    
    # Create table
    table = Table(title="Tasks")
    table.add_column("ID", style="cyan")
    table.add_column("Status", style="bold")
    table.add_column("Progress")
    table.add_column("Pipeline")
    table.add_column("Files")
    table.add_column("Started")
    table.add_column("Duration")
    
    # Add rows
    for task in tasks:
        # Determine status color
        status_style = {
            AnalysisStatus.PENDING: "yellow",
            AnalysisStatus.RUNNING: "cyan",
            AnalysisStatus.COMPLETED: "green",
            AnalysisStatus.FAILED: "red",
            AnalysisStatus.CANCELLED: "red"
        }.get(task.status, "white")
        
        # Calculate duration
        duration = ""
        if task.start_time:
            end_time = task.end_time or orchestrator._get_current_time()
            duration_seconds = (end_time - task.start_time).total_seconds()
            
            # Format duration
            if duration_seconds < 60:
                duration = f"{int(duration_seconds)}s"
            elif duration_seconds < 3600:
                duration = f"{int(duration_seconds / 60)}m {int(duration_seconds % 60)}s"
            else:
                duration = f"{int(duration_seconds / 3600)}h {int((duration_seconds % 3600) / 60)}m"
        
        # Format start time
        start_time = task.start_time.strftime("%Y-%m-%d %H:%M:%S") if task.start_time else "-"
        
        # Add row
        table.add_row(
            task.task_id,
            f"[{status_style}]{task.status.name}[/{status_style}]",
            f"{task.progress:.1f}%" if task.progress >= 0 else "-",
            task.pipeline_type.name,
            str(len(task.input_files)),
            start_time,
            duration
        )
    
    if not tasks:
        console.print("[yellow]No tasks found[/yellow]")
    else:
        console.print(table)


@app.command("cancel")
def cancel_task_command(
    task_id: str = typer.Argument(..., help="ID of the task to cancel"),
    force: bool = typer.Option(
        False, "--force", "-f",
        help="Force immediate cancellation (may leave resources in inconsistent state)"
    )
):
    """
    Cancel a running task.
    
    This command cancels a running task, stopping any further processing.
    
    [bold cyan]Examples:[/bold cyan]
    
    $ lavoisier cancel abc123
    
    $ lavoisier cancel abc123 --force
    """
    orchestrator = _get_orchestrator()
    
    # Get current status
    status = orchestrator.get_task_status(task_id)
    
    if status.status != AnalysisStatus.RUNNING and status.status != AnalysisStatus.PENDING:
        console.print(f"[yellow]Task {task_id} is not running or pending (status: {status.status.name})[/yellow]")
        return
    
    # Cancel the task
    success = orchestrator.cancel_task(task_id, force=force)
    
    if success:
        console.print(f"[bold green]Task {task_id} cancelled successfully[/bold green]")
    else:
        console.print(f"[bold red]Failed to cancel task {task_id}[/bold red]")
        raise typer.Exit(code=1)


@app.command("version")
def version_command():
    """
    Display version information.
    
    Shows the current version of Lavoisier and information about the 
    system environment.
    """
    _print_intro()
    
    # Show more detailed information
    table = Table(title="Version Information")
    table.add_column("Component", style="cyan")
    table.add_column("Version", style="green")
    
    # Lavoisier version
    table.add_row("Lavoisier", "0.1.0")
    
    # Python version
    table.add_row("Python", platform.python_version())
    
    # Operating system
    table.add_row("OS", f"{platform.system()} {platform.release()}")
    
    # Platform information
    table.add_row("Platform", platform.platform())
    
    # Processor
    table.add_row("Processor", platform.processor())
    
    console.print(table)


def main():
    """Entry point for the CLI"""
    app()


if __name__ == "__main__":
    main()
