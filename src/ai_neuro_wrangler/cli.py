"""
Command-line interface for AI Neuro Wrangler.
"""

import click
from pathlib import Path
from typing import Optional

from .agents.wrangling_agent import DataWranglingAgent
from .utils.config_loader import load_config, get_default_config, save_config
from .utils.logger import setup_logger


@click.group()
@click.version_option(version="0.1.0")
def main():
    """AI-Assisted Neuroimaging Data Wrangler"""
    pass


@main.command()
@click.argument('data_path', type=click.Path(exists=True))
@click.option('--config', '-c', type=click.Path(), help='Configuration file path')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
def analyze(data_path: str, config: Optional[str], verbose: bool):
    """
    Analyze a neuroimaging dataset and get recommendations.
    
    DATA_PATH: Path to the dataset directory or file
    """
    logger = setup_logger("CLI", level=20 if verbose else 30)
    
    try:
        # Load configuration
        if config:
            cfg = load_config(config)
        else:
            cfg = get_default_config()
        
        # Create agent
        agent = DataWranglingAgent(cfg, logger)
        
        # Analyze dataset
        logger.info(f"Analyzing dataset: {data_path}")
        results = agent.analyze_dataset(data_path)
        
        # Display results
        click.echo("\n=== Dataset Analysis ===")
        click.echo(f"Path: {results['path']}")
        click.echo(f"Exists: {results['exists']}")
        click.echo(f"File count: {results['file_count']}")
        
        if results['file_types']:
            click.echo("\nFile types:")
            for ext, count in results['file_types'].items():
                click.echo(f"  {ext}: {count}")
        
        if results['recommended_pipeline']:
            click.echo("\nRecommended pipeline steps:")
            for step in results['recommended_pipeline']:
                click.echo(f"  - {step}")
        
        click.echo("\n✓ Analysis complete")
    
    except Exception as e:
        click.echo(f"✗ Error: {str(e)}", err=True)
        raise click.Abort()


@main.command()
@click.argument('input_path', type=click.Path(exists=True))
@click.argument('output_path', type=click.Path())
@click.option('--config', '-c', type=click.Path(), help='Configuration file path')
@click.option('--metadata', '-m', type=click.Path(), help='Metadata CSV/JSON file')
@click.option('--steps', '-s', multiple=True, help='Specific steps to run')
@click.option('--report', '-r', type=click.Path(), help='Output report path')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
def wrangle(
    input_path: str,
    output_path: str,
    config: Optional[str],
    metadata: Optional[str],
    steps: tuple,
    report: Optional[str],
    verbose: bool
):
    """
    Run the data wrangling pipeline on neuroimaging data.
    
    INPUT_PATH: Path to input dataset
    OUTPUT_PATH: Path for output processed data
    """
    logger = setup_logger("CLI", level=20 if verbose else 30)
    
    try:
        # Load configuration
        if config:
            cfg = load_config(config)
        else:
            cfg = get_default_config()
        
        # Create agent
        agent = DataWranglingAgent(cfg, logger)
        
        # Prepare steps
        step_list = list(steps) if steps else None
        
        # Run pipeline
        logger.info("Starting data wrangling pipeline")
        click.echo(f"Input: {input_path}")
        click.echo(f"Output: {output_path}")
        
        if step_list:
            click.echo(f"Steps: {', '.join(step_list)}")
        
        results = agent.run_pipeline(
            input_path,
            output_path,
            steps=step_list,
            metadata_path=metadata
        )
        
        # Display results
        click.echo("\n=== Pipeline Results ===")
        click.echo(f"Status: {results['status']}")
        click.echo(f"Steps executed: {len(results['steps_executed'])}")
        
        for step in results['steps_executed']:
            click.echo(f"  ✓ {step}")
        
        # Generate report if requested
        if report:
            agent.generate_report(results, report)
            click.echo(f"\nReport saved to: {report}")
        
        if results['status'] == 'success':
            click.echo("\n✓ Pipeline complete")
        else:
            click.echo(f"\n✗ Pipeline failed: {results.get('error', 'Unknown error')}", err=True)
    
    except Exception as e:
        click.echo(f"✗ Error: {str(e)}", err=True)
        raise click.Abort()


@main.command()
@click.argument('output_path', type=click.Path())
def init_config(output_path: str):
    """
    Generate a default configuration file.
    
    OUTPUT_PATH: Where to save the configuration file
    """
    try:
        config = get_default_config()
        save_config(config, output_path)
        click.echo(f"✓ Configuration file created: {output_path}")
    
    except Exception as e:
        click.echo(f"✗ Error: {str(e)}", err=True)
        raise click.Abort()


@main.command()
def version():
    """Display version information."""
    click.echo("AI Neuro Wrangler v0.1.0")
    click.echo("AI-assisted neuroimaging data preprocessing")


if __name__ == '__main__':
    main()
