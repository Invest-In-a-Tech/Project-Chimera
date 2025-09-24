#!/usr/bin/env python3
"""
Project Chimera CLI - Command Line Interface for VBP Data Operations

This module provides a simple command-line interface for common operations
in the Project Chimera research pipeline, including data extraction and
basic analysis tasks.

Usage:
    uv run main.py download-vbp    # Extract historical VBP data
    uv run main.py --help          # Show available commands
"""

# Standard library imports
import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

# Import VBP data class at module level to avoid pylint warnings
# This allows us to check for import success early and provide better error messages
try:
    from src.project_chimera.data_sources.get_vbp_downloader import GetVbpData
except ImportError:
    # Set to None if import fails - will be handled gracefully in the function
    GetVbpData = None

# Configure logging for the CLI with timestamp and structured format
# This provides clear visibility into CLI operations and any errors that occur
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Module-level logger for CLI operations
logger = logging.getLogger(__name__)


def download_vbp_data(output_path: Optional[str] = None) -> None:
    """
    Download historical Volume by Price (VBP) data using the GetVbpData class.
    
    This function handles the complete workflow of VBP data extraction:
    1. Initialize the VBP data fetcher with Sierra Chart bridge
    2. Fetch and process historical VBP data 
    3. Save processed data to CSV format
    4. Clean up connections and resources

    Args:
        output_path: Optional custom output path for CSV file. If None, defaults to
                    data/raw/dataframes/volume_by_price_data.csv
    """
    # Check if VBP class was imported successfully at module load time
    if GetVbpData is None:
        logger.error("Cannot import GetVbpData class")
        logger.error("Make sure all dependencies are installed")
        sys.exit(1)

    try:
        # Initialize the VBP data fetcher with default Sierra Chart bridge settings
        logger.info("Initializing VBP data downloader...")
        vbp_fetcher = GetVbpData()

        # Fetch VBP data from Sierra Chart - this includes OHLCV + Volume by Price profiles
        logger.info("Fetching VBP data from Sierra Chart...")
        df = vbp_fetcher.get_vbp_chart_data()

        # Determine output path - use provided path or create default location
        if output_path is None:
            # Create the standard data directory structure if it doesn't exist
            output_dir = Path("data/raw/dataframes")
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = str(output_dir / "volume_by_price_data.csv")

        # Save the processed DataFrame to CSV with DateTime index preserved
        logger.info("Saving VBP data to: %s", output_path)
        df.to_csv(output_path, index=True)

        # Log summary statistics for verification
        logger.info("Successfully saved %d rows of VBP data", len(df))
        logger.info("Data date range: %s to %s", df.index.min(), df.index.max())

        # Clean up the Sierra Chart bridge connection
        vbp_fetcher.stop_bridge()
        logger.info("VBP data download completed successfully")

    except (ValueError, KeyError, IOError) as e:
        # Handle specific exceptions that might occur during data processing or file I/O
        logger.error("Error downloading VBP data: %s", e)
        sys.exit(1)


def show_project_status() -> None:
    """
    Display comprehensive project status including data files and documentation.
    
    This function provides a dashboard view of the current project state:
    - Available data files and their sizes
    - Documentation structure (experiments, logbook entries)
    - Suggested next steps for users
    """
    # Print project header with consistent formatting
    print("Project Chimera - Trading Edge Discovery Research")
    print("=" * 50)

    # Check for existing data files in the standard data directory
    data_dir = Path("data/raw/dataframes")
    if data_dir.exists():
        # Look for CSV files containing extracted market data
        csv_files = list(data_dir.glob("*.csv"))
        if csv_files:
            print(f"Found {len(csv_files)} data files:")
            # Display each file with size information for quick assessment
            for file in csv_files:
                size_mb = file.stat().st_size / (1024 * 1024)
                print(f"   - {file.name} ({size_mb:.1f} MB)")
        else:
            print("No data files found in data/raw/dataframes/")
    else:
        print("Data directory not found - run 'download-vbp' to extract data")

    # Check documentation structure to show research progress
    docs_dir = Path("docs")
    if docs_dir.exists():
        # Count formal experiments and daily logbook entries
        experiments = list((docs_dir / "research/experiments").glob("*.md"))
        logbook_entries = list((docs_dir / "logbook").glob("*.md"))
        print(f"Documentation: {len(experiments)} experiments, "
              f"{len(logbook_entries)} logbook entries")

    # Provide actionable next steps for users
    print("\nNext Steps:")
    print("   - Run 'uv run main.py download-vbp' to extract VBP data")
    print("   - Check docs/README.md for full documentation")
    print("   - Start a new experiment using docs/research/experiments/template.md")


def main():
    """
    Main CLI entry point for Project Chimera research pipeline.
    
    This function sets up the command-line interface with subcommands for:
    - VBP data extraction from Sierra Chart
    - Project status reporting
    - Help and usage information
    """
    # Create the main argument parser with detailed description and examples
    parser = argparse.ArgumentParser(
        description="Project Chimera - Trading Edge Discovery Research Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run main.py download-vbp                    # Download VBP data to default location
  uv run main.py download-vbp --output data.csv  # Download to custom file
  uv run main.py status                          # Show project status
        """
    )

    # Create subparser for command routing
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Configure the VBP data download command with optional output path
    download_parser = subparsers.add_parser(
        'download-vbp',
        help='Download historical Volume by Price data'
    )
    download_parser.add_argument(
        '--output', '-o',
        help='Output CSV file path (default: data/raw/dataframes/volume_by_price_data.csv)'
    )

    # Configure the project status command (no additional arguments needed)
    subparsers.add_parser('status', help='Show project status and available data')

    # Parse command line arguments
    args = parser.parse_args()

    # Route to appropriate function based on command
    if args.command == 'download-vbp':
        # Execute VBP data download with optional custom output path
        download_vbp_data(args.output)
    elif args.command == 'status':
        # Display project status dashboard
        show_project_status()
    else:
        # No command provided - show help information
        parser.print_help()


if __name__ == "__main__":
    main()
