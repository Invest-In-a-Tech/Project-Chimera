#!/usr/bin/env python3
"""
Project Chimera CLI - Command Line Interface for VBP Data Operations

This module provides a simple command-line interface for common operations
in the Project Chimera research pipeline, including data extraction and
basic analysis ta    process_parser.add_argument(
        '--input', '-i',
        help='Input CSV file path (default: auto-detect VBP data in data/raw/dataframes/)'
    )

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
    from src.common.data_pipeline.run_data_pipeline import DataPipelineRunner, PipelineMode
except ImportError:
    # Set to None if import fails - will be handled gracefully in the function
    GetVbpData = None
    DataPipelineRunner = None
    PipelineMode = None

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

    Returns:
        None: Outputs are saved to disk and logged to console

    Raises:
        SystemExit: Exits with code 1 if import fails or data processing encounters errors

    Example:
        >>> download_vbp_data()  # Uses default output path
        >>> download_vbp_data('custom_data.csv')  # Uses custom output path
    """
    # Check if VBP class was imported successfully at module load time
    # GetVbpData is set to None in the try-except block at module level if import fails
    # This graceful handling allows the CLI to provide better error messages
    if GetVbpData is None:
        # Log error message to inform user about missing dependencies
        logger.error("Cannot import GetVbpData class")
        logger.error("Make sure all dependencies are installed")
        # Exit with error code 1 to indicate failure to calling process
        sys.exit(1)

    # Wrap data extraction in try-except to handle any errors gracefully
    # This prevents uncaught exceptions from crashing the CLI with ugly stack traces
    try:
        # Initialize the VBP data fetcher with default Sierra Chart bridge settings
        # This creates a connection to Sierra Chart via the DTC protocol bridge
        # Log informational message to track progress in the data extraction workflow
        logger.info("Initializing VBP data downloader...")
        vbp_fetcher = GetVbpData()

        # Fetch VBP data from Sierra Chart via the initialized bridge connection
        # This retrieves historical OHLCV data + Volume by Price profiles
        # The data is returned as a pandas DataFrame with DatetimeIndex
        logger.info("Fetching VBP data from Sierra Chart...")
        df = vbp_fetcher.get_vbp_chart_data()

        # Determine output path - use provided path or create default location
        # Check if user provided a custom output path via CLI argument
        if output_path is None:
            # No custom path provided - use standard project data directory structure
            # Create Path object for the default raw dataframes directory
            # This follows the project convention: data/raw/dataframes/
            output_dir = Path("data/raw/dataframes")

            # Create the directory structure if it doesn't already exist
            # parents=True creates intermediate directories (data/, data/raw/, etc.)
            # exist_ok=True prevents errors if directory already exists
            output_dir.mkdir(parents=True, exist_ok=True)

            # Construct the full output path using path concatenation
            # The / operator joins path components in a platform-independent way
            # Convert Path object to string for compatibility with df.to_csv()
            output_path = str(output_dir / "volume_by_price_data.csv")

        # Save the processed DataFrame to CSV with DateTime index preserved
        # index=True ensures the DatetimeIndex is saved as the first column
        # This is critical for time-series analysis and future data loading
        logger.info("Saving VBP data to: %s", output_path)
        df.to_csv(output_path, index=True)

        # Log summary statistics for verification and debugging
        # len(df) returns the number of rows in the DataFrame
        logger.info("Successfully saved %d rows of VBP data", len(df))

        # Log the date range to help users verify data completeness
        # df.index.min() and max() return the earliest and latest timestamps
        logger.info("Data date range: %s to %s", df.index.min(), df.index.max())

        # Clean up the Sierra Chart bridge connection to free resources
        # This properly closes the DTC protocol connection and cleans up threads
        vbp_fetcher.stop_bridge()
        logger.info("VBP data download completed successfully")

    # Catch specific exceptions that might occur during data processing
    # ValueError: Data validation errors, invalid data formats
    # KeyError: Missing expected columns or keys in data structures
    # IOError: File system errors during CSV writing or directory creation
    except (ValueError, KeyError, IOError) as e:
        # Log the specific error message to help users troubleshoot issues
        # Use lazy % formatting (not f-strings) for Pylint compliance
        logger.error("Error downloading VBP data: %s", e)
        # Exit with error code 1 to indicate failure to calling process
        sys.exit(1)


def process_data_pipeline(input_path: Optional[str] = None, output_path: Optional[str] = None,
                         mode: str = "auto") -> None:
    """
    Process data through the data pipeline with mode selection.

    This function uses the DataPipelineRunner to process data in different modes:
    - 'training': File-based processing for ML model training and backtesting
    - 'live': Real-time data processing from Sierra Chart (future feature)
    - 'auto': Auto-detect mode based on available data sources (default)

    Args:
        input_path: Optional path to input CSV file. If None, auto-detects VBP data location
        output_path: Optional path for processed output. If None, displays results to console
        mode: Pipeline mode - 'training', 'live', or 'auto' (default: 'auto')

    Returns:
        None: Results are either saved to file or displayed to console

    Raises:
        SystemExit: Exits with code 1 if imports fail, invalid mode, or processing errors

    Example:
        >>> process_data_pipeline()  # Auto mode with auto-detected input
        >>> process_data_pipeline('data.csv', 'output.csv', 'training')  # Training mode
    """
    # Check if required classes were imported successfully at module load time
    # Both DataPipelineRunner and PipelineMode are set to None if import fails
    # This allows graceful error handling with informative messages
    if DataPipelineRunner is None or PipelineMode is None:
        # Log detailed error message about missing imports
        logger.error("Cannot import DataPipelineRunner or PipelineMode classes")
        logger.error(
            "Make sure all dependencies are installed and the pipeline module exists"
        )
        # Exit with error code 1 to indicate failure to calling process
        sys.exit(1)

    # Wrap pipeline processing in try-except to handle errors gracefully
    # This prevents uncaught exceptions from crashing the CLI
    try:
        # Convert string mode argument to PipelineMode enum value
        # Create a mapping dictionary for easy lookup and validation
        # This translates user-friendly CLI arguments to internal enum values
        mode_mapping = {
            'training': PipelineMode.TRAINING,
            'live': PipelineMode.LIVE,
            'auto': PipelineMode.AUTO
        }

        # Validate that the provided mode is in the allowed set
        # Check if user's mode string exists as a key in our mapping
        if mode not in mode_mapping:
            # Log error with the invalid mode and list of valid options
            # This helps users correct their command quickly
            logger.error("Invalid mode '%s'. Valid modes: %s", mode, list(mode_mapping.keys()))
            # Exit with error code 1 to indicate invalid argument
            sys.exit(1)

        # Retrieve the enum value corresponding to the user's mode choice
        # This converts the string to the proper PipelineMode enum member
        pipeline_mode = mode_mapping[mode]
        # Log the selected mode for transparency and debugging
        logger.info("Selected pipeline mode: %s", mode)

        # Handle different pipeline modes with appropriate logic
        # Check if user selected live mode (real-time data from Sierra Chart)
        if pipeline_mode == PipelineMode.LIVE:
            # Live mode requires Sierra Chart integration for real-time processing
            logger.info("Live mode selected - Sierra Chart integration")

            # NOTE: Live mode is a planned feature not yet implemented
            # Warn user that this functionality is coming in a future release
            logger.warning("Live mode with Sierra Chart integration is not yet implemented")
            logger.info(
                "This feature will connect to Sierra Chart for real-time data processing"
            )
            # Exit since live mode cannot proceed without implementation
            sys.exit(1)

        # Handle training mode or auto-detect mode (both use file-based processing)
        elif pipeline_mode in [PipelineMode.TRAINING, PipelineMode.AUTO]:
            # These modes process data from CSV files on disk

            # Determine input path - use provided path or auto-detect VBP data location
            # Check if user specified a custom input file path via CLI argument
            if input_path is None:
                # No custom path provided - auto-detect VBP data files
                # Define a list of possible VBP data file locations in order of preference
                # Check these paths sequentially to find the first existing file
                possible_files = [
                    Path("data/raw/dataframes/volume_by_price_data.csv"),
                    Path("data/raw/dataframes/1.volume_by_price_15years.csv"),
                    Path("data/raw/dataframes/volume_by_price_15years.csv")
                ]

                # Initialize variable to store the detected input path
                # Will remain None if no files are found
                default_input = None

                # Iterate through possible file paths to find first existing file
                # This provides automatic fallback if primary data file is missing
                for file_path in possible_files:
                    # Check if the current file path exists on disk
                    # .exists() returns True if file is present, False otherwise
                    if file_path.exists():
                        # Found a valid data file - store it and exit loop
                        default_input = file_path
                        # Break immediately to use the first found file
                        break

                # Check if we found any valid data files
                # If default_input is still None, no files were found
                if default_input is None:
                    # No VBP data files found - guide user to download data first
                    logger.error("No VBP data files found in data/raw/dataframes/")
                    logger.info(
                        "Run 'uv run main.py download-vbp' first, or specify --input path"
                    )
                    # Exit since we cannot proceed without input data
                    sys.exit(1)

                # Convert Path object to string for compatibility with pipeline
                # The pipeline expects a string path, not a Path object
                input_path = str(default_input)

            # Verify that the determined input file actually exists on disk
            # This handles both auto-detected and user-provided paths
            # Path(input_path) converts string back to Path for .exists() check
            if not Path(input_path).exists():
                # Input file doesn't exist - inform user with specific path
                logger.error("Input file not found: %s", input_path)
                # Exit since we cannot proceed with missing input file
                sys.exit(1)

            # Configure the data pipeline with input file path
            # Create configuration dictionary with required parameters
            # The pipeline expects 'file_path' key for file-based processing
            logger.info("Initializing data pipeline for file: %s", input_path)
            config = {
                'file_path': input_path
            }

            # Initialize the pipeline runner with configuration and mode
            # This creates the pipeline instance ready for data processing
            # Pass both config dict and explicit pipeline_mode enum
            pipeline = DataPipelineRunner(config, pipeline_mode)

            # Execute the pipeline to process the input data
            # run_pipeline() performs the complete data transformation workflow
            # Returns a processed pandas DataFrame with enriched features
            processed_df = pipeline.run_pipeline()

            # Get pipeline metadata for reporting and verification
            # This retrieves information about how the pipeline processed the data
            # Returns a dictionary with mode, data source, shape, columns, etc.
            pipeline_info = pipeline.get_data_info()

            # Log comprehensive processing summary for transparency
            logger.info("Pipeline processing completed:")
            # Log the mode that was requested by the user
            logger.info("  - Mode: %s", pipeline_info['mode'])
            # Log the effective mode after auto-detection (important for auto mode)
            logger.info("  - Effective mode: %s", pipeline_info['effective_mode'])
            # Log the data source that was actually used
            logger.info("  - Data source: %s", pipeline_info['data_source'])
            # Log the shape of the processed DataFrame (rows, columns)
            logger.info("  - Shape: %s", pipeline_info['shape'])

            # Calculate and log the number of columns in the processed data
            # Use len() if columns exist, otherwise default to 0
            # This handles the case where columns might be None
            columns_count = (
                len(pipeline_info['columns']) if pipeline_info['columns'] else 0
            )
            logger.info("  - Columns: %s", columns_count)

            # Handle output based on whether user specified an output path
            # Check if user provided a custom output file path via CLI argument
            if output_path:
                # User wants to save processed data to a file

                # Extract the parent directory from the output path
                # This is needed to ensure the directory exists before writing
                output_dir = Path(output_path).parent

                # Create the output directory structure if it doesn't exist
                # parents=True creates intermediate directories
                # exist_ok=True prevents errors if directory already exists
                output_dir.mkdir(parents=True, exist_ok=True)

                # Save the processed DataFrame to CSV with index preserved
                # index=True ensures DatetimeIndex is saved for time-series analysis
                processed_df.to_csv(output_path, index=True)
                # Log confirmation message with the output file path
                logger.info("Processed data saved to: %s", output_path)
            else:
                # No output path specified - display results to console
                # This is useful for quick inspection without saving files

                # Print formatted summary header with effective mode
                print(f"\nProcessed Data Summary ({pipeline_info['effective_mode']} mode):")
                # Print DataFrame shape (rows, columns) for quick overview
                print(f"Shape: {processed_df.shape}")
                # Print list of column names to see available features
                print(f"Columns: {list(processed_df.columns)}")
                # Print date range to verify temporal coverage
                print(
                    f"Date range: {processed_df.index.min()} to {processed_df.index.max()}"
                )
                # Print first 5 rows for data inspection
                print("\nFirst 5 rows:")
                print(processed_df.head())

            # Log final success message to confirm completion
            logger.info("Data pipeline processing completed successfully")

    # Catch specific exceptions that might occur during pipeline processing
    # ValueError: Invalid data values, configuration errors
    # KeyError: Missing expected keys in config or data structures
    # IOError: File system errors during reading or writing
    # FileNotFoundError: Input file doesn't exist (additional catch)
    except (ValueError, KeyError, IOError, FileNotFoundError) as e:
        # Log the specific error message to help users troubleshoot
        # Use lazy % formatting (not f-strings) for Pylint compliance
        logger.error("Error processing data through pipeline: %s", e)
        # Exit with error code 1 to indicate failure
        sys.exit(1)


def show_project_status() -> None:
    """
    Display comprehensive project status including data files and documentation.

    This function provides a dashboard view of the current project state:
    - Available data files and their sizes
    - Documentation structure (experiments, logbook entries)
    - Suggested next steps for users

    Returns:
        None: Outputs status information directly to console

    Example:
        >>> show_project_status()
        Project Chimera - Trading Edge Discovery Research
        ==================================================
        Found 2 data files:
           - volume_by_price_data.csv (15.3 MB)
           - 1.volume_by_price_15years.csv (18.7 MB)
    """
    # Print project header with consistent formatting
    # This provides a clear visual separation and project identification
    print("Project Chimera - Trading Edge Discovery Research")
    # Print separator line using repeated '=' character for visual structure
    # 50 characters provides good balance for typical terminal widths
    print("=" * 50)

    # Check for existing data files in the standard data directory
    # Create Path object for the standard raw dataframes directory
    # This is where VBP data and other raw market data files are stored
    data_dir = Path("data/raw/dataframes")

    # Check if the data directory exists before trying to list files
    # .exists() returns True if the directory is present, False otherwise
    if data_dir.exists():
        # Directory exists - look for CSV files containing extracted market data
        # .glob("*.csv") finds all files with .csv extension in the directory
        # Returns a generator, convert to list for len() and iteration
        csv_files = list(data_dir.glob("*.csv"))

        # Check if we found any CSV files in the directory
        if csv_files:
            # Found data files - display count for quick overview
            print(f"Found {len(csv_files)} data files:")

            # Display each file with size information for quick assessment
            # Iterate through the list of CSV file paths
            for file in csv_files:
                # Get file size in bytes using .stat().st_size
                # Divide by (1024 * 1024) to convert bytes to megabytes
                # This provides human-readable file size information
                size_mb = file.stat().st_size / (1024 * 1024)

                # Print file name and size with 1 decimal precision
                # file.name extracts just the filename without directory path
                # .1f formats float with 1 decimal place
                print(f"   - {file.name} ({size_mb:.1f} MB)")
        else:
            # Directory exists but contains no CSV files
            # Inform user that no data files are present
            print("No data files found in data/raw/dataframes/")
    else:
        # Data directory doesn't exist yet
        # Guide user to run data download command to initialize data structure
        print("Data directory not found - run 'download-vbp' to extract data")

    # Check documentation structure to show research progress
    # Create Path object for the documentation directory
    # This contains research experiments, logbook entries, and project docs
    docs_dir = Path("docs")

    # Check if the documentation directory exists
    if docs_dir.exists():
        # Directory exists - count formal experiments and daily logbook entries

        # Find all markdown files in the research/experiments subdirectory
        # These represent formal experiment documentation with hypotheses and results
        # .glob("*.md") finds all markdown files in the experiments folder
        experiments = list((docs_dir / "research/experiments").glob("*.md"))

        # Find all markdown files in the logbook subdirectory
        # These represent daily research notes and informal observations
        # Logbook entries track the ongoing research process
        logbook_entries = list((docs_dir / "logbook").glob("*.md"))

        # Display documentation statistics for research tracking
        # len() counts the number of found files in each category
        print(f"Documentation: {len(experiments)} experiments, "
              f"{len(logbook_entries)} logbook entries")

    # Provide actionable next steps for users
    # This guides users on how to get started or continue their work
    print("\nNext Steps:")
    # Suggest downloading VBP data if not already done
    print("   - Run 'uv run main.py download-vbp' to extract VBP data")
    # Point users to main documentation for comprehensive information
    print("   - Check docs/README.md for full documentation")
    # Guide users to start new experiments using the provided template
    print("   - Start a new experiment using docs/research/experiments/template.md")


def main():
    """
    Main CLI entry point for Project Chimera research pipeline.

    This function sets up the command-line interface with subcommands for:
    - VBP data extraction from Sierra Chart
    - Data pipeline processing with multiple modes
    - Project status reporting
    - Help and usage information

    The CLI uses argparse with subcommands to provide a clean, modular interface.
    Each subcommand has its own parser with specific arguments and help text.

    Returns:
        None: Executes appropriate subcommand and exits

    Example:
        >>> # Called automatically when script runs:
        >>> # python main.py download-vbp
        >>> # python main.py process-data --mode training
        >>> # python main.py status
    """
    # Create the main argument parser with detailed description and examples
    # This is the root parser that handles the base command and --help
    # formatter_class=RawDescriptionHelpFormatter preserves formatting in epilog
    parser = argparse.ArgumentParser(
        description="Project Chimera - Trading Edge Discovery Research Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        # epilog provides usage examples that appear after argument descriptions
        # These examples help users understand common command patterns
        epilog="""
Examples:
  uv run main.py download-vbp                              # Download VBP data to default location
  uv run main.py download-vbp --output data.csv            # Download to custom file
  uv run main.py process-data                               # Process VBP data (auto-detect mode)
  uv run main.py process-data --mode training               # Process for ML training/backtesting
  uv run main.py process-data --mode live                   # Process real-time data (Sierra Chart)
  uv run main.py process-data --input data.csv --mode auto # Process custom file with auto-detect
  uv run main.py status                                     # Show project status
        """
    )

    # Create subparser for command routing
    # add_subparsers creates a special action object that manages subcommands
    # dest='command' stores the chosen subcommand name in args.command
    # help='Available commands' provides help text for the subcommand group
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Configure the VBP data download command with optional output path
    # add_parser creates a new subcommand parser for 'download-vbp'
    # This subcommand handles extraction of historical VBP data from Sierra Chart
    download_parser = subparsers.add_parser(
        'download-vbp',
        help='Download historical Volume by Price data'
    )

    # Add optional --output/-o argument to specify custom output file path
    # If not provided, defaults to data/raw/dataframes/1.volume_by_price_15years.csv
    download_parser.add_argument(
        '--output', '-o',
        help='Output CSV file path (default: data/raw/dataframes/1.volume_by_price_15years.csv)'
    )

    # Configure the data processing pipeline command
    # add_parser creates a new subcommand parser for 'process-data'
    # This subcommand processes data through the transformation pipeline
    process_parser = subparsers.add_parser(
        'process-data',
        help='Process data through the data pipeline'
    )

    # Add optional --input/-i argument to specify input CSV file
    # If not provided, auto-detects VBP data files in standard locations
    process_parser.add_argument(
        '--input', '-i',
        help='Input CSV file path (default: auto-detect VBP data in data/raw/dataframes/)'
    )

    # Add optional --output/-o argument to specify output file for processed data
    # If not provided, displays results to console instead of saving to file
    process_parser.add_argument(
        '--output', '-o',
        help='Output CSV file path (default: display results to console)'
    )

    # Add optional --mode/-m argument to select pipeline processing mode
    # choices restricts valid values to: training, live, auto
    # default='auto' automatically detects the best mode based on data source
    process_parser.add_argument(
        '--mode', '-m',
        choices=['training', 'live', 'auto'],
        default='auto',
        help=(
            'Pipeline mode: training (file-based ML/backtesting), '
            'live (Sierra Chart real-time), auto (auto-detect) (default: auto)'
        )
    )

    # Configure the project status command (no additional arguments needed)
    # add_parser creates a simple subcommand for displaying project status
    # This command has no arguments - it just displays the current project state
    subparsers.add_parser('status', help='Show project status and available data')

    # Parse command line arguments
    # parse_args() processes sys.argv and returns a Namespace object
    # The Namespace contains attributes for each argument (command, output, input, mode)
    args = parser.parse_args()

    # Route to appropriate function based on command
    # Check which subcommand was chosen and call the corresponding function

    # Check if user chose the 'download-vbp' subcommand
    if args.command == 'download-vbp':
        # Execute VBP data download with optional custom output path
        # args.output will be None if --output wasn't provided (uses default)
        download_vbp_data(args.output)

    # Check if user chose the 'process-data' subcommand
    elif args.command == 'process-data':
        # Execute data pipeline processing with optional input/output paths and mode
        # Pass all three optional arguments to the processing function
        # args.input, args.output can be None (triggers auto-detection or console output)
        # args.mode defaults to 'auto' if not specified
        process_data_pipeline(args.input, args.output, args.mode)

    # Check if user chose the 'status' subcommand
    elif args.command == 'status':
        # Display project status dashboard
        # No arguments needed - function displays current project state
        show_project_status()

    else:
        # No command provided - show help information
        # This happens when user runs script without any subcommand
        # e.g., just 'uv run main.py' with no arguments
        # print_help() displays the full help message with usage examples
        parser.print_help()


# Standard Python idiom to check if script is being run directly
# This prevents code from running when the module is imported elsewhere
if __name__ == "__main__":
    # Call the main CLI function to start the application
    # This initiates argument parsing and command routing
    main()
