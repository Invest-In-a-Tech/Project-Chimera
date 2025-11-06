#!/usr/bin/env python3
# pylint: disable=too-many-lines
# Justification: Extensive inline documentation explaining "why" for each operation
# makes this CLI module longer than 1000 lines, but the comments provide essential
# context for understanding the research pipeline. The alternative (splitting into
# multiple modules) would reduce code cohesion for a CLI entry point.
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

# Third-party imports
import pandas as pd

# Import VBP data class at module level to avoid pylint warnings
# This allows us to check for import success early and provide better error messages
try:
    from src.project_chimera.data_sources.get_vbp_downloader import GetVbpData
    from src.common.data_pipeline.run_data_pipeline import DataPipelineRunner, PipelineMode
except ImportError as import_error:
    # Set to None if import fails - will be handled gracefully in the function
    GetVbpData = None
    DataPipelineRunner = None
    PipelineMode = None
    # Store the import error for later debugging
    _IMPORT_ERROR = import_error
else:
    _IMPORT_ERROR = None

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
        # Log the specific import error if available for debugging
        if _IMPORT_ERROR:
            logger.error("Import error details: %s", _IMPORT_ERROR)
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
    except (ValueError, KeyError) as validation_error:
        # Log the specific error message to help users troubleshoot issues
        # Use lazy % formatting (not f-strings) for Pylint compliance
        logger.error("Data validation error: %s", validation_error)
        # Exit with error code 1 to indicate failure to calling process
        sys.exit(1)
    # IOError/OSError: File system errors during CSV writing or directory creation
    except (IOError, OSError) as file_error:
        # Log file system errors separately for clarity
        logger.error("File system error: %s", file_error)
        # Exit with error code 1 to indicate failure to calling process
        sys.exit(1)
    except Exception as unexpected_error:  # pylint: disable=broad-except
        # Catch-all for unexpected errors during bridge communication or processing
        # Broad exception is necessary here for user-friendly error handling in CLI
        logger.error("Unexpected error downloading VBP data: %s", unexpected_error)
        logger.exception("Full traceback:")
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
        # Log the specific import error if available for debugging
        if _IMPORT_ERROR:
            logger.error("Import error details: %s", _IMPORT_ERROR)
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
            # Live mode ONLY streams real-time data from Sierra Chart
            # No CSV files, no pre-loaded DataFrames - just live streaming
            logger.info("Live mode selected - Streaming real-time data from Sierra Chart")

            # Ignore input_path if provided - live mode doesn't use files
            if input_path:
                logger.warning(
                    "Input file ignored in live mode - live mode only streams from Sierra Chart"
                )

            # Import Sierra Chart manager components
            # These imports are inside the function to avoid loading Sierra Chart dependencies
            # unless live mode is explicitly selected by the user
            # pylint: disable=import-outside-toplevel
            try:
                from src.common.sierra_chart_manager import (
                    SierraChartSubscriptionManager,
                    ResponseProcessor
                )
                from src.common.sierra_chart_manager.subscription_manager import SubscriptionType
            except ImportError as e:
                logger.error("Cannot import Sierra Chart components: %s", e)
                logger.error("Make sure sierra_chart_manager package is available")
                sys.exit(1)

            # Connect to Sierra Chart for real-time streaming
            logger.info("Connecting to Sierra Chart for real-time data streaming...")

            # Initialize Sierra Chart components
            sc_manager = SierraChartSubscriptionManager()
            response_processor = ResponseProcessor()

            # Initialize sequential data processor for handling multiple rows per timestamp
            # This processor is critical for VBP data because each timestamp has multiple rows
            # (one per price level), and we need granular processing with de-duplication
            # Import here to avoid loading unless live mode is selected
            # pylint: disable=import-outside-toplevel
            from src.common.sequential_data_processor.process_multiple_rows_per_timestamp import (
                ProcessMultipleRowsPerTimestamp
            )

            # Initialize storage list for processed granular data from sequential processor
            # This list accumulates processed rows for each update cycle
            # Each row contains: {'current': {...}, 't-1': {...}, 't-2': {...}, etc.}
            # The 'current' key holds the latest data, 't-1' holds previous bar data, etc.
            # This structure provides historical context for each granular price level
            processed_rows_list = []

            # Define callback function to capture processed data from sequential processor
            # The sequential processor uses callback pattern (not return values) to deliver results
            # This callback is invoked for each successfully processed row with de-duplication
            # It captures granular, enriched data with historical lookback context
            def capture_processed_row(processed_data):
                """
                Capture each processed row from the sequential processor.

                This callback receives enriched market data dictionaries containing both
                current bar data and historical lookback periods (t-1, t-2, etc.).
                Each row represents a single price level in the VBP distribution.

                Args:
                    processed_data: Dictionary with structure:
                        {
                            'current': {'Open': ..., 'Close': ..., 'Price': ..., 'BidVol': ...},
                            't-1': {previous bar data},
                            't-2': {two bars ago data},
                            ...
                        }
                        Returns None if processing failed for that row.

                Side Effects:
                    Appends valid processed_data to processed_rows_list for downstream use.
                """
                # Check if processing returned valid data (not None)
                # process_rows() may return None if row validation fails or errors occur
                if processed_data is not None:
                    # Append the enriched data dictionary to our accumulator list
                    # This data includes current values plus historical lookback context
                    # Used later for display, analysis, and trading logic
                    processed_rows_list.append(processed_data)

            # Create sequential processor instance with callback configured
            # Pass the capture function as data_callback parameter to enable data flow
            # The processor will invoke this callback for each successfully processed row
            # This establishes the data pipeline: processor -> callback -> processed_rows_list
            sequential_processor = ProcessMultipleRowsPerTimestamp(
                data_callback=capture_processed_row
            )

            try:
                # Subscribe to VBP chart data
                vbp_config = {
                    'historical_init_bars': 50,     # Fetch 50 bars of history initially
                    'realtime_update_bars': 1,      # Get 1 bar per update (latest)
                    'on_bar_close': True            # Update only on bar close (stable)
                }

                # Subscribe to VBP chart data stream from Sierra Chart
                # Why: Establishes persistent connection for real-time updates
                # The subscription tells Sierra Chart to send us updates whenever new bars close
                # This is more efficient than polling - Sierra Chart pushes data to us automatically
                # The vbp_config specifies: initial history bars, update frequency, and timing
                logger.info("Subscribing to VBP chart data...")
                vbp_subscription_id = sc_manager.subscribe_vbp_chart_data(vbp_config)
                logger.info("VBP subscription created (ID: %s)", vbp_subscription_id)

                # Get initial historical data to establish context and baseline
                # Why: We need historical bars BEFORE we can process live updates meaningfully
                # Without history, we can't calculate indicators that need lookback (RVOL, moving averages, etc.)
                # The initial_response contains the requested historical_init_bars (e.g., 50 bars)
                # This provides the necessary context for our sequential processor and feature engineering
                logger.info("Fetching initial historical data from Sierra Chart...")
                initial_response = sc_manager.get_next_response(SubscriptionType.VBP_CHART_DATA)

                # Process the raw DTC protocol response into a structured pandas DataFrame
                # Why: Raw Sierra Chart responses are in binary DTC format, unusable for analysis
                # The response_processor converts this to a clean DataFrame with proper columns and index
                # Result: DataFrame with DatetimeIndex and VBP structure (multiple rows per timestamp)
                initial_data = response_processor.process_vbp_response(initial_response)

                # Display initial data summary for immediate user feedback and verification
                # Why: Users need to verify the subscription worked and data looks correct
                # Shows data shape (rows, columns) and date range to confirm historical depth
                # This catches configuration issues early (wrong symbol, missing data, etc.)
                # First 10 rows reveal VBP structure: OHLCV bar data + multiple price level rows
                logger.info("Initial data received - Shape: %s", initial_data.shape)
                print("\nInitial Sierra Chart Data (Live Mode):")
                print(f"Shape: {initial_data.shape}")
                print(f"Date range: {initial_data.index.min()} to {initial_data.index.max()}")
                print("\nFirst 10 rows:")
                print(initial_data.head(10))

                # Run entire history through pipeline for feature engineering
                # Why: Live mode needs the SAME features as training mode for model consistency
                # The pipeline adds derived features: RVOL, technical indicators, time-based features
                # We process ALL historical bars (not just the latest) to build proper indicator state
                # Example: A 20-period moving average needs 20 bars of history to be accurate
                # Result: features DataFrame with same columns as training data, ready for model input
                pipeline_config = {'df': initial_data}
                pipeline = DataPipelineRunner(pipeline_config, pipeline_mode)
                features = pipeline.run_pipeline()
                logger.info("Initial features engineered: %s", features.shape)

                # Process entire history through sequential processor to build lookback context
                # CRITICAL: Must process ALL historical bars, not just the latest one
                # The sequential processor needs the complete history to build accurate t-1, t-2, etc.
                # Without processing all bars, the latest bar wouldn't have proper historical context
                # Example: If we only processed the 14:00 bar, it wouldn't know what t-1 (13:45) was
                processed_rows_list.clear()
                sequential_processor.process_multiple_rows(features)

                # Filter to display only the most recent bar to the user
                # We processed all history above (for context), but only show the latest completed bar
                # This gives immediate visibility: at 14:15, user sees the completed 14:00 bar
                # Then at 14:30, they'll get the update showing the 14:15 bar
                # Check if we have any processed data before filtering
                if processed_rows_list:
                    # Extract timestamp from the last processed row (most recent bar)
                    # processed_rows_list is ordered chronologically, so [-1] is the latest
                    latest_timestamp = processed_rows_list[-1]['current']['timestamp']

                    # Filter to keep only rows matching the latest timestamp
                    # Remember: VBP has multiple rows per timestamp (one per price level)
                    # So we keep all price levels for the latest bar, discard older bars
                    # List comprehension iterates through all processed rows and keeps matches
                    latest_bar_rows = [row for row in processed_rows_list
                                      if row['current']['timestamp'] == latest_timestamp]

                    # Replace the full historical list with just the latest bar's rows
                    # This is what we'll display to the user - only the most recent completed bar
                    # The historical context is already embedded in each row's 't-1', 't-2' keys
                    processed_rows_list = latest_bar_rows

                if processed_rows_list:
                    # Extract timestamp from the first processed row
                    first_row = processed_rows_list[0]
                    latest_timestamp = first_row['current']['timestamp']

                    # Display formatted header for initial data
                    print(f"\n{'='*80}")
                    print(f"LATEST BAR (Initial) - {latest_timestamp}")
                    print(f"{'='*80}")
                    print(f"Processed {len(processed_rows_list)} granular rows (price levels)")

                    # Display bar-level OHLCV data
                    print("\nBar Data (OHLCV):")
                    print(f"  Open:   {first_row['current']['Open']:.2f}")
                    print(f"  High:   {first_row['current']['High']:.2f}")
                    print(f"  Low:    {first_row['current']['Low']:.2f}")
                    print(f"  Close:  {first_row['current']['Close']:.2f}")
                    print(f"  Volume: {first_row['current']['Volume']:.0f}")
                    print(f"  RVOL:   {first_row['current']['RVOL']:.2f}")
                    print(f"  Delta:  {first_row['current']['Delta']:.0f}")
                    print(f"  CumDelta: {first_row['current']['CumulativeDelta']:.0f}")
                    print(f"  LargeBidTrade: {first_row['current']['LargeBidTrade']:.0f}")
                    print(f"  LargeAskTrade: {first_row['current']['LargeAskTrade']:.0f}")

                    # Display Volume by Price distribution
                    print(f"\nVolume by Price Distribution ({len(processed_rows_list)} price levels):")
                    print(f"{'Price':>10} {'BidVol':>12} {'AskVol':>12} {'TotalVol':>12} {'Trades':>10}")
                    print("-" * 60)

                    for processed_row in processed_rows_list:
                        current_data = processed_row['current']
                        price = current_data.get('Price', 0.0)
                        bid_vol = current_data.get('BidVol', 0)
                        ask_vol = current_data.get('AskVol', 0)
                        total_vol = current_data.get('TotalVolume', 0)
                        trades = current_data.get('NumOfTrades', 0)
                        print(f"{price:>10.2f} {bid_vol:>12.0f} {ask_vol:>12.0f} "
                              f"{total_vol:>12.0f} {trades:>10.0f}")

                # Enter real-time processing loop
                logger.info("\nStarting real-time data processing loop...")
                logger.info("Press Ctrl+C to stop\n")

                update_count = 0
                while True:
                    # Get next update from Sierra Chart (blocks until new data)
                    response = sc_manager.get_next_response(SubscriptionType.VBP_CHART_DATA)
                    update_count += 1

                    # Process response to DataFrame (already has VBP structure)
                    update_df = response_processor.process_vbp_response(response)

                    # Run through pipeline for feature engineering
                    # Pipeline adds derived features and indicators to the raw VBP data
                    # The resulting DataFrame maintains multiple rows per timestamp structure
                    pipeline_config = {'df': update_df}
                    pipeline = DataPipelineRunner(pipeline_config, pipeline_mode)
                    features_df = pipeline.run_pipeline()

                    # Clear the processed rows list for this update cycle
                    # Ensures we only have fresh data for the current update
                    # Previous update's data is no longer needed since we process sequentially
                    processed_rows_list.clear()

                    # Process the update through sequential processor with granular approach
                    # This is critical for VBP data accuracy - processes each price level individually
                    # The processor handles: de-duplication, historical context, row-by-row enrichment
                    # Results are delivered via callback (not return value) to processed_rows_list
                    # Each row gets enriched with lookback data ('t-1', 't-2', etc.)
                    sequential_processor.process_multiple_rows(features_df)

                    # Process the captured rows from sequential processor
                    # Check if callback received any processed data (non-empty list)
                    # processed_rows_list is populated by capture_processed_row() callback
                    # Each element is a dict with 'current', 't-1', 't-2' keys for historical context
                    if processed_rows_list:
                        # Extract timestamp from the first processed row for display header
                        # All rows in this update share the same timestamp (multiple price levels)
                        # Access nested structure: row -> 'current' -> 'timestamp'
                        first_row = processed_rows_list[0]
                        latest_timestamp = first_row['current']['timestamp']

                        # Display formatted header with update number and timestamp
                        # Provides clear visual separation between updates in the console
                        # 80 character separator line matches standard terminal width
                        print(f"\n{'='*80}")
                        print(f"UPDATE #{update_count} - {latest_timestamp}")
                        print(f"{'='*80}")

                        # Display count of granular rows (price levels) processed in this update
                        # VBP data has multiple rows per timestamp - one for each price level
                        # This shows how many price levels were in the VBP distribution
                        print(f"Processed {len(processed_rows_list)} granular rows (price levels)")

                        # Display bar-level OHLCV data from the first processed row
                        # Bar data (Open, High, Low, Close, Volume) is identical across all price levels
                        # RVOL is a derived indicator showing relative volume compared to average
                        # Delta is volume delta showing net buying/selling pressure
                        # CumulativeDelta is the running total of net volume delta
                        # LargeBidTrade and LargeAskTrade show institutional order flow
                        # Access nested dict: first_row['current'] contains current bar's data
                        print("\nBar Data (OHLCV):")
                        print(f"  Open:   {first_row['current']['Open']:.2f}")
                        print(f"  High:   {first_row['current']['High']:.2f}")
                        print(f"  Low:    {first_row['current']['Low']:.2f}")
                        print(f"  Close:  {first_row['current']['Close']:.2f}")
                        print(f"  Volume: {first_row['current']['Volume']:.0f}")
                        print(f"  RVOL:   {first_row['current']['RVOL']:.2f}")
                        print(f"  Delta:  {first_row['current']['Delta']:.0f}")
                        print(f"  CumDelta: {first_row['current']['CumulativeDelta']:.0f}")
                        print(f"  LargeBidTrade: {first_row['current']['LargeBidTrade']:.0f}")
                        print(f"  LargeAskTrade: {first_row['current']['LargeAskTrade']:.0f}")

                        # Display Volume by Price distribution table header
                        # Each row in processed_rows_list represents a different price level
                        # This shows where volume was concentrated within the bar's price range
                        print(f"\nVolume by Price Distribution ({len(processed_rows_list)} price levels):")

                        # Print formatted table header with right-aligned column names
                        # :>10 means right-align in 10 character width
                        # Provides consistent spacing for numeric data alignment
                        print(f"{'Price':>10} {'BidVol':>12} {'AskVol':>12} {'TotalVol':>12} {'Trades':>10}")
                        print("-" * 60)

                        # Iterate through each processed price level to display VBP distribution
                        # Each processed_row contains enriched data for one price level
                        # Shows granular volume distribution across different prices in the bar
                        for processed_row in processed_rows_list:
                            # Extract current data from the processed row dictionary
                            # 'current' key contains this bar's data (vs 't-1' for previous bar)
                            current_data = processed_row['current']

                            # Extract VBP-specific fields for this price level
                            # .get() with default values handles missing keys gracefully
                            # Price: The specific price level for this VBP row
                            price = current_data.get('Price', 0.0)
                            # BidVol: Volume traded at bid (sellers) at this price
                            bid_vol = current_data.get('BidVol', 0)
                            # AskVol: Volume traded at ask (buyers) at this price
                            ask_vol = current_data.get('AskVol', 0)
                            # TotalVolume: Combined bid + ask volume at this price
                            total_vol = current_data.get('TotalVolume', 0)
                            # NumOfTrades: Number of individual trades at this price
                            trades = current_data.get('NumOfTrades', 0)

                            # Print formatted row with right-aligned numeric values
                            # .2f formats price with 2 decimals, .0f formats volumes as integers
                            # Maintains consistent column alignment for easy visual analysis
                            print(f"{price:>10.2f} {bid_vol:>12.0f} {ask_vol:>12.0f} "
                                  f"{total_vol:>12.0f} {trades:>10.0f}")

                        # Trading logic placeholder showing how to access historical context
                        # Each processed_row contains not just 'current' but also 't-1', 't-2', etc.
                        # This enables trading decisions based on current data + historical patterns
                        # Example use cases:
                        #   - Compare current RVOL to previous bar's RVOL for momentum
                        #   - Analyze price change from t-1 Close to current Close
                        #   - Detect volume spikes by comparing current vs t-1 volume
                        # Example: Access historical data for analysis
                        # Each processed_row contains 'current', 't-1', 't-2', etc.
                        # You can use this for your trading logic:
                        # if first_row['current']['RVOL'] > 2.0:
                        #     prev_close = first_row.get('t-1', {}).get('Close', 0)
                        #     current_close = first_row['current']['Close']
                        #     # Make trading decision based on current + historical context

                        # Optionally save processed data to CSV file for persistence or analysis
                        # Check if user provided --output path via CLI argument
                        if output_path:
                            # Convert processed rows list to pandas DataFrame for CSV export
                            # Extract only 'current' data from each processed row (ignore t-1, t-2, etc.)
                            # List comprehension iterates through all processed rows and extracts 'current' dict
                            # Result: list of dicts, each containing current bar data for one price level
                            current_data_list = [row['current'] for row in processed_rows_list]

                            # Create DataFrame from list of current data dictionaries
                            # Each dict becomes a row, dict keys become DataFrame columns
                            # Columns: Open, High, Low, Close, Volume, Price, BidVol, AskVol, etc.
                            processed_df = pd.DataFrame(current_data_list)

                            # Set the timestamp as the DataFrame index for time-series structure
                            # Check if 'timestamp' column exists before attempting to set as index
                            # This maintains temporal ordering and enables time-based operations
                            if 'timestamp' in processed_df.columns:
                                # Convert timestamp column to index (removes from columns)
                                # inplace=True modifies the DataFrame directly without creating a copy
                                processed_df.set_index('timestamp', inplace=True)

                            # Append this update's data to the output CSV file
                            # mode='a' opens file in append mode (adds to end without overwriting)
                            # header=not Path(output_path).exists() writes column headers only if file is new
                            # This creates a growing CSV file with all updates across the live session
                            # Each update adds multiple rows (one per price level) to the file
                            processed_df.to_csv(
                                output_path,
                                mode='a',
                                header=not Path(output_path).exists()
                            )
                            # Log file save operation at debug level (minimal console noise)
                            # Confirms data is being persisted for later analysis
                            logger.debug("Update saved to: %s", output_path)
                    else:
                        # Handle case where sequential processor returned no data
                        # This shouldn't normally happen but indicates a processing issue
                        # Could occur if all rows were duplicates or processing failed
                        # Log warning with update number to help identify problematic updates
                        logger.warning("No processed rows from sequential processor for update #%d",
                                     update_count)

                    # Your trading logic or model predictions would go here
                    # Example: if latest_row.get('RVOL', 0) > 2.0:
                    #     logger.info("High volume alert!")
                    #     # Make prediction with model
                    #     # Execute trade if conditions met

            except KeyboardInterrupt:
                # User pressed Ctrl+C to stop
                logger.info("\nStopping real-time data processing...")
                print("\nStopping live data stream...")

            finally:
                # Always clean up the subscriptions
                logger.info("Cleaning up Sierra Chart subscriptions...")
                sc_manager.stop_all_subscriptions()
                logger.info("Sierra Chart subscriptions terminated")
                print("Live mode terminated successfully")

            # Exit after live mode completes
            return

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
    except ValueError as value_error:
        # Log value/configuration errors
        logger.error("Value error in pipeline processing: %s", value_error)
        # Exit with error code 1 to indicate failure
        sys.exit(1)
    # KeyError: Missing expected keys in config or data structures
    except KeyError as key_error:
        # Log missing key errors
        logger.error("Key error in pipeline processing: %s", key_error)
        # Exit with error code 1 to indicate failure
        sys.exit(1)
    # FileNotFoundError: Input file doesn't exist (must come before IOError)
    except FileNotFoundError as file_error:
        # Log file not found errors
        logger.error("File not found error: %s", file_error)
        # Exit with error code 1 to indicate failure
        sys.exit(1)
    # IOError/OSError: File system errors during reading or writing
    except (IOError, OSError) as io_error:
        # Log file system errors
        logger.error("File system error in pipeline processing: %s", io_error)
        # Exit with error code 1 to indicate failure
        sys.exit(1)
    except Exception as unexpected_error:  # pylint: disable=broad-except
        # Catch-all for unexpected pipeline errors
        # Broad exception is necessary here for user-friendly error handling in CLI
        logger.error("Unexpected error processing data through pipeline: %s", unexpected_error)
        logger.exception("Full traceback:")
        # Exit with error code 1 to indicate failure
        sys.exit(1)


def show_project_status() -> None:
    """
    Display project status including data files and documentation.

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


def validate_data(input_path: Optional[str] = None) -> None:
    """
    Validate VBP data quality with comprehensive checks.

    This function performs data quality validation including:
    - File existence and readability checks
    - Schema validation (required columns present)
    - Missing data detection (NaN values, gaps in time series)
    - Data range validation (reasonable OHLCV values)
    - Timestamp continuity checks
    - Statistical anomaly detection

    Args:
        input_path: Optional path to CSV file. If None, auto-detects VBP data files
                   in data/raw/dataframes/ directory.

    Returns:
        None: Outputs validation report to console and exits with code 0 (pass) or 1 (fail)

    Raises:
        SystemExit: Exits with code 1 if critical validation errors are found

    Example:
        >>> validate_data()  # Auto-detect and validate
        >>> validate_data('data/custom.csv')  # Validate specific file
    """
    logger.info("Starting data validation...")

    # Determine which file to validate
    if input_path is None:
        # Auto-detect VBP data files in standard location
        data_dir = Path("data/raw/dataframes")
        if not data_dir.exists():
            logger.error("Data directory not found: %s", data_dir)
            logger.error("Run 'download-vbp' command first to extract data")
            sys.exit(1)

        # Find CSV files in the data directory
        csv_files = list(data_dir.glob("*.csv"))
        if not csv_files:
            logger.error("No CSV files found in %s", data_dir)
            sys.exit(1)

        # Use the first CSV file found (or most recent)
        input_path = str(csv_files[0])
        logger.info("Auto-detected data file: %s", input_path)
    else:
        # Use provided path
        if not Path(input_path).exists():
            logger.error("File not found: %s", input_path)
            sys.exit(1)

    try:
        # Load the data
        logger.info("Loading data from: %s", input_path)
        df = pd.read_csv(input_path, index_col=0, parse_dates=True)
        logger.info("✓ Successfully loaded %d rows", len(df))

        # Track validation issues
        issues = []
        warnings_list = []

        # Check 1: Required columns for VBP data
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        optional_columns = ['RVOL', 'Price', 'Delta', 'CumulativeDelta']

        missing_required = [col for col in required_columns if col not in df.columns]
        if missing_required:
            issues.append(f"Missing required columns: {missing_required}")
        else:
            logger.info("✓ All required columns present")

        missing_optional = [col for col in optional_columns if col not in df.columns]
        if missing_optional:
            warnings_list.append(f"Missing optional columns: {missing_optional}")

        # Check 2: Missing data (NaN values)
        nan_counts = df.isnull().sum()
        columns_with_nans = nan_counts[nan_counts > 0]
        if not columns_with_nans.empty:
            warnings_list.append("Columns with missing values:")
            for col, count in columns_with_nans.items():
                pct = (count / len(df)) * 100
                warnings_list.append(f"  - {col}: {count} ({pct:.2f}%)")
        else:
            logger.info("✓ No missing values detected")

        # Check 3: Data ranges (OHLC relationships)
        if all(col in df.columns for col in ['Open', 'High', 'Low', 'Close']):
            invalid_ranges = (
                (df['High'] < df['Low']) |
                (df['High'] < df['Open']) |
                (df['High'] < df['Close']) |
                (df['Low'] > df['Open']) |
                (df['Low'] > df['Close'])
            )
            invalid_count = invalid_ranges.sum()
            if invalid_count > 0:
                issues.append(f"Invalid OHLC relationships in {invalid_count} rows")
            else:
                logger.info("✓ All OHLC relationships valid")

        # Check 4: Negative values where they shouldn't be
        if 'Volume' in df.columns:
            negative_volume = (df['Volume'] < 0).sum()
            if negative_volume > 0:
                issues.append(f"Negative volume values in {negative_volume} rows")
            else:
                logger.info("✓ No negative volume values")

        # Check 5: Timestamp continuity (check for large gaps)
        if isinstance(df.index, pd.DatetimeIndex):
            time_diffs = df.index.to_series().diff()
            median_diff = time_diffs.median()
            # Flag gaps larger than 10x the median time difference
            large_gaps = time_diffs[time_diffs > median_diff * 10]
            if not large_gaps.empty:
                warnings_list.append(f"Found {len(large_gaps)} large time gaps")
                warnings_list.append(f"  Median interval: {median_diff}")
                warnings_list.append(f"  Largest gap: {large_gaps.max()}")
            else:
                logger.info("✓ No large time gaps detected")

        # Check 6: Statistical anomalies (extreme outliers)
        if 'Close' in df.columns:
            close_mean = df['Close'].mean()
            close_std = df['Close'].std()
            outliers = df[(df['Close'] > close_mean + 5 * close_std) |
                         (df['Close'] < close_mean - 5 * close_std)]
            if not outliers.empty:
                warnings_list.append(f"Found {len(outliers)} extreme price outliers (>5 std dev)")
            else:
                logger.info("✓ No extreme price outliers")

        # Print validation summary
        print("\n" + "=" * 70)
        print("DATA VALIDATION REPORT")
        print("=" * 70)
        print(f"File: {input_path}")
        print(f"Rows: {len(df):,}")
        print(f"Columns: {len(df.columns)}")
        print(f"Date Range: {df.index.min()} to {df.index.max()}")
        print()

        if issues:
            print("❌ CRITICAL ISSUES FOUND:")
            for issue in issues:
                print(f"   {issue}")
            print()

        if warnings_list:
            print("⚠️  WARNINGS:")
            for warning in warnings_list:
                print(f"   {warning}")
            print()

        if not issues and not warnings_list:
            print("✅ ALL CHECKS PASSED - Data quality is excellent!")
        elif not issues:
            print("✅ VALIDATION PASSED - Minor warnings noted above")
        else:
            print("❌ VALIDATION FAILED - Critical issues must be resolved")

        print("=" * 70)

        # Exit with appropriate code
        if issues:
            sys.exit(1)
        else:
            sys.exit(0)

    except Exception as error:  # pylint: disable=broad-except
        logger.error("Validation failed with error: %s", error)
        logger.exception("Full traceback:")
        sys.exit(1)


def subscribe_raw(
    bars: int = 50,
    update_interval: str = 'close'
) -> None:
    """
    Subscribe to raw VBP data from Sierra Chart for debugging and monitoring.

    This command establishes a real-time subscription to Sierra Chart and displays
    raw market data updates without processing. Useful for:
    - Debugging Sierra Chart connection issues
    - Monitoring live data feed quality
    - Verifying subscription configuration
    - Testing before running full pipeline

    Args:
        bars: Number of historical bars to fetch initially (default: 50)
        update_interval: When to receive updates - 'close' (bar close only) or
                        'tick' (every price change) (default: 'close')

    Returns:
        None: Runs continuously until Ctrl+C, displaying updates to console

    Raises:
        SystemExit: Exits with code 1 if connection fails or errors occur

    Example:
        >>> subscribe_raw()  # Default: 50 bars, update on close
        >>> subscribe_raw(bars=100, update_interval='tick')  # 100 bars, tick updates
    """
    logger.info("Starting raw VBP subscription...")
    logger.info("Configuration: %d bars, update on %s", bars, update_interval)

    try:
        # Import Sierra Chart subscription module
        # pylint: disable=import-outside-toplevel
        from src.sc_py_bridge.subscribe_to_vbp_chart_data import SubscribeToVbpChartData
    except ImportError as e:
        logger.error("Cannot import Sierra Chart subscription module: %s", e)
        logger.error("Make sure all dependencies are installed")
        sys.exit(1)

    subscriber = None  # Initialize to None for proper cleanup in exception handlers

    try:
        # Initialize subscriber
        on_bar_close = update_interval == 'close'
        logger.info("Connecting to Sierra Chart...")

        subscriber = SubscribeToVbpChartData(
            historical_init_bars=bars,
            realtime_update_bars=1,
            on_bar_close=on_bar_close
        )

        logger.info("✓ Connected successfully!")
        logger.info("Streaming live data... (Press Ctrl+C to stop)")
        print("\n" + "=" * 70)

        update_count = 0

        # Continuous update loop
        while True:
            # Get next update (blocking call)
            df = subscriber.get_subscribed_vbp_chart_data()

            update_count += 1

            # Get the latest bar data
            if not df.empty:
                latest = df.iloc[-1]
                timestamp = df.index[-1]

                print(f"\n[Update #{update_count}] {timestamp}")
                print("-" * 70)

                # Display OHLCV data
                if all(col in df.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume']):
                    print(f"  O: {latest['Open']:.2f}  H: {latest['High']:.2f}  "
                          f"L: {latest['Low']:.2f}  C: {latest['Close']:.2f}")
                    print(f"  Volume: {latest['Volume']:.0f}", end="")

                    # Show RVOL if available
                    if 'RVOL' in df.columns:
                        print(f"  RVOL: {latest['RVOL']:.2f}", end="")

                    # Show Delta indicators if available
                    if 'Delta' in df.columns:
                        print(f"  Delta: {latest['Delta']:.0f}", end="")
                    if 'CumulativeDelta' in df.columns:
                        print(f"  CumDelta: {latest['CumulativeDelta']:.0f}", end="")

                    print()  # New line

                # Show VBP data summary if available
                if 'Price' in df.columns:
                    vbp_count = df[df.index == timestamp].shape[0]
                    print(f"  VBP Levels: {vbp_count}")

                print("-" * 70)

    except KeyboardInterrupt:
        print("\n\nStopping subscription...")
        logger.info("Subscription stopped by user")
        # Clean up connection if subscriber was initialized
        if subscriber is not None:
            try:
                subscriber.stop_bridge()
                logger.info("Connection closed successfully")
            except:  # pylint: disable=bare-except
                pass
    except Exception as error:  # pylint: disable=broad-except
        logger.error("Subscription error: %s", error)
        logger.exception("Full traceback:")
        # Clean up on error if subscriber was initialized
        if subscriber is not None:
            try:
                subscriber.stop_bridge()
            except:  # pylint: disable=bare-except
                pass
        sys.exit(1)


def export_features(
    input_path: Optional[str] = None,
    output_path: Optional[str] = None,
    format_type: str = 'csv'
) -> None:
    """
    Export processed features for ML model training.

    This function processes VBP data through the feature engineering pipeline
    and exports the results in formats suitable for machine learning:
    - CSV: Standard comma-separated values (compatible with all tools)
    - Parquet: Columnar format (fast, compressed, preserves types)
    - HDF5: Hierarchical format (efficient for large datasets)

    The exported features include:
    - Processed OHLCV data
    - Volume indicators (RVOL, Delta, Cumulative Delta)
    - VBP distribution features
    - Historical lookback periods (t-1, t-2, etc.)

    Args:
        input_path: Optional path to input CSV. If None, auto-detects VBP data
        output_path: Optional output path. If None, creates output in data/processed/
        format_type: Output format - 'csv', 'parquet', or 'hdf5' (default: 'csv')

    Returns:
        None: Saves features to disk and logs summary

    Raises:
        SystemExit: Exits with code 1 if processing fails

    Example:
        >>> export_features()  # Auto-detect input, CSV output
        >>> export_features('data.csv', 'features.parquet', 'parquet')
        >>> export_features(format_type='hdf5')
    """
    logger.info("Starting feature export...")

    # Validate format type
    valid_formats = ['csv', 'parquet', 'hdf5']
    if format_type not in valid_formats:
        logger.error("Invalid format '%s'. Valid formats: %s", format_type, valid_formats)
        sys.exit(1)

    # Determine input file
    if input_path is None:
        data_dir = Path("data/raw/dataframes")
        if not data_dir.exists():
            logger.error("Data directory not found: %s", data_dir)
            logger.error("Run 'download-vbp' command first")
            sys.exit(1)

        csv_files = list(data_dir.glob("*.csv"))
        if not csv_files:
            logger.error("No CSV files found in %s", data_dir)
            sys.exit(1)

        input_path = str(csv_files[0])
        logger.info("Auto-detected input file: %s", input_path)
    else:
        if not Path(input_path).exists():
            logger.error("Input file not found: %s", input_path)
            sys.exit(1)

    # Determine output path
    if output_path is None:
        output_dir = Path("data/processed")
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        filename = f"features_{timestamp}.{format_type}"
        output_path = str(output_dir / filename)
        logger.info("Output will be saved to: %s", output_path)

    try:
        # Validate input file
        if not Path(input_path).exists():
            logger.error("Input file not found: %s", input_path)
            sys.exit(1)

        logger.info("Processing features from: %s", input_path)

        # Process through pipeline (training mode for feature engineering)
        logger.info("Processing features through data pipeline...")

        if DataPipelineRunner is None or PipelineMode is None:
            logger.error("Cannot import DataPipelineRunner")
            if _IMPORT_ERROR:
                logger.error("Import error: %s", _IMPORT_ERROR)
            sys.exit(1)

        # Initialize pipeline in training mode with config
        config = {'file_path': input_path}
        pipeline = DataPipelineRunner(config, PipelineMode.TRAINING)

        # Process the data through the pipeline
        processed_data = pipeline.run_pipeline()

        logger.info("Feature engineering complete")
        logger.info("Processed features shape: %d rows", len(processed_data))

        # Export in requested format
        logger.info("Exporting features as %s...", format_type)

        if format_type == 'csv':
            processed_data.to_csv(output_path, index=True)
        elif format_type == 'parquet':
            processed_data.to_parquet(output_path, index=True)
        elif format_type == 'hdf5':
            processed_data.to_hdf(output_path, key='features', mode='w')

        # Get file size
        file_size = Path(output_path).stat().st_size / (1024 * 1024)  # MB

        # Print summary
        print("\n" + "=" * 70)
        print("FEATURE EXPORT COMPLETE")
        print("=" * 70)
        print(f"Input file: {input_path}")
        print(f"Output file: {output_path}")
        print(f"Format: {format_type.upper()}")
        print(f"Rows: {len(processed_data):,}")
        print(f"Columns: {len(processed_data.columns)}")
        print(f"File size: {file_size:.2f} MB")
        print()
        print("Features included:")
        for col in processed_data.columns[:20]:  # Show first 20 columns
            print(f"  - {col}")
        if len(processed_data.columns) > 20:
            print(f"  ... and {len(processed_data.columns) - 20} more")
        print("=" * 70)

        logger.info("✓ Features exported successfully")

    except Exception as error:  # pylint: disable=broad-except
        logger.error("Feature export failed: %s", error)
        logger.exception("Full traceback:")
        sys.exit(1)


def main():
    """
    Main CLI entry point for Project Chimera research pipeline.

    This function sets up the command-line interface with subcommands for:
    - VBP data extraction from Sierra Chart
    - Data pipeline processing with multiple modes
    - Project status reporting
    - Data validation and quality checks
    - Raw subscription monitoring for debugging
    - Feature export for ML model training
    - Help and usage information

    The CLI uses argparse with subcommands to provide a clean, modular interface.
    Each subcommand has its own parser with specific arguments and help text.

    Returns:
        None: Executes appropriate subcommand and exits

    Example:
        >>> # Called automatically when script runs:
        >>> # python main.py download-vbp
        >>> # python main.py process-data --mode training
        >>> # python main.py validate-data
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
  uv run main.py download-vbp                                   # Download VBP data to default location
  uv run main.py download-vbp --output data.csv                 # Download to custom file
  uv run main.py process-data                                   # Process VBP data (auto-detect mode)
  uv run main.py process-data --mode training                   # Process for ML training/backtesting
  uv run main.py process-data --mode live                       # Process real-time data (Sierra Chart)
  uv run main.py process-data --input data.csv --mode auto      # Process custom file with auto-detect
  uv run main.py validate-data                                  # Validate data quality
  uv run main.py validate-data --input custom.csv               # Validate specific file
  uv run main.py subscribe-raw                                  # Monitor raw Sierra Chart data
  uv run main.py subscribe-raw --bars 100 --interval tick       # Subscribe with custom settings
  uv run main.py export-features                                # Export features as CSV
  uv run main.py export-features --format parquet               # Export features as Parquet
  uv run main.py status                                         # Show project status
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

    # Configure the data validation command
    # add_parser creates a new subcommand parser for 'validate-data'
    # This subcommand performs comprehensive data quality checks
    validate_parser = subparsers.add_parser(
        'validate-data',
        help='Validate VBP data quality and integrity'
    )

    # Add optional --input/-i argument to specify file to validate
    # If not provided, auto-detects VBP data files in standard locations
    validate_parser.add_argument(
        '--input', '-i',
        help='Input CSV file path (default: auto-detect VBP data in data/raw/dataframes/)'
    )

    # Configure the raw subscription monitoring command
    # add_parser creates a new subcommand parser for 'subscribe-raw'
    # This subcommand displays raw Sierra Chart data for debugging
    subscribe_parser = subparsers.add_parser(
        'subscribe-raw',
        help='Subscribe to raw VBP data from Sierra Chart (debugging)'
    )

    # Add optional --bars argument to specify historical context
    # Controls how many historical bars to fetch initially
    subscribe_parser.add_argument(
        '--bars', '-b',
        type=int,
        default=50,
        help='Number of historical bars to fetch initially (default: 50)'
    )

    # Add optional --interval argument to control update frequency
    # 'close' updates only on bar close, 'tick' updates on every price change
    subscribe_parser.add_argument(
        '--interval', '-t',
        choices=['close', 'tick'],
        default='close',
        help='Update interval: close (bar close only) or tick (every change) (default: close)'
    )

    # Configure the feature export command
    # add_parser creates a new subcommand parser for 'export-features'
    # This subcommand exports processed features for ML training
    export_parser = subparsers.add_parser(
        'export-features',
        help='Export processed features for ML model training'
    )

    # Add optional --input/-i argument to specify input CSV file
    # If not provided, auto-detects VBP data files in standard locations
    export_parser.add_argument(
        '--input', '-i',
        help='Input CSV file path (default: auto-detect VBP data in data/raw/dataframes/)'
    )

    # Add optional --output/-o argument to specify output file path
    # If not provided, creates timestamped file in data/processed/
    export_parser.add_argument(
        '--output', '-o',
        help='Output file path (default: data/processed/features_TIMESTAMP.{format})'
    )

    # Add optional --format/-f argument to select export format
    # Supports CSV, Parquet, and HDF5 formats
    export_parser.add_argument(
        '--format', '-f',
        choices=['csv', 'parquet', 'hdf5'],
        default='csv',
        help='Output format: csv, parquet, or hdf5 (default: csv)'
    )

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

    # Check if user chose the 'validate-data' subcommand
    elif args.command == 'validate-data':
        # Execute data validation with optional input path
        # args.input will be None if --input wasn't provided (auto-detects file)
        validate_data(args.input)

    # Check if user chose the 'subscribe-raw' subcommand
    elif args.command == 'subscribe-raw':
        # Execute raw subscription monitoring with custom settings
        # args.bars controls historical context, args.interval controls update frequency
        subscribe_raw(args.bars, args.interval)

    # Check if user chose the 'export-features' subcommand
    elif args.command == 'export-features':
        # Execute feature export with optional paths and format
        # args.input and args.output can be None (triggers auto-detection)
        # args.format specifies the output file format (csv, parquet, hdf5)
        export_features(args.input, args.output, args.format)

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


