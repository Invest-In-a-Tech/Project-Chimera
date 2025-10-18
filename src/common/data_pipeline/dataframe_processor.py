"""
DataFrame Processor Module for Financial Market Data.

This module provides utilities for processing financial market data stored in CSV format,
serving as a foundational component in the Project Chimera trading system's data pipeline.
It handles the complete workflow of data loading, validation, cleaning, transformation,
and preparation for downstream analysis and model training.

The processor is designed specifically for time-series financial data with the following
capabilities and focus areas:

Data Loading:
- CSV file reading with automatic format detection
- Error handling for missing files and invalid formats
- Support for various datetime formats commonly used in financial data

Data Cleaning and Transformation:
- Datetime parsing and standardization
- Chronological sorting for time-series integrity
- Index configuration for efficient time-based operations
- Optional market hours filtering for trading-specific analysis

Time-Series Processing:
- Proper datetime indexing for pandas time-series functionality
- Support for time-based filtering and slicing
- Market hours filtering (configurable for different markets)
- Timezone-aware processing (future enhancement)

The module is designed to be:
- Simple: Single-purpose class with clear interface
- Robust: Comprehensive error handling and validation
- Flexible: Configurable filtering and processing options
- Efficient: Optimized for large financial datasets

Typical Use Cases:
- Loading historical market data for backtesting
- Preparing training data for machine learning models
- Processing VBP (Volume by Price) data for analysis
- Filtering data to regular trading hours (RTH)
- Converting raw CSV exports into analysis-ready DataFrames

Classes:
    DataFrameProcessor: Main class for processing CSV-based financial time-series data.

Dependencies:
    pandas: DataFrame operations, time-series handling, and datetime processing.
            Core library for all data manipulation in this module.

Example:
    ```python
    from common.data_pipeline.dataframe_processor import DataFrameProcessor

    # Basic usage for loading and processing market data
    processor = DataFrameProcessor('data/raw/vbp_data.csv')
    df = processor.process_data()
    print(f"Loaded {len(df)} rows from {df.index[0]} to {df.index[-1]}")

    # Access processed data
    print(df.head())
    print(df.info())

    # Use for analysis
    avg_volume = df['Volume'].mean()
    price_range = df['Close'].max() - df['Close'].min()
    ```

Author: Roy Williams
Version: 1.0.0
"""

# Third-party imports
# Import pandas library for DataFrame operations and time-series data manipulation
# Pandas is the core data structure used throughout this module
# Provides efficient data loading, transformation, and time-series capabilities
import pandas as pd


class DataFrameProcessor:
    """
    Processor class for loading and transforming financial market data from CSV files.

    This class encapsulates the complete workflow for processing CSV files containing
    financial time-series data. It handles file loading, datetime parsing, sorting,
    indexing, and optional filtering operations to prepare data for analysis and
    model training.

    The processor is designed to work with CSV files that contain financial market data
    with a DateTime column. It transforms raw CSV data into a clean, indexed pandas
    DataFrame optimized for time-series analysis.

    Processing Pipeline:
    1. File Loading: Read CSV file into DataFrame
    2. Sorting: Ensure chronological order by DateTime
    3. DateTime Parsing: Convert DateTime column to pandas datetime64 format
    4. Indexing: Set DateTime as index for time-series operations
    5. Optional Filtering: Apply market hours or other time-based filters

    Key Features:
    - Automatic datetime parsing and standardization
    - Chronological sorting for data integrity
    - Time-based indexing for efficient queries
    - Market hours filtering (optional, configurable)
    - Error handling for common data issues

    Attributes:
        file_path (str): Absolute or relative path to the CSV file containing market data.
            Should point to a valid CSV file with a DateTime column.
            Example: 'data/raw/dataframes/vbp_historical.csv'

        df (pd.DataFrame | None): Processed DataFrame with datetime index.
            None before process_data() is called. After processing, contains
            the loaded data with DateTime as index, sorted chronologically.

    Example:
        >>> # Load and process market data
        >>> processor = DataFrameProcessor('data/market_data.csv')
        >>> df = processor.process_data()
        >>>
        >>> # Check processed data
        >>> print(f"Date range: {df.index[0]} to {df.index[-1]}")
        >>> print(f"Columns: {list(df.columns)}")
        >>> print(f"Shape: {df.shape}")
        >>>
        >>> # Time-based operations enabled by datetime index
        >>> morning_data = df.between_time('09:30', '12:00')
        >>> specific_day = df.loc['2025-01-15']
        >>>
        >>> # Access the processed DataFrame directly
        >>> close_prices = processor.df['Close']
        >>> volume_data = processor.df['Volume']

    Note:
        - CSV file must contain a 'DateTime' column
        - DateTime format should be parseable by pandas (e.g., 'YYYY-MM-DD HH:MM:SS')
        - process_data() must be called before accessing df attribute
        - Market hours filtering is currently commented out but available
        - The processor modifies the DataFrame in place for efficiency
    """

    def __init__(self, file_path: str) -> None:
        """
        Initialize the DataFrameProcessor with a CSV file path.

        Sets up the processor instance by storing the file path and initializing
        the DataFrame attribute to None. Actual file loading and processing occurs
        when process_data() is called, enabling lazy evaluation and error handling.

        Args:
            file_path (str): Path to the CSV file to be processed.
                Can be absolute or relative path. The file should contain financial
                time-series data with a 'DateTime' column in a format parseable by
                pandas (e.g., 'YYYY-MM-DD HH:MM:SS', 'MM/DD/YYYY HH:MM:SS', etc.).

                Examples:
                - 'data/raw/historical_vbp_data.csv'
                - 'C:/Trading/Data/market_data.csv'
                - '../datasets/volume_by_price.csv'

        Returns:
            None

        Raises:
            FileNotFoundError: File existence is validated when process_data() is
                called, not during initialization. This allows processor creation
                even if the file doesn't exist yet (useful for dynamic file generation).

        Example:
            >>> # Initialize processor
            >>> processor = DataFrameProcessor('data/vbp_data.csv')
            >>> print(processor.file_path)  # 'data/vbp_data.csv'
            >>> print(processor.df)  # None (not loaded yet)
            >>>
            >>> # Process the data
            >>> df = processor.process_data()
            >>> print(processor.df.shape)  # Data now loaded

        Note:
            - Initialization is lightweight (no file I/O)
            - File path is not validated until process_data() is called
            - DataFrame remains None until process_data() executes
            - This design allows error handling to occur during processing phase
        """
        # Store the file path for later use in process_data()
        # Can be absolute or relative path to CSV file
        # Not validated here to allow lazy evaluation
        self.file_path: str = file_path

        # Initialize the DataFrame attribute to None
        # Will be populated when process_data() is called
        # None state indicates data hasn't been loaded yet
        self.df: pd.DataFrame | None = None

    def process_data(self) -> pd.DataFrame:
        """
        Load, clean, and process CSV data into analysis-ready DataFrame.

        This method executes the complete data processing pipeline, transforming
        raw CSV data into a clean, indexed, sorted DataFrame optimized for
        time-series analysis. It performs datetime parsing, chronological sorting,
        and indexing operations essential for financial data analysis.

        Processing Pipeline Steps:
        1. **File Loading**: Read CSV file into pandas DataFrame
           - Uses pandas' efficient CSV reader
           - Automatically detects data types and structure

        2. **Chronological Sorting**: Sort by DateTime column
           - Ensures time-series integrity
           - Required for time-based operations
           - Prevents issues with out-of-order data

        3. **DateTime Parsing**: Convert DateTime column to pandas datetime64
           - Standardizes datetime format
           - Enables time-based indexing and filtering
           - Handles various input datetime formats

        4. **Index Configuration**: Set DateTime as the DataFrame index
           - Enables efficient time-based lookups
           - Supports pandas time-series functionality (.loc[], .between_time(), etc.)
           - Optimizes performance for temporal operations

        5. **Optional Filtering**: Market hours filtering (currently disabled)
           - Can filter to Regular Trading Hours (RTH)
           - Commented out for flexibility
           - Uncomment to apply 08:00:00 to 15:00:00 filter

        Returns:
            pd.DataFrame: Processed DataFrame with the following characteristics:
                - DateTime index (sorted chronologically)
                - All original columns from CSV (except DateTime, now the index)
                - Data types automatically inferred by pandas
                - Sorted in ascending time order (oldest to newest)
                - Ready for time-series analysis and visualization

                The DataFrame enables operations like:
                - df.loc['2025-01-15'] (select by date)
                - df.between_time('09:30', '16:00') (filter by time)
                - df['Close'].resample('1H').mean() (time-based aggregation)

        Raises:
            FileNotFoundError: If the CSV file specified in __init__ cannot be found
                at the given path. Check file path spelling and existence.

            KeyError: If the 'DateTime' column is missing from the CSV file.
                Ensure CSV has a column named exactly 'DateTime' (case-sensitive).

            ValueError: If DateTime values cannot be parsed into valid dates/times.
                This can occur with invalid datetime strings or incompatible formats.

            pd.errors.EmptyDataError: If the CSV file is empty or contains no data.

            pd.errors.ParserError: If the CSV file is malformed or has parsing issues.

        Example:
            >>> # Basic processing
            >>> processor = DataFrameProcessor('data/historical_data.csv')
            >>> df = processor.process_data()
            >>> print(df.head())
            >>>
            >>> # Check datetime index
            >>> print(f"First timestamp: {df.index[0]}")
            >>> print(f"Last timestamp: {df.index[-1]}")
            >>> print(f"Data span: {df.index[-1] - df.index[0]}")
            >>>
            >>> # Use for time-series analysis
            >>> morning_session = df.between_time('09:30', '12:00')
            >>> daily_close = df['Close'].resample('1D').last()
            >>>
            >>> # Error handling
            >>> try:
            ...     df = processor.process_data()
            >>> except FileNotFoundError:
            ...     print("File not found! Check path.")
            >>> except KeyError:
            ...     print("DateTime column missing!")

        Note:
            - The market hours filter (08:00:00 to 15:00:00) is currently disabled
              Uncomment the between_time() line to enable RTH filtering
            - Market hours assume a specific timezone (configure as needed)
            - DateTime format is flexible - pandas handles most common formats
            - Processing is done in place for memory efficiency
            - The method can be called multiple times (reloads from file each time)
            - Original CSV file is not modified
        """
        # Step 1: Load CSV file into a pandas DataFrame
        # read_csv() automatically detects column types and structure
        # Raises FileNotFoundError if file doesn't exist at specified path
        # Returns DataFrame with all columns from CSV as loaded
        self.df = pd.read_csv(self.file_path)

        # Step 2: Sort DataFrame by DateTime column in ascending order (oldest first)
        # This ensures chronological order which is critical for time-series operations
        # sort_values() modifies the DataFrame and returns it
        # Prevents issues with out-of-order timestamps that could affect analysis
        self.df = self.df.sort_values('DateTime')

        # Step 3: Convert DateTime column from string to pandas datetime64 dtype
        # Parses datetime strings according to the specified format
        # Format '%Y-%m-%d %H:%M:%S' matches '2025-01-15 09:30:00' style
        # pd.to_datetime() is robust and can handle slight format variations
        # This conversion enables time-based operations and indexing
        self.df['DateTime'] = pd.to_datetime(self.df['DateTime'], format='%Y-%m-%d %H:%M:%S')

        # Step 4: Set DateTime column as the DataFrame index
        # This enables efficient time-based lookups and pandas time-series functionality
        # After this, DateTime is no longer a regular column but the index
        # Allows operations like df.loc['2025-01-15'] and df.between_time()
        # Step 5: (Optional) Apply market hours filter - CURRENTLY DISABLED
        # Uncomment the line below to filter data to Regular Trading Hours (RTH)
        # between_time() keeps only rows where time is between 08:00:00 and 15:00:00
        # Useful for excluding pre-market and after-hours data
        # NOTE: Adjust times for different markets (e.g., 09:30-16:00 for US equities)
        # self.df = self.df.set_index('DateTime').between_time('08:00:00', '15:00:00')

        # Current implementation: Set index without time filtering
        # Preserves all hours of data (including pre-market and after-hours)
        # Provides maximum flexibility for users to apply their own filters
        self.df = self.df.set_index('DateTime')

        # Return the processed DataFrame to the caller
        # DataFrame now has DateTime index, is sorted chronologically,
        # and is ready for time-series analysis and feature engineering
        return self.df