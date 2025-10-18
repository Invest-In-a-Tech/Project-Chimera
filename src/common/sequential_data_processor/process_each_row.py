"""
Sequential Row Processor Module.

This module provides functionality for processing individual rows of market data in a
sequential manner, serving as a coordinator between raw data rows and the market data
processor. It acts as a bridge layer that handles row-by-row processing with proper
error handling and logging.

The module is designed for sequential data processing workflows where each data point
(row) needs to be processed individually with its temporal context. This is essential
for time-series analysis, feature engineering, and model training where each observation
needs to be enriched with historical lookback data.

Processing Flow:
1. Receive a single row of market data with its timestamp
2. Pass the row to the market data processor for enrichment
3. Market data processor adds historical context (lookback periods)
4. Return enriched market data dictionary with current and historical features
5. Handle any errors gracefully with logging

Use Cases:
- Sequential processing of historical data for model training
- Real-time processing of streaming data with historical context
- Feature engineering where each data point needs lookback data
- Time-series analysis requiring temporal context for each observation

Design Pattern:
This module implements a façade/wrapper pattern, providing a simplified interface
to the more complex market data processing logic. It adds error handling and logging
around the core processing functionality.

Classes:
    ProcessEachRow: Coordinator class for sequential row-by-row data processing.

Dependencies:
    logging: For error logging and monitoring during processing
    pandas: For Series and DataFrame data structures (via type hints)
    typing: For type annotations and type safety
    ProcessMarketData: Core processor for enriching market data with historical context

Example:
    ```python
    from common.sequential_data_processor.process_each_row import ProcessEachRow
    import pandas as pd

    # Initialize the row processor
    processor = ProcessEachRow(lookback_period=10)

    # Process a single row with its timestamp
    row = df.loc['2025-01-15 09:30:00']
    timestamp = pd.Timestamp('2025-01-15 09:30:00')
    market_data = processor.process_rows(row, timestamp, df)

    # Access current and historical data
    print(f"Current Close: {market_data['current']['Close']}")
    print(f"Previous Close: {market_data['t-1']['Close']}")
    print(f"Lookback available: {len(market_data) - 1} periods")
    ```

Author: Roy Williams
Version: 1.0.0
"""

# Standard library imports
# Import logging module for error tracking and monitoring during processing
# Provides structured logging with levels for debugging and error handling
import logging

# Import type hinting primitives for documenting function signatures
# Dict: typed dictionary container, Any: accepts any type
# Union: multiple possible types, datetime: date/time type
from typing import Dict, Any, Union
from datetime import datetime

# Third-party imports
# Import pandas for DataFrame and Series data structures used in processing
import pandas as pd

# Local imports
# Import ProcessMarketData for core market data enrichment with historical context
# This processor adds lookback data to each row for time-series analysis
from src.common.market_data_processor.process_market_data import ProcessMarketData


class ProcessEachRow:
    """
    Sequential processor for handling individual market data rows with historical context.

    This class provides a high-level interface for processing market data rows one at a time,
    coordinating between the raw data and the market data processor. It handles initialization
    of the underlying processor, error management, and logging for robust sequential processing.

    The processor is designed for workflows that require processing data points sequentially,
    where each row needs to be enriched with historical lookback data from previous periods.
    This is essential for:
    - Feature engineering with temporal context
    - Model training with sequential data
    - Real-time processing with historical features
    - Time-series analysis requiring lookback windows

    The class acts as a wrapper/façade around ProcessMarketData, providing:
    - Simplified interface for row processing
    - Comprehensive error handling with logging
    - Consistent processing workflow
    - Easy integration into data pipelines

    Attributes:
        market_data_processor (ProcessMarketData): Instance of the core market data processor.
            Handles the actual processing logic including lookback data retrieval and
            feature enrichment. Configured with specified lookback period.

    Example:
        >>> # Initialize with default lookback period (10)
        >>> processor = ProcessEachRow()
        >>>
        >>> # Process individual rows in a loop
        >>> for timestamp, row in df.iterrows():
        ...     market_data = processor.process_rows(row, timestamp, df)
        ...     # Use enriched market data for analysis or model input
        ...     features = extract_features(market_data)
        ...     predictions = model.predict(features)
        >>>
        >>> # Custom lookback period
        >>> processor = ProcessEachRow(lookback_period=20)
        >>> market_data = processor.process_rows(row, timestamp, df)

    Note:
        - The processor maintains state (market_data_processor instance)
        - All processing errors are logged but not re-raised by default
        - Returns None if processing fails (check logs for details)
        - Thread-safe for sequential processing (not tested for concurrent use)
    """

    def __init__(self, lookback_period: int = 10) -> None:
        """
        Initialize the ProcessEachRow instance with configured lookback period.

        Sets up the sequential row processor by creating an instance of the market
        data processor with the specified lookback period. Logs initialization for
        monitoring and debugging purposes.

        Args:
            lookback_period (int, optional): Number of previous periods to include
                in historical lookback data. Defaults to 10.

                This determines how many previous time periods (e.g., bars, rows)
                will be included in the market_data dictionary returned by process_rows().

                - Larger values: More historical context, higher memory usage
                - Smaller values: Less context, faster processing
                - Minimum: 0 (current data only, no lookback)

                Example values:
                - 10: Last 10 periods (good for short-term patterns)
                - 50: Last 50 periods (medium-term context)
                - 200: Last 200 periods (long-term trends)

        Returns:
            None

        Example:
            >>> # Default lookback (10 periods)
            >>> processor = ProcessEachRow()
            >>>
            >>> # Custom lookback for more historical context
            >>> processor = ProcessEachRow(lookback_period=50)
            >>>
            >>> # Minimal lookback (current data only)
            >>> processor = ProcessEachRow(lookback_period=0)

        Note:
            - Initialization is lightweight (no heavy operations)
            - Logs to INFO level for tracking processor creation
            - The lookback_period is passed to ProcessMarketData
            - Can create multiple instances with different lookback periods
        """
        # Log initialization at INFO level for monitoring and debugging
        # Helps track when processors are created and identify processing starts
        # Uses module-level logger (logging.info, not self.logger)
        logging.info("ProcessEachRow initialized with lookback_period: %d", lookback_period)

        # Create an instance of ProcessMarketData with the specified lookback period
        # This processor handles the core logic of enriching rows with historical data
        # Configured with lookback_period to control how much historical context is included
        # The processor will retrieve up to lookback_period previous time periods for each row
        self.market_data_processor: ProcessMarketData = ProcessMarketData(
            lookback_period=lookback_period
        )

    def process_rows(
        self,
        row: pd.Series,
        current_timestamp: Union[pd.Timestamp, datetime],
        vbp_chart_data_df: pd.DataFrame
    ) -> Dict[str, Dict[str, Any]] | None:
        """
        Process a single row of market data with historical lookback context.

        This method takes an individual row of market data, enriches it with historical
        lookback data, and returns a comprehensive dictionary containing both current
        and historical market information. It delegates the core processing to the
        market data processor while providing error handling and logging.

        The returned dictionary contains:
        - 'current': Current period's data with all columns from the row
        - 't-1', 't-2', ..., 't-N': Historical periods' data (N = lookback_period)

        Each period's dictionary includes:
        - All original columns (Open, High, Low, Close, Volume, etc.)
        - Timestamp information
        - Index position

        Args:
            row (pd.Series): Current row of market data being processed.
                Should contain columns like Open, High, Low, Close, Volume, etc.
                This is typically extracted from a DataFrame using .loc[] or .iloc[].
                Example: df.loc['2025-01-15 09:30:00']

            current_timestamp (Union[pd.Timestamp, datetime]): Timestamp for the current row.
                Used to locate the row's position in the time series and retrieve
                historical data. Should match the index value of the row.
                Example: pd.Timestamp('2025-01-15 09:30:00')

            vbp_chart_data_df (pd.DataFrame): Complete DataFrame containing all market data.
                Must have a datetime index and include the current row plus sufficient
                historical data for lookback. Used to retrieve previous periods' data.
                Should contain columns: Open, High, Low, Close, Volume, and any indicators.

        Returns:
            Dict[str, Dict[str, Any]]: Nested dictionary containing processed market data:

                Structure:
                {
                    'current': {
                        'Open': 150.25,
                        'High': 151.00,
                        'Low': 150.00,
                        'Close': 150.75,
                        'Volume': 1000000,
                        'timestamp': Timestamp('2025-01-15 09:30:00'),
                        'index': 42,
                        ... (other columns)
                    },
                    't-1': {
                        'Open': 149.50,
                        'Close': 150.25,
                        'timestamp': Timestamp('2025-01-15 09:29:00'),
                        ... (previous period's data)
                    },
                    't-2': { ... },  # Two periods ago
                    ...
                    't-N': { ... }   # N periods ago (N = lookback_period)
                }

                Returns None if processing fails (error is logged).

        Raises:
            Exception: Exceptions are caught and logged, not re-raised.
                The method returns None on error to allow processing to continue.
                Check logs for error details if None is returned.

        Example:
            >>> processor = ProcessEachRow(lookback_period=5)
            >>>
            >>> # Process a single row
            >>> row = df.loc['2025-01-15 09:30:00']
            >>> timestamp = pd.Timestamp('2025-01-15 09:30:00')
            >>> market_data = processor.process_rows(row, timestamp, df)
            >>>
            >>> # Access current data
            >>> current_close = market_data['current']['Close']
            >>> current_volume = market_data['current']['Volume']
            >>>
            >>> # Access historical data
            >>> prev_close = market_data['t-1']['Close']
            >>> two_bars_ago = market_data['t-2']['Close']
            >>>
            >>> # Calculate features
            >>> close_change = market_data['current']['Close'] - market_data['t-1']['Close']
            >>> avg_volume = sum(market_data[f't-{i}']['Volume'] for i in range(1, 6)) / 5
            >>>
            >>> # Loop through all rows
            >>> for timestamp, row in df.iterrows():
            ...     market_data = processor.process_rows(row, timestamp, df)
            ...     if market_data is not None:
            ...         # Process the enriched data
            ...         features = extract_features(market_data)

        Note:
            - This method does not modify the input DataFrame or Series
            - Errors are logged but not re-raised (returns None on failure)
            - The current row's timestamp must exist in vbp_chart_data_df index
            - Historical data availability depends on current_timestamp position
            - At the start of the dataset, fewer lookback periods may be available
            - Use lazy % formatting in logs for Pylint compliance
        """
        try:
            # Delegate core processing to the market data processor
            # This method handles the actual logic of:
            # 1. Extracting data from the current row
            # 2. Retrieving historical data for lookback periods
            # 3. Building the nested dictionary structure
            # 4. Validating data structure consistency
            # Returns a dictionary with 'current' and historical period keys ('t-1', 't-2', etc.)
            market_data = self.market_data_processor.process_market_data(
                current_row=row,
                current_timestamp=current_timestamp,
                vbp_chart_data_df=vbp_chart_data_df
            )

            # Return the processed market data dictionary to the caller
            # Contains enriched data with current and historical context
            # Ready for feature engineering, model input, or analysis
            return market_data

        # Catch specific exceptions that may occur during processing
        # Allows processing to continue even if individual rows fail
        except (KeyError, ValueError, IndexError, TypeError) as e:
            # Log the error with timestamp for debugging and monitoring
            # Uses lazy % formatting for Pylint compliance
            # Includes timestamp to identify which row failed and exception message
            logging.error("Error processing row for timestamp %s: %s", current_timestamp, e)

            # Return None to indicate processing failure
            # Caller can check for None and handle accordingly
            # Error details are in the logs for debugging
            return None
