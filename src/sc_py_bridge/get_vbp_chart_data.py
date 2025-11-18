"""
Volume by Price (VBP) Chart Data Fetcher Module.

This module provides functionality to fetch and process Volume by Price (VBP) chart data
from Sierra Chart using the trade29 SC bridge. The module contains the GetVbpData class
which handles the connection to Sierra Chart, data retrieval, and processing of VBP data
into structured pandas DataFrames.

Volume by Price (VBP) data shows the distribution of trading volume across different price
levels for each time bar. This is critical for analyzing market structure, identifying
support/resistance zones, and understanding where traders are most active.

Classes:
    GetVbpData: Main class for fetching and processing VBP chart data from Sierra Chart.

Dependencies:
    - trade29.sc: Sierra Chart bridge library for data communication
    - pandas: Data manipulation and analysis library for time-series processing
    - typing: Type hinting support for improved code quality and IDE integration

Example:
    Basic usage of the GetVbpData class:

    ```python
    from sc_py_bridge.get_vbp_chart_data import GetVbpData

    # Initialize with default settings (1M bars, auto bridge creation)
    vbp_fetcher = GetVbpData()

    # Fetch and process VBP data into a DataFrame
    vbp_data = vbp_fetcher.get_vbp_chart_data()

    # Access specific data points
    print(vbp_data['Close'])  # Price data
    print(vbp_data['RVOL'])   # Relative volume indicator
    print(vbp_data['Price'])  # VBP price levels

    # Clean up connection when done
    vbp_fetcher.stop_bridge()
    ```

Author: Roy Williams
Version: 1.0.0
"""

# Enable postponed evaluation of type annotations to avoid forward-reference issues
# This allows using classes in type hints before they're fully defined
from __future__ import annotations

# Standard library imports
# Import type hinting primitives for documenting function signatures and return types
# Any: accepts any type, List: typed list container, Optional: value or None
from typing import Any, List, Optional

# Import logging module to enable debug/info/warning messages throughout execution
import logging

# Third-party imports
# Import pandas for DataFrame manipulation, time-series analysis, and data transformation
import pandas as pd

# Import Sierra Chart bridge components for communicating with the Sierra Chart platform
# SCBridge: main connection object, SubgraphQuery: study/indicator query builder
# constants: predefined Sierra Chart constants for data types and studies
from trade29.sc import SCBridge, SubgraphQuery, constants

# Configure pandas display options to show complete DataFrames without truncation
# This is helpful during development and debugging to see all data without ellipsis
pd.set_option('display.max_rows', None)      # Show all rows instead of truncating
pd.set_option('display.max_columns', None)   # Show all columns instead of truncating
pd.set_option('display.width', None)         # Don't wrap columns to screen width
pd.set_option('display.max_colwidth', None)  # Show full content of each cell

# Local imports
# (No local imports currently - this module is self-contained)

# Create a module-level logger instance for this file
# Uses __name__ so log messages show which module they came from (sc_py_bridge.get_vbp_chart_data)
logger = logging.getLogger(__name__)

class GetVbpData:
    """
    Fetch and process Volume by Price (VBP) chart data from Sierra Chart.

    This class manages the complete workflow of retrieving VBP data from Sierra Chart,
    including establishing the bridge connection, requesting historical data with
    specified base data and study indicators, and transforming the nested VBP structure
    into a flat, analysis-ready pandas DataFrame.

    The class handles:
    - Bridge connection management (creation and cleanup)
    - Chart data requests with configurable historical depth
    - VBP data extraction and flattening (nested lists to tabular format)
    - Column normalization and renaming for consistency
    - Integration of price data with volume distribution data

    Attributes:
        bridge (SCBridge): Active connection instance for communicating with Sierra Chart.
            Used to send data requests and receive responses.
        columns_to_drop (List[str]): List of column names to remove from final DataFrame.
            Useful for cleaning up helper columns that aren't needed in analysis.
        historical_bars (int): Number of historical bars to request from Sierra Chart.
            Default is 1,000,000 bars to ensure comprehensive historical coverage.

    Example:
        >>> # Initialize with custom settings
        >>> fetcher = GetVbpData(historical_bars=5000)
        >>> df = fetcher.get_vbp_chart_data()
        >>>
        >>> # Access VBP data
        >>> print(df[['Price', 'TotalVolume', 'Close']].head())
        >>>
        >>> # Cleanup
        >>> fetcher.stop_bridge()

    Note:
        - The bridge connection should be closed with stop_bridge() when done
        - Each bar's VBP data is expanded into multiple rows (one per price level)
        - DateTime index enables easy time-series operations and filtering
    """

    # SCBridge instance that maintains the connection to Sierra Chart
    # This is the communication channel for all data requests and responses
    # Type: SCBridge object from trade29.sc library
    bridge: SCBridge

    # List of column names to remove from the processed DataFrame
    # Used during cleanup to drop temporary or unwanted columns
    # Type: List of strings representing column names
    columns_to_drop: List[str]

    # Number of historical bars to retrieve in each chart data request
    # Determines how much historical data is available for analysis
    # Type: Integer representing bar count (default 1,000)
    historical_bars: int

    def __init__(
        self,
        # Optional pre-existing SCBridge instance for connection reuse
        # If None, a new bridge will be created automatically
        # Useful when managing multiple data fetchers with a single connection
        bridge: Optional[SCBridge] = None,
        # Optional list of column names to drop from the final DataFrame
        # If None, defaults to ['IsBarClosed'] which is a helper column
        # Allows customization of which columns are excluded from results
        columns_to_drop: Optional[List[str]] = None,
        # Number of historical bars to request from Sierra Chart
        # Default 1,000 provides extensive historical depth
        # Can be reduced for faster queries when less history is needed
        historical_bars: int = 1000
    ) -> None:
        """
        Initialize the GetVbpData instance with configuration parameters.

        Sets up the Sierra Chart bridge connection, configures data cleanup settings,
        and establishes the historical depth for data requests. This constructor
        allows flexible configuration while providing sensible defaults for common use cases.

        Args:
            bridge (Optional[SCBridge], optional): Pre-existing SCBridge instance to reuse.
                If None, creates a new bridge connection automatically. Defaults to None.
                Useful for sharing a single bridge across multiple fetcher instances.
            columns_to_drop (Optional[List[str]], optional): Column names to remove from
                the processed DataFrame. If None, defaults to ['IsBarClosed']. Defaults to None.
                The IsBarClosed column is a helper field that indicates if a bar is complete.
            historical_bars (int, optional): Number of historical bars to request.
                Defaults to 1000000. Larger values provide more historical data but may
                increase query time and memory usage.

        Returns:
            None

        Example:
            >>> # Initialize with defaults
            >>> fetcher1 = GetVbpData()
            >>>
            >>> # Initialize with custom settings
            >>> fetcher2 = GetVbpData(
            ...     historical_bars=10000,
            ...     columns_to_drop=['IsBarClosed', 'ID9.SG4'],
            ... )
            >>>
            >>> # Initialize with shared bridge
            >>> shared_bridge = SCBridge()
            >>> fetcher3 = GetVbpData(bridge=shared_bridge)
            >>> fetcher4 = GetVbpData(bridge=shared_bridge)

        Note:
            - Creating multiple instances with separate bridges consumes more resources
            - The bridge connection is NOT automatically closed; call stop_bridge() explicitly
            - Historical bars count applies to each get_chart_data() request
        """
        # Log a debug message to track object instantiation for troubleshooting
        # Helps identify when and how many instances are created during execution
        logger.debug("Initializing GetVbpData class")

        # Assign the bridge parameter or create a new SCBridge if none was provided
        # The ternary expression checks if bridge is not None and reuses it
        # Falls back to creating a fresh SCBridge when no shared instance is provided
        # This pattern enables both standalone usage and connection sharing scenarios
        self.bridge = bridge if bridge is not None else SCBridge()

        # Assign the columns_to_drop parameter or use default list with 'IsBarClosed'
        # IsBarClosed is a helper column that indicates bar completion status
        # It's typically not needed in final analysis, so we remove it by default
        self.columns_to_drop = (
            columns_to_drop if columns_to_drop is not None else ['IsBarClosed']
        )

        # Store the number of historical bars to request in subsequent data fetches
        # This value is used by fetch_vbp_chart_data() when calling bridge.get_chart_data()
        # Affects how much historical data is available for analysis and backtesting
        self.historical_bars = historical_bars

    def fetch_vbp_chart_data(self) -> pd.DataFrame:
        """
        Fetch raw Volume by Price (VBP) chart data from Sierra Chart.

        This method constructs and executes a data request to Sierra Chart, specifying
        which base data fields (OHLCV) and study indicators to retrieve. The VBP data
        is embedded in the response as nested lists within each bar's data. This raw
        DataFrame requires further processing to be analysis-ready.

        The method requests:
        - Base OHLCV (Open, High, Low, Close, Volume) data
        - Volume by Price distribution for each bar
        - Study indicators: Relative Volume, Daily OHLC, Large Trades
        - Historical bars as configured in __init__
        - Live (in-progress) bar data

        Returns:
            pd.DataFrame: Raw chart data with nested VBP lists in 'VolumeByPrice' column.
                Each row represents a time bar, and the VolumeByPrice column contains
                a list of [Price, BidVol, AskVol, TotalVolume, NumOfTrades] arrays.
                Requires processing via process_vbp_chart_data() for flat structure.

        Raises:
            ConnectionError: If bridge cannot connect to Sierra Chart.
            ValueError: If chart data request returns invalid or empty data.

        Example:
            >>> fetcher = GetVbpData()
            >>> raw_df = fetcher.fetch_vbp_chart_data()
            >>> print(raw_df.columns)
            >>> print(type(raw_df['VolumeByPrice'].iloc[0]))  # List of lists
            >>> # Further processing needed to flatten VBP data

        Note:
            - This returns RAW data with nested VBP structure
            - Call process_vbp_chart_data() to flatten and normalize
            - Study IDs are Sierra Chart-specific (6=RVOL, 4=Daily OHLC, 9=Large Trades)
            - The response includes live bar if market is open
        """
        # Define the base price/volume data fields to retrieve from Sierra Chart
        # These are the standard OHLCV (Open, High, Low, Close, Volume) fields
        # Using Sierra Chart's constant definitions ensures compatibility with the API
        price_data_to_fetch = [
            # Opening price of each bar - first trade price in the time period
            constants.SCBaseData.SC_OPEN,
            # Highest price reached during the bar - used for range and volatility analysis
            constants.SCBaseData.SC_HIGH,
            # Lowest price reached during the bar - used for range and support/resistance
            constants.SCBaseData.SC_LOW,
            # Last/closing price of the bar - most commonly used for analysis and indicators
            constants.SCBaseData.SC_LAST,
            # Total volume traded during the bar - critical for volume analysis
            constants.SCBaseData.SC_VOLUME
        ]

        # Define which Sierra Chart study indicators and subgraphs to include in the response
        # Each SubgraphQuery specifies a study_id and which of its subgraphs to retrieve
        # Studies are pre-configured indicators in Sierra Chart that calculate derived values
        sg_to_fetch = [
            # Study ID 6: Relative Volume (RVOL) - compares current volume to average
            # Subgraph 1 returns the RVOL value (ratio of current volume to typical volume)
            # Used to identify abnormal volume activity
            SubgraphQuery(study_id=6, subgraphs=[1]),

            # Study ID 4: Daily High, Low, Open levels
            # Subgraph 1: Today's opening price (first trade of the session)
            # Subgraph 2: Today's highest price (intraday high)
            # Subgraph 3: Today's lowest price (intraday low)
            SubgraphQuery(study_id=4, subgraphs=[1, 2, 3]),

            # Study ID 9: Large Trades analysis - tracks unusually large orders
            # Subgraph 1: Maximum volume in a single large trade
            # Subgraph 2: Total volume from all large trades
            # Subgraph 3: Bid volume from large trades
            # Subgraph 4: Ask volume from large trades
            # Used to identify institutional activity and significant market moves
            SubgraphQuery(study_id=9, subgraphs=[1, 2, 3, 4]),

            # Study ID 2: Volume Delta - tracks net buying/selling pressure
            # Subgraph 1: Delta - Ask volume minus Bid volume difference
            # Subgraph 10: Cumulative Delta - running total of net volume delta
            SubgraphQuery(study_id=2, subgraphs=[1, 10])
        ]

        # Execute the chart data request through the Sierra Chart bridge
        # This sends the request to Sierra Chart and receives the response object
        # The response contains all requested data in Sierra Chart's native format
        vbp_chart_data_response = self.bridge.get_chart_data(
            # Unique identifier for this data stream/subscription
            # Used by the bridge to track and manage multiple concurrent data requests
            key='15minkey',

            # Enable Volume by Price data in the response
            # This adds the VolumeByPrice column with nested price distribution data
            # Without this flag, only standard OHLCV data would be returned
            include_volume_by_price=True,

            # Specify how many historical bars to retrieve
            # Uses the value set in __init__ (default 1,000,000)
            # More bars = more historical data but slower query
            historical_bars=self.historical_bars,

            # Include the current in-progress bar if market is actively trading
            # This provides real-time data for the most recent incomplete bar
            # Useful for live monitoring and trading systems
            include_live_bar=True,

            # Pass the list of base data fields (OHLCV) to include
            # These are the fundamental price/volume series for each bar
            base_data=price_data_to_fetch,

            # Pass the list of study subgraph queries to include
            # These add calculated indicators (RVOL, Daily OHLC, Large Trades)
            sg_data=sg_to_fetch
        )

        # Convert the Sierra Chart response object into a pandas DataFrame
        # The as_df() method transforms the native format into tabular structure
        # Returns a DataFrame with DateTime index and columns for all requested data
        # The VolumeByPrice column contains nested lists that need further processing
        return vbp_chart_data_response.as_df()

    def process_vbp_chart_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process and flatten raw VBP chart data into analysis-ready DataFrame.

        This method transforms the nested VolumeByPrice structure from Sierra Chart
        into a flat, tabular format suitable for analysis and visualization. Each bar's
        VBP data (which is a list of price levels with volume distributions) is exploded
        into separate rows, creating multiple rows per time bar - one for each price level
        where volume occurred.

        The processing pipeline:
        1. Extract nested VBP lists and convert to individual DataFrames
        2. Concatenate all VBP data with DateTime keys to maintain time reference
        3. Sort by time and price for consistent ordering
        4. Join VBP data back with original OHLCV and indicator data
        5. Clean up helper columns and normalize column names

        Args:
            df (pd.DataFrame): Raw chart data from fetch_vbp_chart_data() containing
                nested VolumeByPrice column. Expected to have DateTime index and
                columns: Open, High, Low, Last, Volume, VolumeByPrice, and study data.

        Returns:
            pd.DataFrame: Processed DataFrame with flattened VBP structure:
                - DateTime index (may have duplicate timestamps for different price levels)
                - Price, BidVol, AskVol, TotalVolume, NumOfTrades columns (from VBP)
                - Original OHLCV columns (Open, High, Low, Close, Volume)
                - Study indicator columns (RVOL, TodayOpen, TodayHigh, etc.)
                - Sorted by DateTime then Price for consistent structure

        Raises:
            KeyError: If expected columns (VolumeByPrice, study columns) are missing.
            ValueError: If VolumeByPrice contains invalid nested structure.

        Example:
            >>> fetcher = GetVbpData()
            >>> raw_df = fetcher.fetch_vbp_chart_data()
            >>> processed_df = fetcher.process_vbp_chart_data(raw_df)
            >>>
            >>> # Check structure
            >>> print(processed_df.columns)
            >>> print(processed_df[['Price', 'TotalVolume', 'Close']].head(10))
            >>>
            >>> # Each bar now has multiple rows (one per price level)
            >>> timestamp = processed_df.index[0]
            >>> bar_data = processed_df.loc[timestamp]
            >>> print(f"Price levels in bar: {len(bar_data)}")

        Note:
            - Output DataFrame has MORE rows than input (one row per price level per bar)
            - DateTime index will have repeated values (multiple price levels per time)
            - Use .loc[timestamp] to get all price levels for a specific bar
            - Column names are standardized (e.g., 'Last' → 'Close', 'ID6.SG1' → 'RVOL')
        """

        # Define nested helper function to convert a single bar's VBP data into a DataFrame
        # This function is called for each bar's VolumeByPrice list to create individual DataFrames
        # Takes a list of lists where each inner list captures price and volume details
        # Format: [Price, BidVol, AskVol, TotalVolume, NumOfTrades]
        def vbp_to_df(vbp_rows: list[list[Any]]) -> pd.DataFrame:
            """
            Convert a single bar's VolumeByPrice nested list into a tabular DataFrame.

            Each bar from Sierra Chart contains a VolumeByPrice field with nested data
            showing how volume was distributed across different price levels. This helper
            flattens that nested structure into a DataFrame with named columns.

            Args:
                vbp_rows (list[list[Any]]): Nested list of VBP data from one bar.
                    Each inner list has 5 elements:
                    [Price, BidVol, AskVol, TotalVolume, NumOfTrades]
                    Example: [[100.25, 50, 75, 125, 10], [100.50, 30, 45, 75, 8]]

            Returns:
                pd.DataFrame: DataFrame with columns ['Price', 'BidVol', 'AskVol',
                    'TotalVolume', 'NumOfTrades'] where each row is a price level
                    from the input bar.

            Example:
                >>> vbp_data = [[100.0, 10, 15, 25, 5], [100.5, 20, 10, 30, 8]]
                >>> result = vbp_to_df(vbp_data)
                >>> print(result)
                   Price  BidVol  AskVol  TotalVolume  NumOfTrades
                0  100.0      10      15           25            5
                1  100.5      20      10           30            8
            """
            # Define the column names for the VBP DataFrame schema
            # Each position in the nested list corresponds to one of these fields
            # Price: price level, BidVol: buy volume, AskVol: sell volume,
            # TotalVolume: total volume at this price, NumOfTrades: trade count
            columns = ['Price', 'BidVol', 'AskVol', 'TotalVolume', 'NumOfTrades']

            # Create and return a DataFrame from the nested lists using the defined schema
            # pandas automatically aligns each inner list to the column names
            # This converts unstructured nested data into structured tabular format
            return pd.DataFrame(vbp_rows, columns=columns)

        # Process all bars' VBP data and concatenate into a single DataFrame
        # This is the core transformation from nested to flat structure
        # List comprehension: [vbp_to_df(v) for v in df['VolumeByPrice']] creates one DF per bar
        # pd.concat combines all individual DataFrames vertically
        # keys=df.index adds the DateTime from each bar as a hierarchical index level
        vbp_combined = pd.concat(
            # Apply vbp_to_df to each bar's VolumeByPrice column to create list of DataFrames
            # Iterates through df['VolumeByPrice'] series, converting each nested list
            [vbp_to_df(v) for v in df['VolumeByPrice']],
            # Use the original DataFrame's index (DateTime) as keys for the hierarchical index
            # This maintains the time reference for each price level
            keys=df.index
        )\
        .reset_index(level=1, drop=True)\
        .reset_index()
        # After concat, we have a MultiIndex (DateTime, row_number)
        # reset_index(level=1, drop=True) removes the row_number level
        # reset_index() converts the DateTime index into a regular column

        # Rename the 'index' column to 'DateTime' for clarity and consistency
        # After reset_index(), pandas names the former index column 'index'
        # We rename it to reflect its actual meaning (timestamp of the bar)
        # inplace=True modifies the DataFrame directly without creating a copy
        vbp_combined.rename(columns={'index': 'DateTime'}, inplace=True)

        # Sort the VBP data by DateTime (primary) and Price (secondary)
        # This ensures chronological order and consistent price ordering within each bar
        # Makes the data easier to visualize and analyze sequentially
        # Result: all price levels for 09:30 bar, then all for 09:31 bar, etc.
        vbp_combined = vbp_combined.sort_values(['DateTime', 'Price'])

        # Convert DateTime column to proper pandas datetime64 dtype
        # Ensures consistent datetime operations, indexing, and time-series functionality
        # pd.to_datetime handles various input formats and converts to standardized type
        vbp_combined['DateTime'] = pd.to_datetime(vbp_combined['DateTime'])

        # Set DateTime as the index to enable time-series operations
        # This allows .loc[timestamp] queries and time-based filtering/slicing
        # inplace=True modifies the DataFrame directly
        # Note: Index will have duplicate DateTime values (multiple price levels per bar)
        vbp_combined.set_index('DateTime', inplace=True)

        # Merge the flattened VBP data back with the original OHLCV and indicator data
        # Drop VolumeByPrice column since it's now exploded into separate columns
        # Join operation uses index (DateTime) to align data
        # how='outer' ensures we keep all rows from both DataFrames
        # Result: DataFrame with both price data and VBP distribution data
        combined_df = df.drop(columns=['VolumeByPrice']).join(vbp_combined, how='outer')

        # Remove columns specified in columns_to_drop (typically helper columns like IsBarClosed)
        # inplace=True modifies DataFrame directly
        # errors='ignore' prevents errors if specified columns don't exist
        # This cleanup step removes temporary/internal columns not needed for analysis
        combined_df.drop(columns=self.columns_to_drop, inplace=True, errors='ignore')

        # Standardize column names to match project conventions and improve readability
        # Maps Sierra Chart's internal column names to more intuitive names
        # inplace=True modifies the DataFrame directly without creating a copy
        combined_df.rename(columns={
            # Rename 'Last' to 'Close' - more commonly used term for closing/last price
            'Last': 'Close',
            # Study ID 6, Subgraph 1: Relative Volume indicator
            # RVOL shows current volume relative to average (e.g., 1.5 = 50% above average)
            'ID6.SG1': 'RVOL',

            # Study ID 4, Subgraph 1: Today's session opening price
            # First trade price of the current trading day
            'ID4.SG1': 'TodayOpen',

            # Study ID 4, Subgraph 2: Today's session high price
            # Highest price reached during current trading day
            'ID4.SG2': 'TodayHigh',

            # Study ID 4, Subgraph 3: Today's session low price
            # Lowest price reached during current trading day
            'ID4.SG3': 'TodayLow',

            # Study ID 9, Subgraph 1: Large Trades - Maximum volume in single trade
            # Largest individual trade volume detected in the bar
            'ID9.SG1': 'LTMaxVol',

            # Study ID 9, Subgraph 2: Large Trades - Total volume from all large trades
            # Sum of volume from all trades classified as "large"
            'ID9.SG2': 'LTTotalVol',

            # Study ID 9, Subgraph 3: Large Trades - Bid side volume
            # Volume from large trades executed at bid (selling pressure)
            'ID9.SG3': 'LargeBidTrade',

            # Study ID 9, Subgraph 4: Large Trades - Ask side volume
            # Volume from large trades executed at ask (buying pressure)
            'ID9.SG4': 'LargeAskTrade',

            # Study ID 2, Subgraph 1: Volume Delta - Net volume delta
            'ID2.SG1': 'Delta',

            # Study ID 2, Subgraph 10: Volume Delta - Cumulative delta
            'ID2.SG10': 'CumulativeDelta'
        }, inplace=True)

        # Return the fully processed, flattened, and normalized DataFrame
        # Ready for analysis, visualization, and feature engineering
        # Contains integrated price data, volume distribution, and indicators

        return combined_df

    def get_vbp_chart_data(self) -> pd.DataFrame:
        """
        Fetch and process VBP chart data in one complete workflow.

        This is the main public method that orchestrates the complete data pipeline:
        fetching raw data from Sierra Chart and transforming it into an analysis-ready
        DataFrame. This method combines fetch_vbp_chart_data() and process_vbp_chart_data()
        into a single convenient call.

        The complete workflow:
        1. Request chart data from Sierra Chart via the bridge
        2. Receive raw data with nested VBP structure
        3. Flatten VBP data into tabular format
        4. Join VBP with OHLCV and indicator data
        5. Normalize column names and clean up
        6. Return ready-to-use DataFrame

        Returns:
            pd.DataFrame: Fully processed VBP chart data with:
                - DateTime index (may have duplicates for multiple price levels)
                - OHLCV columns (Open, High, Low, Close, Volume)
                - VBP columns (Price, BidVol, AskVol, TotalVolume, NumOfTrades)
                - Indicator columns (RVOL, TodayOpen/High/Low, LT metrics)
                - Sorted by DateTime and Price for consistent structure

        Raises:
            ConnectionError: If Sierra Chart bridge connection fails.
            ValueError: If data format is invalid or processing fails.
            KeyError: If expected columns are missing from Sierra Chart response.

        Example:
            >>> # Simple usage
            >>> fetcher = GetVbpData()
            >>> df = fetcher.get_vbp_chart_data()
            >>> print(df.head())
            >>>
            >>> # Analyze specific timestamp
            >>> timestamp = df.index[100]
            >>> bar_data = df.loc[timestamp]
            >>> print(f"Price levels: {bar_data['Price'].values}")
            >>> print(f"Close: {bar_data['Close'].iloc[0]}")
            >>>
            >>> # Filter by time
            >>> morning_data = df.between_time('09:30', '12:00')
            >>>
            >>> # Cleanup
            >>> fetcher.stop_bridge()

        Note:
            - This method internally calls fetch_vbp_chart_data() and process_vbp_chart_data()
            - Each call creates a new request to Sierra Chart (not cached)
            - For multiple queries, consider caching the result
            - Remember to call stop_bridge() when done to clean up connection
        """
        # Step 1: Fetch raw chart data from Sierra Chart
        # This retrieves OHLCV data, study indicators, and nested VBP lists
        # Returns DataFrame with VolumeByPrice column containing nested structure
        raw_df = self.fetch_vbp_chart_data()

        # Step 2: Transform the raw data into a flattened, analysis-friendly DataFrame
        # This expands nested VBP structure into separate rows
        # Joins VBP data with OHLCV/indicator data and normalizes column names
        processed_df = self.process_vbp_chart_data(raw_df)

        # Step 3: Return the fully processed dataset ready for analysis
        # DataFrame is now in optimal format for time-series analysis, visualization,
        # feature engineering, and machine learning applications
        return processed_df

    def stop_bridge(self) -> None:
        """
        Stop and clean up the Sierra Chart bridge connection.

        This method properly closes the connection to Sierra Chart, releasing
        resources and ensuring clean shutdown. Should be called when done fetching
        data to prevent resource leaks and ensure graceful disconnection.

        Best practice is to call this method after all data fetching operations
        are complete, or use a try/finally block to ensure cleanup even if errors occur.

        Returns:
            None

        Example:
            >>> # Basic usage
            >>> fetcher = GetVbpData()
            >>> df = fetcher.get_vbp_chart_data()
            >>> # ... use data ...
            >>> fetcher.stop_bridge()
            >>>
            >>> # With error handling
            >>> fetcher = GetVbpData()
            >>> try:
            ...     df = fetcher.get_vbp_chart_data()
            ...     # ... process data ...
            >>> finally:
            ...     fetcher.stop_bridge()
            >>>
            >>> # Context manager pattern (if implemented)
            >>> with GetVbpData() as fetcher:
            ...     df = fetcher.get_vbp_chart_data()

        Note:
            - Always call this method when done to prevent resource leaks
            - After calling stop_bridge(), the bridge cannot be reused
            - If bridge was passed to __init__, it will be stopped for all users
            - Safe to call multiple times (subsequent calls have no effect)
        """
        # Invoke the bridge's stop() method to terminate the connection
        # This closes the socket, releases system resources, and performs cleanup
        # Ensures graceful shutdown of communication with Sierra Chart
        # Check if bridge exists and has stop_event attribute before calling stop
        if self.bridge is not None and hasattr(self.bridge, 'stop_event'):
            try:
                self.bridge.stop()
            except Exception as e:  # pylint: disable=broad-except
                # Log error but don't raise - cleanup should be graceful
                logger.warning("Error stopping bridge: %s", e)

# if __name__ == "__main__":
#     # Example usage when running this module as a script
#     vbp_data = GetVbpData()
#     result_df = vbp_data.get_vbp_chart_data()

#     # Print the full DataFrame with all rows and columns visible
#     # logger.debug(f"VBP chart data:\n{result_df}")

#     # Use context manager for one-time full display
#     with pd.option_context('display.max_rows', None, 'display.max_columns', None):
#         logger.debug(f"VBP chart data (full display):\n{result_df}")
#     # vbp_data.stop_bridge()
