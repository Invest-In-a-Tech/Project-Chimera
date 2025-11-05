"""
Volume by Price (VBP) Chart Data Subscription Module.

This module provides functionality to subscribe to real-time and historical Volume by Price (VBP)
chart data updates from Sierra Chart using the trade29 SC bridge. Unlike the get_vbp_chart_data
module which performs one-time data fetches, this module establishes a persistent subscription
that continuously receives incremental updates as new bars form or complete.

The subscription model is ideal for:
- Real-time trading systems that need continuous market data
- Live monitoring and visualization of market activity
- Event-driven trading strategies that react to price/volume changes
- Streaming data pipelines and live analysis

Classes:
    SubscribeToVbpChartData: Main class for subscribing to and processing VBP chart data updates.

Dependencies:
    - trade29.sc: Sierra Chart bridge library for data communication and subscriptions
    - pandas: Data manipulation and analysis library for time-series processing
    - typing: Type hinting support for improved code quality and IDE integration

Example:
    Basic usage of the SubscribeToVbpChartData class:

    ```python
    from sc_py_bridge.subscribe_to_vbp_chart_data import SubscribeToVbpChartData

    # Initialize with default settings (50 historical bars, updates on bar close)
    subscriber = SubscribeToVbpChartData()

    try:
        # Continuously fetch and process updates
        while True:
            vbp_update_df = subscriber.get_subscribed_vbp_chart_data()
            print(f"New data received: {len(vbp_update_df)} rows")
            print(vbp_update_df.tail(10))

            # Process the update (your trading logic here)
            # analyze_market_data(vbp_update_df)

    except KeyboardInterrupt:
        print("Stopping subscription...")
    finally:
        # Clean up connection
        subscriber.stop_bridge()
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

# Import Sierra Chart bridge components for real-time data subscriptions
# SCBridge: main connection object, SubgraphQuery: study/indicator query builder
# constants: predefined Sierra Chart constants for data types and studies
from trade29.sc import SCBridge, SubgraphQuery, constants

# Create a module-level logger instance for this file
# Uses __name__ so log messages show which module they came from
logger = logging.getLogger(__name__)

class SubscribeToVbpChartData:
    """
    Subscribe to real-time Volume by Price (VBP) chart data updates from Sierra Chart.

    This class manages a persistent subscription to Sierra Chart for continuous VBP data
    updates. Unlike one-time data fetches, a subscription provides an ongoing stream of
    chart data as bars form and complete. The class handles subscription initialization,
    update retrieval from a response queue, and data processing into analysis-ready format.

    The subscription workflow:
    1. Initialize with desired parameters (historical depth, update frequency, etc.)
    2. Automatically establishes subscription and receives initial historical bars
    3. Continuously receives incremental updates as new bars form/complete
    4. Processes each update by flattening VBP structure and normalizing columns
    5. Returns processed data through get_subscribed_vbp_chart_data() method

    Update modes:
    - on_bar_close=True: Updates only when bars complete (default, more stable)
    - on_bar_close=False: Updates on every tick (higher frequency, live data)

    Attributes:
        bridge (SCBridge): Active connection instance for communicating with Sierra Chart.
            Manages the WebSocket/TCP connection and message queue.
        columns_to_drop (List[str]): List of column names to remove from final DataFrame.
            Used to clean up helper columns like 'IsBarClosed'.
        historical_init_bars (int): Number of historical bars sent in initial subscription.
            Provides context for the real-time data. Default is 50 bars.
        realtime_update_bars (int): Number of most recent bars included in each update.
            Usually 1 (just the latest bar), but can be more for analysis. Default is 1.
        on_bar_close (bool): Whether updates are sent only on bar close (True) or
            on every tick (False). True reduces update frequency. Default is True.
        chart_data_id (int): Unique subscription ID assigned by Sierra Chart bridge.
            Used to filter the correct updates from the response queue.

    Example:
        >>> # Real-time monitoring with custom settings
        >>> subscriber = SubscribeToVbpChartData(
        ...     historical_init_bars=100,
        ...     on_bar_close=True
        ... )
        >>>
        >>> # Process updates in a loop
        >>> while True:
        ...     df = subscriber.get_subscribed_vbp_chart_data()
        ...     print(f"Close: {df['Close'].iloc[-1]}")
        ...     print(f"RVOL: {df['RVOL'].iloc[-1]}")
        >>>
        >>> # Cleanup
        >>> subscriber.stop_bridge()

    Note:
        - Subscription starts automatically in __init__
        - Updates are queued and retrieved with get_subscribed_vbp_chart_data()
        - Always call stop_bridge() to clean up the subscription
        - Each instance maintains its own subscription ID
    """

    # SCBridge instance that maintains the connection to Sierra Chart
    # Manages the subscription, WebSocket communication, and response queue
    # Type: SCBridge object from trade29.sc library
    bridge: SCBridge

    # List of column names to remove from the processed DataFrame
    # Used during cleanup to drop temporary or unwanted columns
    # Type: List of strings representing column names
    columns_to_drop: List[str]

    # Number of historical bars to receive when subscription is first established
    # Provides initial context before real-time updates begin
    # Type: Integer representing bar count (default 50)
    historical_init_bars: int

    # Number of most recent bars to include in each real-time update
    # Typically 1 (just the current bar) but can be more for additional context
    # Type: Integer representing bar count (default 1)
    realtime_update_bars: int

    # Flag controlling when updates are sent from Sierra Chart
    # True: updates only when bars close (stable, less frequent)
    # False: updates on every tick (real-time, high frequency)
    # Type: Boolean (default True)
    on_bar_close: bool

    # Unique identifier for this subscription assigned by Sierra Chart
    # Used to match incoming updates from the response queue to this subscription
    # Multiple subscriptions can coexist with different IDs
    # Type: Integer subscription ID
    chart_data_id: int

    def __init__(
        self,
        # Optional pre-existing SCBridge instance for connection reuse
        # If None, a new bridge will be created automatically
        # Useful when managing multiple subscriptions with a single connection
        bridge: Optional[SCBridge] = None,
        # Optional list of column names to drop from the final DataFrame
        # If None, defaults to ['IsBarClosed'] which is a helper column
        # Allows customization of which columns are excluded from results
        columns_to_drop: Optional[List[str]] = None,
        # Number of historical bars to receive initially when subscription starts
        # Default 50 provides reasonable context without overwhelming the system
        # Larger values give more history but slower initial load
        historical_init_bars: int = 50,
        # Number of bars to include in each real-time update
        # Default 1 sends only the most recent bar (typical use case)
        # Can be increased if you need multiple recent bars for calculations
        realtime_update_bars: int = 1,
        # Whether updates are sent only when bars close (True) or on every tick (False)
        # Default True reduces noise and provides stable, complete bar data
        # Set False for tick-by-tick updates in high-frequency trading
        on_bar_close: bool = True
    ) -> None:
        """
        Initialize the SubscribeToVbpChartData instance and start the subscription.

        Sets up the Sierra Chart bridge connection, configures subscription parameters,
        and immediately establishes the subscription to begin receiving data. The initial
        response will contain historical_init_bars for context, followed by continuous
        real-time updates.

        Args:
            bridge (Optional[SCBridge], optional): Pre-existing SCBridge instance to reuse.
                If None, creates a new bridge connection automatically. Defaults to None.
                Useful for sharing a single bridge across multiple subscriptions.
            columns_to_drop (Optional[List[str]], optional): Column names to remove from
                the processed DataFrame. If None, defaults to ['IsBarClosed']. Defaults to None.
                The IsBarClosed column indicates if a bar is complete (True) or forming (False).
            historical_init_bars (int, optional): Number of historical bars in initial response.
                Defaults to 50. Provides context for the real-time data stream.
                More bars = better context but slower initialization.
            realtime_update_bars (int, optional): Number of bars in each update.
                Defaults to 1. Usually set to 1 (latest bar only) for efficiency.
                Increase if your analysis needs multiple recent bars.
            on_bar_close (bool, optional): Whether updates occur only on bar close.
                Defaults to True. True = updates when bars complete (stable, recommended).
                False = updates on every tick (high frequency, for advanced use).

        Returns:
            None

        Example:
            >>> # Default settings (50 historical bars, updates on bar close)
            >>> sub1 = SubscribeToVbpChartData()
            >>>
            >>> # Custom settings for more history
            >>> sub2 = SubscribeToVbpChartData(historical_init_bars=200)
            >>>
            >>> # Tick-by-tick updates for HFT
            >>> sub3 = SubscribeToVbpChartData(on_bar_close=False)
            >>>
            >>> # Shared bridge for multiple subscriptions
            >>> shared_bridge = SCBridge()
            >>> sub4 = SubscribeToVbpChartData(bridge=shared_bridge)
            >>> sub5 = SubscribeToVbpChartData(bridge=shared_bridge)

        Note:
            - Subscription starts IMMEDIATELY in __init__ (not lazy)
            - First call to get_subscribed_vbp_chart_data() returns initial historical bars
            - Subsequent calls return incremental updates as they arrive
            - Must call stop_bridge() to terminate subscription and clean up
        """
        # Log a debug message to track subscription initialization for troubleshooting
        # Helps identify when subscriptions start and how many are active
        logger.debug("Initializing SubscribeToVbpChartData class")

        # Assign the bridge parameter or create a new SCBridge if none was provided
        # The ternary expression checks if bridge is not None and reuses it
        # Falls back to instantiating a fresh bridge when no shared instance exists
        # This pattern enables both standalone usage and connection sharing scenarios
        self.bridge = bridge if bridge is not None else SCBridge()

        # Assign the columns_to_drop parameter or use default list with 'IsBarClosed'
        # IsBarClosed indicates whether a bar is complete (True) or still forming (False)
        # Omit it by default because downstream analytics rarely need this helper flag
        self.columns_to_drop = (
            columns_to_drop if columns_to_drop is not None else ['IsBarClosed']
        )

        # Store the number of historical bars to include in the initial subscription response
        # This provides historical context when the subscription first starts
        # More bars give better context but slower initial response
        self.historical_init_bars = historical_init_bars

        # Store the number of bars to include in each real-time update
        # Typically 1 (just the latest bar) but can be more for rolling window analysis
        # Each update will contain this many bars of data
        self.realtime_update_bars = realtime_update_bars

        # Store the on_bar_close flag that controls update frequency
        # True: updates sent only when bars complete (stable, less frequent)
        # False: updates sent on every tick (real-time, high frequency)
        self.on_bar_close = on_bar_close

    # Establish the subscription to Sierra Chart and store the returned subscription ID
    # This ID is used to identify which updates belong to this subscription
    # The subscribe_to_vbp_chart_data() method sends the subscription request
    # Sierra Chart responds with a unique ID that we save for filtering updates
        self.chart_data_id = self.subscribe_to_vbp_chart_data()

    def subscribe_to_vbp_chart_data(self) -> int:
        """
        Establish a subscription to VBP chart data from Sierra Chart.

        This method sends a subscription request to Sierra Chart, specifying which data
        fields, indicators, and parameters to include in the continuous data stream.
        Sierra Chart responds with a unique subscription ID that identifies this stream.

        The subscription includes:
        - Base OHLCV data (Open, High, Low, Close, Volume)
        - Volume by Price distribution for each bar
        - Study indicators (Relative Volume, Daily OHLC)
        - Initial historical bars for context
        - Configuration for update frequency

        Unlike get_chart_data() which is one-time, this establishes a persistent stream
        that continues until explicitly stopped with stop_bridge().

        Returns:
            int: Unique subscription ID assigned by Sierra Chart. This ID is used to
                identify and filter updates from the response queue. Multiple subscriptions
                can coexist with different IDs.

        Raises:
            ConnectionError: If bridge cannot connect to Sierra Chart.
            ValueError: If subscription request fails or returns invalid ID.

        Example:
            >>> subscriber = SubscribeToVbpChartData.__new__(SubscribeToVbpChartData)
            >>> subscriber.bridge = SCBridge()
            >>> subscriber.historical_init_bars = 50
            >>> subscriber.realtime_update_bars = 1
            >>> subscriber.on_bar_close = True
            >>> sub_id = subscriber.subscribe_to_vbp_chart_data()
            >>> print(f"Subscription ID: {sub_id}")

        Note:
            - This method is called automatically by __init__
            - The subscription starts immediately and begins queuing data
            - Subscription remains active until stop_bridge() is called
            - The returned ID must be stored to retrieve correct updates
        """
        # Define the base price/volume data fields to retrieve from Sierra Chart
        # These are the standard OHLCV (Open, High, Low, Close, Volume) fields
        # Using Sierra Chart's constant definitions ensures API compatibility
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

        # Define which Sierra Chart study indicators and subgraphs to include in updates
        # Each SubgraphQuery specifies a study_id and which of its subgraphs to retrieve
        # Studies are pre-configured indicators in Sierra Chart that calculate derived values
        sg_to_fetch = [
            # Study ID 6: Relative Volume (RVOL) - compares current volume to average
            # Subgraph 1 returns the RVOL value (ratio of current volume to typical volume)
            # Used to identify abnormal volume activity and potential breakouts
            SubgraphQuery(study_id=6, subgraphs=[1]),

            # Study ID 4: Daily High, Low, Open levels
            # Subgraph 1: Today's opening price (first trade of the session)
            # Subgraph 2: Today's highest price (intraday high)
            # Subgraph 3: Today's lowest price (intraday low)
            # Used for intraday support/resistance and range analysis
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

        # Send the subscription request to Sierra Chart via the bridge
        # This establishes a persistent data stream and returns a unique subscription ID
        # The subscription remains active until explicitly stopped
        return self.bridge.subscribe_to_chart_data(
            # Unique key identifier for this subscription stream
            # Used by the bridge to track and manage multiple concurrent subscriptions
            # Can be any string, typically describes the chart or timeframe
            key='15minkey',

            # Enable Volume by Price data in all updates
            # This adds the VolumeByPrice column with nested price distribution data
            # Without this flag, only standard OHLCV data would be included
            include_volume_by_price=True,

            # Specify how many historical bars to include in the initial response
            # Uses the value set in __init__ (default 50)
            # Provides context before real-time updates begin
            # First update will contain this many historical bars
            historical_init_bars=self.historical_init_bars,

            # Specify how many bars to include in each real-time update
            # Uses the value set in __init__ (default 1)
            # Typically 1 (just the latest bar) for efficiency
            # Can be increased if your analysis needs multiple recent bars
            realtime_update_bars=self.realtime_update_bars,

            # Pass the list of base data fields (OHLCV) to include in updates
            # These are the fundamental price/volume series for each bar
            base_data=price_data_to_fetch,

            # Pass the list of study subgraph queries to include in updates
            # These add calculated indicators (RVOL, Daily OHLC)
            sg_data=sg_to_fetch,

            # Control when updates are sent from Sierra Chart
            # Uses the value set in __init__ (default True)
            # True: updates only when bars close (stable, recommended)
            # False: updates on every tick (high frequency, for advanced use)
            on_bar_close=self.on_bar_close
        )

    def process_vbp_chart_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process and flatten raw VBP chart data from subscription updates.

        This method transforms the nested VolumeByPrice structure from Sierra Chart
        subscription updates into a flat, tabular format suitable for analysis. Each
        bar's VBP data (a list of price levels with volume distributions) is exploded
        into separate rows, creating multiple rows per time bar.

        This is identical to the processing in get_vbp_chart_data module, ensuring
        consistent data structure whether fetching once or subscribing continuously.

        The processing pipeline:
        1. Extract nested VBP lists and convert to individual DataFrames
        2. Concatenate all VBP data with DateTime keys
        3. Sort by time and price for consistent ordering
        4. Join VBP data back with OHLCV and indicator data
        5. Clean up helper columns and normalize column names

        Args:
            df (pd.DataFrame): Raw chart data from subscription update containing
                nested VolumeByPrice column. Expected to have DateTime index and
                columns: Open, High, Low, Last, Volume, VolumeByPrice, and study data.

        Returns:
            pd.DataFrame: Processed DataFrame with flattened VBP structure:
                - DateTime index (may have duplicate timestamps for different price levels)
                - Price, BidVol, AskVol, TotalVolume, NumOfTrades columns (from VBP)
                - Original OHLCV columns (Open, High, Low, Close, Volume)
                - Indicator columns (RVOL, TodayOpen, TodayHigh, TodayLow)
                - Sorted by DateTime then Price for consistent structure

        Raises:
            KeyError: If expected columns (VolumeByPrice, study columns) are missing.
            ValueError: If VolumeByPrice contains invalid nested structure.

        Example:
            >>> subscriber = SubscribeToVbpChartData()
            >>> raw_update = subscriber.bridge.get_response_queue().get()
            >>> raw_df = raw_update.as_df()
            >>> processed_df = subscriber.process_vbp_chart_data(raw_df)
            >>> print(processed_df[['Price', 'TotalVolume', 'Close']].tail())

        Note:
            - Output DataFrame has MORE rows than input (one row per price level per bar)
            - DateTime index will have repeated values (multiple price levels per time)
            - Column names are standardized for consistency with project conventions
            - This method is called internally by get_subscribed_vbp_chart_data()
        """

        # Define nested helper to convert a single bar's VBP data into a DataFrame
        # Called for each bar's VolumeByPrice list to create per-bar DataFrames
        # Input format: [Price, BidVol, AskVol, TotalVolume, NumOfTrades] for each price level
        def vbp_to_df(vbp_data: list[list[Any]]) -> pd.DataFrame:
            """
            Convert a single bar's VolumeByPrice nested list into a tabular DataFrame.

            Each bar from Sierra Chart contains a VolumeByPrice field with nested data
            showing how volume was distributed across different price levels. This helper
            flattens that nested structure into a DataFrame with named columns.

            Args:
                vbp_data (list[list[Any]]): Nested list of VBP data from one bar.
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
            return pd.DataFrame(vbp_data, columns=columns)

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
        # Result: all price levels for first bar, then all for second bar, etc.
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
        combined_df.rename(
            columns={
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
            },
            inplace=True,
        )

        # Return the fully processed, flattened, and normalized DataFrame
        # Ready for analysis, visualization, and feature engineering
        # Contains integrated price data, volume distribution, and indicators
        return combined_df

    def get_subscribed_vbp_chart_data(self) -> pd.DataFrame:
        """
        Retrieve the next available chart data update from the subscription queue.

        This method blocks until a new update is available from the Sierra Chart
        subscription, then retrieves it from the response queue, processes it into
        analysis-ready format, and returns it. This is the main method for consuming
        the subscription data stream.

        The method operates in a blocking loop:
        1. Access the bridge's response queue
        2. Wait for next response (blocks if queue is empty)
        3. Check if response matches this subscription's ID
        4. If not a match, discard and continue waiting
        5. If match, convert to DataFrame and process
        6. Return processed data to caller

        This blocking behavior is intentional - it ensures your application processes
        updates in order without missing data or overwhelming the system with polling.

        Returns:
            pd.DataFrame: Processed chart data update with:
                - DateTime index (may have duplicates for multiple price levels)
                - OHLCV columns (Open, High, Low, Close, Volume)
                - VBP columns (Price, BidVol, AskVol, TotalVolume, NumOfTrades)
                - Indicator columns (RVOL, TodayOpen, TodayHigh, TodayLow)
                - Sorted by DateTime then Price

                First call returns historical_init_bars for context.
                Subsequent calls return realtime_update_bars (typically 1 bar).

        Raises:
            KeyboardInterrupt: If user interrupts with Ctrl+C (propagates up).
            Exception: If queue retrieval or processing fails.

        Example:
            >>> subscriber = SubscribeToVbpChartData()
            >>>
            >>> # Process first update (historical context)
            >>> initial_data = subscriber.get_subscribed_vbp_chart_data()
            >>> print(f"Initial bars: {len(initial_data)}")
            >>>
            >>> # Process real-time updates in a loop
            >>> while True:
            ...     update = subscriber.get_subscribed_vbp_chart_data()
            ...     print(f"New bar close: {update['Close'].iloc[-1]}")
            ...     print(f"RVOL: {update['RVOL'].iloc[-1]}")
            ...
            ...     # Your trading logic here
            ...     if update['RVOL'].iloc[-1] > 2.0:
            ...         print("High volume alert!")

        Note:
            - This method BLOCKS until new data arrives
            - First call returns historical bars (context)
            - Subsequent calls return incremental updates
            - Use in a loop for continuous processing
            - Wrap in try/except to handle KeyboardInterrupt gracefully
            - Multiple subscriptions with same bridge work correctly (filtered by ID)
        """
        # Access the response queue from the SCBridge instance
        # This queue receives all incoming messages from Sierra Chart
        # Multiple subscriptions share the same queue, so filtering by ID is necessary
        # Returns: Queue object that can be polled with .get()
        response_queue = self.bridge.get_response_queue()

        # Enter an infinite loop to continuously check for new subscription updates
        # This loop runs until a matching response is found and processed
        # Each iteration checks one response from the queue
        while True:
            # Retrieve the next response from the queue (blocks if queue is empty)
            # This is a blocking call - execution waits here until data is available
            # Returns: Response object containing chart data and metadata
            # The blocking behavior prevents CPU waste from busy polling
            response = response_queue.get()

            # Check if this response belongs to our subscription by comparing IDs
            # The request_id in the response must match our stored chart_data_id
            # This filtering is essential when multiple subscriptions share the bridge
            # If IDs don't match, this response is for a different subscription
            if response.request_id != self.chart_data_id:
                # Skip this response and continue to the next iteration
                # This response belongs to a different subscription, so we ignore it
                # Continue loops back to response_queue.get() to wait for the next response
                continue

            # We found a matching response! Convert it from Sierra Chart format to DataFrame
            # The as_df() method transforms the response object into a pandas DataFrame
            # Returns: DataFrame with DateTime index and nested VolumeByPrice column
            # This is still in raw format and needs processing
            vbp_chart_data_df = response.as_df()

            # Process the raw DataFrame into analysis-ready format
            # This flattens the nested VBP structure, normalizes columns, and cleans up
            # Calls process_vbp_chart_data() method to handle the transformation
            # Returns: Processed DataFrame ready for analysis and trading logic
            processed_chart_data = self.process_vbp_chart_data(vbp_chart_data_df)

            # Return the processed data to the caller
            # This exits the while loop and the method
            # The caller receives clean, structured data ready to use
            return processed_chart_data

    def stop_bridge(self) -> None:
        """
        Stop and clean up the Sierra Chart subscription and bridge connection.

        This method properly terminates the subscription, closes the connection to
        Sierra Chart, and releases resources. Should be called when done receiving
        updates to ensure graceful shutdown and prevent resource leaks.

        Best practice is to call this in a finally block or exception handler to
        ensure cleanup happens even if errors occur during data processing.

        Returns:
            None

        Example:
            >>> # Basic usage
            >>> subscriber = SubscribeToVbpChartData()
            >>> df = subscriber.get_subscribed_vbp_chart_data()
            >>> # ... process data ...
            >>> subscriber.stop_bridge()
            >>>
            >>> # With error handling (recommended)
            >>> subscriber = SubscribeToVbpChartData()
            >>> try:
            ...     while True:
            ...         df = subscriber.get_subscribed_vbp_chart_data()
            ...         # ... process data ...
            >>> except KeyboardInterrupt:
            ...     print("Stopping subscription...")
            >>> finally:
            ...     subscriber.stop_bridge()
            >>>
            >>> # Context manager pattern (if implemented)
            >>> with SubscribeToVbpChartData() as subscriber:
            ...     while True:
            ...         df = subscriber.get_subscribed_vbp_chart_data()

        Note:
            - Always call this method when done to prevent resource leaks
            - After calling stop_bridge(), the subscription cannot be reused
            - If bridge was passed to __init__, it will be stopped for all users
            - Safe to call multiple times (subsequent calls have no effect)
            - Subscription stops immediately, no more data will be queued
        """
        # Invoke the bridge's stop() method to terminate the subscription and connection
        # This performs several cleanup operations:
        # - Unsubscribes from all active data streams
        # - Closes the WebSocket/TCP connection to Sierra Chart
        # - Releases system resources (sockets, memory buffers)
        # - Clears the response queue
        # Ensures graceful shutdown and prevents resource leaks
        self.bridge.stop()

# # Usage example
# if __name__ == '__main__':
#     subscribe_to_chart_data = SubscribeToVbpChartData()

#     try:
#         # Continuously fetch and process chart data
#         while True:
#             vbp_chart_data_df = subscribe_to_chart_data.get_subscribed_vbp_chart_data()
#             if vbp_chart_data_df is not None:
#                 logger.info(f"VBP chart data (last 10 rows):\n{vbp_chart_data_df.tail(10)}")
#                 print(vbp_chart_data_df.tail(30))

#     except KeyboardInterrupt:
#         logger.info("Stopping subscription and exiting...")
#     finally:
#         subscribe_to_chart_data.stop_bridge()
