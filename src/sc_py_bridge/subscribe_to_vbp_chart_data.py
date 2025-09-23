"""
Volume by Price (VBP) Chart Data Subscription Module.

This module provides functionality to subscribe to real-time and historical Volume by Price (VBP)
chart data updates from Sierra Chart using the trade29 SC bridge. The module contains the
SubscribeToVbpChartData class which handles the connection to Sierra Chart, starting a subscription,
receiving incremental updates, and processing VBP data into structured pandas DataFrames.

Classes:
    SubscribeToVbpChartData: Main class for subscribing to and processing VBP chart data updates.

Dependencies:
    - trade29.sc: Sierra Chart bridge library for data communication
    - pandas: Data manipulation and analysis library
    - typing: Type hinting support

Example:
    Basic usage of the SubscribeToVbpChartData class:

    ```python
    from sc_py_bridge.subscribe_to_vbp_chart_data import SubscribeToVbpChartData

    # Initialize with default settings
    subscriber = SubscribeToVbpChartData()

    try:
        # Fetch the next available update and process it
        vbp_update_df = subscriber.get_subscribed_vbp_chart_data()
        print(vbp_update_df.tail(10))
    finally:
        # Clean up connection
        subscriber.stop_bridge()
    ```

Author: Roy Williams
Version: 1.0.0
"""

# Postpone evaluation of type annotations to avoid forward-reference issues
from __future__ import annotations

# Standard library imports
# Bring in common typing primitives used for annotations
from typing import Any, List, Optional

import logging

# Third-party imports
# Import pandas for DataFrame manipulation and analysis
import pandas as pd

# Local imports
# Import Sierra Chart bridge types and constants for requesting data
from trade29.sc import SCBridge, SubgraphQuery, constants

# Module-level logger
logger = logging.getLogger(__name__)

class SubscribeToVbpChartData:
    """
    Class to subscribe to Volume by Price (VBP) chart data updates from Sierra Chart.
    """
    # SCBridge instance used to communicate with Sierra Chart
    bridge: SCBridge

    # Columns to drop from the final DataFrame (cleanup step)
    columns_to_drop: List[str]

    # Number of historical bars to request from Sierra Chart
    historical_init_bars: int

    # Number of real-time bars to request from Sierra Chart
    realtime_update_bars: int

    # Whether to set onbar close true or false
    on_bar_close: bool

    # Subscription ID for the VBP chart data
    chart_data_id: int

    def __init__(
        self,

        # Optional externally-managed SCBridge instance; if not provided, create one
        bridge: Optional[SCBridge] = None,

        # Optional list of column names to remove from the final DataFrame
        columns_to_drop: Optional[List[str]] = None,

        # How many historical bars to include in the request (default 50)
        historical_init_bars: int = 50,

        # How many real-time bars to include in the request (default 1)
        realtime_update_bars: int = 1,

        # Whether to set onbar close true or false (default True)
        on_bar_close: bool = True
    ) -> None:
        """
        Initialize the SubscribeToVbpChartData instance.

        Args:
            bridge (Optional[SCBridge]): An instance of SCBridge to communicate with Sierra Chart.
            columns_to_drop (Optional[List[str]]): Columns to remove from the resulting DataFrame.
            historical_init_bars (int): Number of historical bars to request initially.
            realtime_update_bars (int): Number of real-time bars to include in updates.
            on_bar_close (bool): Whether updates are sent only when a bar closes.
        """

        # Startup message for tracing object lifecycle
        logger.debug("Initializing SubscribeToVbpChartData class")

        # Use provided bridge or create a new SCBridge to talk to Sierra Chart
        self.bridge = bridge if bridge is not None else SCBridge()

        # Use provided drop-list or default to removing the 'IsBarClosed' helper column
        self.columns_to_drop = columns_to_drop if columns_to_drop is not None else ['IsBarClosed']

        # Store the number of bars to request for later calls
        self.historical_init_bars = historical_init_bars

        # Store the number of real-time bars to request for later calls
        self.realtime_update_bars = realtime_update_bars

        # Store the on_bar_close flag for later calls
        self.on_bar_close = on_bar_close

        # Subscribe & store request_id
        self.chart_data_id = self.subscribe_to_vbp_chart_data()

    def subscribe_to_vbp_chart_data(self) -> int:
        """
        Subscribes to Volume by Price (VBP) chart data from Sierra Chart.

        Returns:
            int: The subscription ID for the VBP chart data.
        """
        # Base price/volume fields we want included in the chart data response
        price_data_to_fetch = [
            constants.SCBaseData.SC_OPEN,
            constants.SCBaseData.SC_HIGH,
            constants.SCBaseData.SC_LOW,
            constants.SCBaseData.SC_LAST,
            constants.SCBaseData.SC_VOLUME
        ]

        # Subgraph studies to include (study id and subgraph indices)
        sg_to_fetch = [
            SubgraphQuery(study_id=6, subgraphs=[1]),           # Relative Volume
            SubgraphQuery(study_id=4, subgraphs=[1, 2, 3])      # Daily High, Low, Open
        ]

        # Initiate subscription with SCBridge and store the subscription ID
        return self.bridge.subscribe_to_chart_data(
            key='1minkey',

            # Include Volume by Price data in the payload
            include_volume_by_price=True,

            # Request this many historical bars
            historical_init_bars=self.historical_init_bars,

           # Include this many real-time bars in each update
            realtime_update_bars=self.realtime_update_bars,

            # Include base data series (open/high/low/close/volume)
            base_data=price_data_to_fetch,

            # Include study subgraphs (e.g., RVOL, Today OHLC)
            sg_data=sg_to_fetch,

            # Whether to set onbar close true or false
            on_bar_close=self.on_bar_close
        )

    def process_vbp_chart_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Processes the raw VBP chart data DataFrame.

        Args:
            df (pd.DataFrame): The raw DataFrame of chart data.

        Returns:
            pd.DataFrame: The processed DataFrame with VBP data combined and DateTime set as index.
        """

        # Helper to convert a single bar's VolumeByPrice nested list into a tabular DataFrame
        def vbp_to_df(vbp_data: list[list[Any]]) -> pd.DataFrame:
            """
            Converts VolumeByPrice data into a DataFrame.

            Args:
                vbp_data (list): List of VBP data [Price, BidVol, AskVol, TotalVolume, NumOfTrades].

            Returns:
                pd.DataFrame: DataFrame containing VBP data.
            """
            # Define the schema for each VBP row
            columns = ['Price', 'BidVol', 'AskVol', 'TotalVolume', 'NumOfTrades']

            # Build and return the per-bar VBP DataFrame
            return pd.DataFrame(vbp_data, columns=columns)

        # Concatenate VBP DataFrames for each bar, using the parent row index (DateTime) as a key
        vbp_combined = pd.concat(
            [vbp_to_df(v) for v in df['VolumeByPrice']],
            keys=df.index
        )\
        .reset_index(level=1, drop=True)\
        .reset_index()

        # Rename the key column to 'DateTime' to reflect its meaning
        vbp_combined.rename(columns={'index': 'DateTime'}, inplace=True)

        # Sort rows first by bar time then by price within each bar
        vbp_combined = vbp_combined.sort_values(['DateTime', 'Price'])

        # Ensure DateTime column is a proper datetime dtype
        vbp_combined['DateTime'] = pd.to_datetime(vbp_combined['DateTime'])

        # Use DateTime as the index for easy time-based joins/queries
        vbp_combined.set_index('DateTime', inplace=True)

        # Bring the non-VBP columns back together with the exploded VBP rows
        combined_df = df.drop(columns=['VolumeByPrice']).join(vbp_combined, how='outer')

        # Remove helper columns if present (ignore if missing)
        combined_df.drop(columns=self.columns_to_drop, inplace=True, errors='ignore')
        # Normalize column names to common conventions used elsewhere in the project
        combined_df.rename(columns={
            'Last': 'Close',
            'ID6.SG1': 'RVOL',
            'ID4.SG1': 'TodayOpen',
            'ID4.SG2': 'TodayHigh',
            'ID4.SG3': 'TodayLow',
        }, inplace=True)

        # Return the fully processed DataFrame
        return combined_df

    def get_subscribed_vbp_chart_data(self) -> pd.DataFrame:
        """
        Fetches the latest chart data from the response queue, processes it, and returns it.

        Returns:
            pd.DataFrame: The processed DataFrame of chart data.
        """
        # Access the response queue from SCBridge to read incoming data
        response_queue = self.bridge.get_response_queue()

        # Continuously check the queue for new data related to the subscription
        while True:
            response = response_queue.get()

            # Only process data matching the subscription ID to avoid unrelated data
            if response.request_id != self.chart_data_id:
                # Skip processing non-matching responses in case of multiple subscriptions
                continue

            # Convert the response to a DataFrame for manipulation
            vbp_chart_data_df = response.as_df()

            # Process the chart data by sorting, renaming, and merging as needed
            processed_chart_data = self.process_vbp_chart_data(vbp_chart_data_df)

            return processed_chart_data



    def stop_bridge(self) -> None:
        """
        Stops the SCBridge object.

        This method ensures that the SCBridge connection is properly closed.
        """
        # Ask the bridge to terminate the connection/session cleanly
        self.bridge.stop()     

# # Usage example
# if __name__ == '__main__':
#     subscribe_to_chart_data = SubscribeToVbpChartData()

#     try:
#         # Continuously fetch and process chart data
#         while True:
#             vbp_chart_data_df = subscribe_to_chart_data.get_subscribed_vbp_chart_data()
#             if vbp_chart_data_df is not None:
#                 print(vbp_chart_data_df.tail(10))

#     except KeyboardInterrupt:
#         print("Stopping subscription and exiting...")
#     finally:
#         subscribe_to_chart_data.stop_bridge()
