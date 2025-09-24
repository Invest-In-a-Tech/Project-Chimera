"""
Volume by Price (VBP) Chart Data Fetcher Downloader Module.

This module allows downloading and saving Volume by Price (VBP) chart data
from Sierra Chart using the trade29 SC bridge. It utilizes the GetVbpData class.

In the near future, I will update this module to just call the GetVbpData class
from the sc_py_bridge package directly. That way, our dataframes remain consistent.

Author: Roy Williams
Version: 1.0.0
Last Updated: September 2025
"""

# Postpone evaluation of type annotations to avoid forward-reference issues
from __future__ import annotations

# Standard library imports
import os
# Bring in common typing primitives used for annotations
from typing import Any, List, Optional

import logging

# Third-party imports
# Import pandas for DataFrame manipulation and analysis
import pandas as pd
# Import Sierra Chart bridge types and constants for requesting data
from trade29.sc import SCBridge, SubgraphQuery, constants

# Configure pandas display options to show full DataFrame
pd.set_option('display.max_rows', None)  # Show all rows
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.width', None)  # Don't wrap columns
pd.set_option('display.max_colwidth', None)  # Show full column content

# Local imports
# (No local imports currently)

# Module-level logger
logger = logging.getLogger(__name__)

class GetVbpData:
    """
    Class to fetch Volume by Price (VBP) chart data using a Sierra Chart bridge.
    """
    # SCBridge instance used to communicate with Sierra Chart
    bridge: SCBridge

    # Columns to drop from the final DataFrame (cleanup step)
    columns_to_drop: List[str]

    # Number of historical bars to request from Sierra Chart
    historical_bars: int

    def __init__(
        self,

        # Optional externally-managed SCBridge instance; if not provided, create one
        bridge: Optional[SCBridge] = None,

        # Optional list of column names to remove from the final DataFrame
        columns_to_drop: Optional[List[str]] = None,

        # How many historical bars to include in the request (default 1000000)
        historical_bars: int = 1000000
    ) -> None:
        """
        Initialize the GetVbpData instance.

        Args:
            bridge (Optional[SCBridge]): An instance of SCBridge to communicate with Sierra Chart.
            columns_to_drop (Optional[List[str]]): Columns to remove from the resulting DataFrame.
            historical_bars (int): Number of historical bars to request.
        """
        # Startup message for tracing object lifecycle
        logger.debug("Initializing GetVbpData class")

        # Use provided bridge or create a new SCBridge to talk to Sierra Chart
        self.bridge = bridge if bridge is not None else SCBridge()

        # Use provided drop-list or default to removing the 'IsBarClosed' helper column
        self.columns_to_drop = columns_to_drop if columns_to_drop is not None else ['IsBarClosed']

        # Store the number of bars to request for later calls
        self.historical_bars = historical_bars

    def fetch_vbp_chart_data(self) -> pd.DataFrame:
        """
        Fetches Volume by Price (VBP) chart data from Sierra Chart.

        Returns:
            pd.DataFrame: A DataFrame containing the raw VBP chart data.
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
            SubgraphQuery(study_id=4, subgraphs=[1, 2, 3]),      # Daily High, Low, Open
            SubgraphQuery(study_id=9, subgraphs=[1, 2, 3, 4]),  # Large Tradess
        ]

        # Issue the request to Sierra Chart via the bridge with VBP enabled
        vbp_chart_data_response = self.bridge.get_chart_data(
            key='vbpKey',

            # Include Volume by Price data in the payload
            include_volume_by_price=True,

            # Request this many historical bars
            historical_bars=self.historical_bars,

            # Include the in-progress live bar for the latest interval
            include_live_bar=True,

            # Include base data series (open/high/low/close/volume)
            base_data=price_data_to_fetch,

            # Include study subgraphs (e.g., RVOL, Today OHLC)
            sg_data=sg_to_fetch
        )

        # Convert the response object into a pandas DataFrame and return
        return vbp_chart_data_response.as_df()

    def process_vbp_chart_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Processes the raw VBP chart data DataFrame.

        Args:
            df (pd.DataFrame): The raw DataFrame of chart data.

        Returns:
            pd.DataFrame: The processed DataFrame with VBP data combined and DateTime set as index.
        """

        # Helper to convert a single bar's VolumeByPrice nested list into a tabular DataFrame
        def vbp_to_df(vbp_rows: list[list[Any]]) -> pd.DataFrame:
            """
            Converts VolumeByPrice data into a DataFrame.

            Args:
                vbp_rows (list): List of VBP data [Price, BidVol, AskVol, TotalVolume, NumOfTrades].

            Returns:
                pd.DataFrame: DataFrame containing VBP data.
            """
            # Define the schema for each VBP row
            columns = ['Price', 'BidVol', 'AskVol', 'TotalVolume', 'NumOfTrades']
            # Build and return the per-bar VBP DataFrame
            return pd.DataFrame(vbp_rows, columns=columns)

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

        # Normalize column names to common conventions used elsewhere in the project.
        # In other words, map the bridge's default column names to more familiar ones.
        combined_df.rename(columns={
            'Last': 'Close',
            'ID6.SG1': 'RVOL',
            'ID4.SG1': 'TodayOpen',
            'ID4.SG2': 'TodayHigh',
            'ID4.SG3': 'TodayLow',
            'ID9.SG1': 'LTMaxVol',
            'ID9.SG2': 'LTTotalVol',
            'ID9.SG3': 'LTBidVol',
            'ID9.SG4': 'LTAskVol'
        }, inplace=True)

        # Return the fully processed DataFrame
        return combined_df

    def get_vbp_chart_data(self) -> pd.DataFrame:
        """
        Fetches and processes the VBP chart data.

        This method first fetches the raw VBP chart data using the `fetch_vbp_chart_data` method,
        then processes it using the `process_vbp_chart_data` method.

        Returns:
            pd.DataFrame: A DataFrame containing the processed VBP chart data.
        """
        # Fetch the raw chart data (including VBP) from Sierra Chart
        raw_df = self.fetch_vbp_chart_data()

        # Transform the raw data into a flattened, analysis-friendly DataFrame
        processed_df = self.process_vbp_chart_data(raw_df)

        # Return the processed dataset
        return processed_df


    def stop_bridge(self) -> None:
        """
        Stops the SCBridge object.

        This method ensures that the SCBridge connection is properly closed.
        """
        # Ask the bridge to terminate the connection/session cleanly
        self.bridge.stop()

if __name__ == "__main__":
    # Example usage when running this module as a script
    vbp_data = GetVbpData()
    result_df = vbp_data.get_vbp_chart_data()

    # Save the DataFrame to the specified directory using os.path
    output_dir = os.path.join('data', 'raw', 'dataframes')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'volume_by_price_15years2.csv')
    result_df.to_csv(output_path)
    # logger.info(f"Volume by Price data saved to: {output_path}")
    print(f"Volume by Price data saved to: {output_path}")
