"""
Volume by Price (VBP) Chart Data Fetcher Module.

This module provides functionality to fetch and process Volume by Price (VBP) chart data
from Sierra Chart using the trade29 SC bridge. The module contains the GetVbpData class
which handles the connection to Sierra Chart, data retrieval, and processing of VBP data
into structured pandas DataFrames.

Classes:
    GetVbpData: Main class for fetching and processing VBP chart data.

Dependencies:
    - trade29.sc: Sierra Chart bridge library for data communication
    - pandas: Data manipulation and analysis library
    - typing: Type hinting support

Example:
    Basic usage of the GetVbpData class:

    ```python
    from sc_py_bridge.get_vbp_chart_data import GetVbpData

    # Initialize with default settings
    vbp_fetcher = GetVbpData()

    # Fetch and process VBP data
    vbp_data = vbp_fetcher.get_vbp_chart_data()

    # Clean up connection
    vbp_fetcher.stop_bridge()
    ```

Author: Project Chimera
Version: 1.0.0
"""

from __future__ import annotations

# Standard library imports
from typing import Any, List, Optional

# Third-party imports
import pandas as pd

# Local imports
from trade29.sc import SCBridge, SubgraphQuery, constants

class GetVbpData:
    """
    Class to fetch Volume by Price (VBP) chart data using a Sierra Chart bridge.
    """

    bridge: SCBridge
    columns_to_drop: List[str]
    historical_bars: int

    def __init__(
        self,
        bridge: Optional[SCBridge] = None,
        columns_to_drop: Optional[List[str]] = None,
        historical_bars: int = 50
    ) -> None:
        """
        Initialize the GetVbpData instance.

        Args:
            bridge (Optional[SCBridge]): An instance of SCBridge to communicate with Sierra Chart.
            columns_to_drop (Optional[List[str]]): Columns to remove from the resulting DataFrame.
            historical_bars (int): Number of historical bars to request.
        """
        print("Initializing GetVbpData class")
        self.bridge = bridge if bridge is not None else SCBridge()
        self.columns_to_drop = columns_to_drop if columns_to_drop is not None else ['IsBarClosed']
        self.historical_bars = historical_bars

    def fetch_vbp_chart_data(self) -> pd.DataFrame:
        """
        Fetches Volume by Price (VBP) chart data from Sierra Chart.

        Returns:
            pd.DataFrame: A DataFrame containing the raw VBP chart data.
        """
        price_data_to_fetch = [
            constants.SCBaseData.SC_OPEN,
            constants.SCBaseData.SC_HIGH,
            constants.SCBaseData.SC_LOW,
            constants.SCBaseData.SC_LAST,
            constants.SCBaseData.SC_VOLUME
        ]

        sg_to_fetch = [
            SubgraphQuery(study_id=6, subgraphs=[1]),           # Relative Volume
            SubgraphQuery(study_id=4, subgraphs=[1, 2, 3])      # Daily High, Low, Open
        ]

        vbp_chart_data_response = self.bridge.get_chart_data(
            key='vbpKey',
            include_volume_by_price=True,
            historical_bars=self.historical_bars,
            include_live_bar=True,
            base_data=price_data_to_fetch,
            sg_data=sg_to_fetch
        )

        return vbp_chart_data_response.as_df()

    def process_vbp_chart_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Processes the raw VBP chart data DataFrame.

        Args:
            df (pd.DataFrame): The raw DataFrame of chart data.

        Returns:
            pd.DataFrame: The processed DataFrame with VBP data combined and DateTime set as index.
        """

        def vbp_to_df(vbp_data: list[list[Any]]) -> pd.DataFrame:
            """
            Converts VolumeByPrice data into a DataFrame.

            Args:
                vbp_data (list): List of VBP data [Price, BidVol, AskVol, TotalVolume, NumOfTrades].

            Returns:
                pd.DataFrame: DataFrame containing VBP data.
            """
            columns = ['Price', 'BidVol', 'AskVol', 'TotalVolume', 'NumOfTrades']
            return pd.DataFrame(vbp_data, columns=columns)

        vbp_combined = pd.concat(
            [vbp_to_df(v) for v in df['VolumeByPrice']],
            keys=df.index
        ).reset_index(level=1, drop=True).reset_index()

        vbp_combined.rename(columns={'index': 'DateTime'}, inplace=True)
        vbp_combined = vbp_combined.sort_values(['DateTime', 'Price'])
        vbp_combined['DateTime'] = pd.to_datetime(vbp_combined['DateTime'])
        vbp_combined.set_index('DateTime', inplace=True)

        combined_df = df.drop(columns=['VolumeByPrice']).join(vbp_combined, how='outer')
        combined_df.drop(columns=self.columns_to_drop, inplace=True, errors='ignore')
        combined_df.rename(columns={
            'Last': 'Close',
            'ID6.SG1': 'RVOL',
            'ID4.SG1': 'TodayOpen',
            'ID4.SG2': 'TodayHigh',
            'ID4.SG3': 'TodayLow',
        }, inplace=True)

        return combined_df

    def get_vbp_chart_data(self) -> pd.DataFrame:
        """
        Fetches and processes the VBP chart data.

        This method first fetches the raw VBP chart data using the `fetch_vbp_chart_data` method,
        then processes it using the `process_vbp_chart_data` method.

        Returns:
            pd.DataFrame: A DataFrame containing the processed VBP chart data.
        """
        vbp_chart_data_df = self.fetch_vbp_chart_data()
        processed_vbp_chart_data = self.process_vbp_chart_data(vbp_chart_data_df)
        return processed_vbp_chart_data


    def stop_bridge(self) -> None:
        """
        Stops the SCBridge object.

        This method ensures that the SCBridge connection is properly closed.
        """
        self.bridge.stop()