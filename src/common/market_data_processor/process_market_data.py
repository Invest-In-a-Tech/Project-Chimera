"""
Market Data Processor Module.

This module provides functionality for processing market data with historical lookback
capabilities. It extracts current and historical market data from Volume by Price (VBP)
chart data to enable time-series analysis and feature engineering for trading strategies.

Classes:
    ProcessMarketData: Main class for processing market data with configurable lookback periods.

Example:
    >>> processor = ProcessMarketData(lookback_period=10)
    >>> market_data = processor.process_market_data(current_row, timestamp, vbp_df)
    >>> print(market_data['current']['Close'])
    >>> print(market_data['t-1']['Close'])  # Previous period's close
"""

import logging
from datetime import datetime
from typing import Union, Dict, Any

import pandas as pd


class ProcessMarketData:
    """
    Process market data with historical lookback capabilities.
    
    This class processes market data for a given timestamp and retrieves historical
    data for a specified number of previous periods. It organizes data into a
    structured dictionary format with keys for current period ('current') and
    historical periods ('t-1', 't-2', etc.).
    
    Attributes:
        market_data (Dict[str, Dict[str, Any]]): Dictionary storing processed market data
            with structure: {'current': {...}, 't-1': {...}, 't-2': {...}, ...}
        lookback_period (int): Number of previous periods to retrieve for historical context.
    
    Example:
        >>> processor = ProcessMarketData(lookback_period=5)
        >>> data = processor.process_market_data(row, timestamp, df)
        >>> current_close = data['current']['Close']
        >>> previous_close = data['t-1']['Close']
    """
    
    def __init__(self, lookback_period: int = 10) -> None:
        """
        Initialize the ProcessMarketData instance.
        
        Args:
            lookback_period (int, optional): Number of previous periods to look back.
                Defaults to 10. This determines how many historical data points
                will be retrieved and stored for analysis.
        
        Returns:
            None
        
        Example:
            >>> processor = ProcessMarketData(lookback_period=20)
        """
        # Initialize an empty dictionary to store processed market data
        # Structure: {'current': {...}, 't-1': {...}, 't-2': {...}, ...}
        self.market_data: Dict[str, Dict[str, Any]] = {}
        
        # Store the number of previous periods to look back
        # This controls how much historical data will be retrieved
        self.lookback_period: int = lookback_period
    
    def process_market_data(
        self,
        current_row: pd.Series,
        current_timestamp: Union[pd.Timestamp, datetime],
        vbp_chart_data_df: pd.DataFrame
    ) -> Dict[str, Dict[str, Any]]:
        """
        Process market data for the current row and timestamp with historical lookback.
        
        This method extracts current market data and retrieves historical data for
        the specified lookback period. It handles missing data gracefully and ensures
        data structure consistency. The returned dictionary contains the current
        period's data and up to `lookback_period` previous periods.
        
        Args:
            current_row (pd.Series): Current row of market data being processed.
                Should contain columns like Open, High, Low, Close, Volume, etc.
            current_timestamp (Union[pd.Timestamp, datetime]): Timestamp for the
                current data point. Used to locate the position in the time series
                and retrieve historical data.
            vbp_chart_data_df (pd.DataFrame): Volume by Price chart data DataFrame.
                Must have a datetime index and contain historical market data.
                
        Returns:
            Dict[str, Dict[str, Any]]: Nested dictionary containing processed market data:
                - 'current': Dict with current period's data, timestamp, and index
                - 't-1', 't-2', ..., 't-N': Dicts with historical periods' data
                Each period dict contains all columns from the original data plus
                a 'timestamp' key.
        
        Raises:
            IndexError: If current_timestamp is not found in the DataFrame index.
            KeyError: If expected columns are missing from the data.
        
        Example:
            >>> processor = ProcessMarketData(lookback_period=3)
            >>> row = df.loc['2025-01-15']
            >>> timestamp = pd.Timestamp('2025-01-15')
            >>> result = processor.process_market_data(row, timestamp, df)
            >>> print(result['current']['Close'])
            >>> print(result['t-1']['Close'])  # Previous day's close
            
        Note:
            - If fewer than `lookback_period` historical records exist, only available
              records will be included (e.g., at the start of the dataset).
            - Missing or invalid timestamps are logged as warnings and skipped.
            - Data structure is validated to ensure consistent pd.Series format.
        """
        # Extract all unique timestamps from the Volume by Price DataFrame index
        # This creates a sorted array of all available time points in the dataset
        # Returns: pd.Index (DatetimeIndex) containing unique timestamp values
        timestamps = vbp_chart_data_df.index.unique()
        
        # Locate the position (index) of the current timestamp in the timestamps array
        # get_indexer returns an array with the position; [0] extracts the single integer
        # This index is used to navigate backward in time for historical lookback
        # Returns: int representing the position of current_timestamp in the index
        current_index = timestamps.get_indexer([current_timestamp])[0]
        
        # Initialize the market_data dictionary with a 'current' key holding an empty dict
        # This resets any previous data and prepares the structure for new data
        # Structure: {'current': {}} will become {'current': {'Close': 100, ...}, ...}
        self.market_data = {'current': {}}
        
        # Iterate through all columns in the current row (e.g., Open, High, Low, Close, Volume)
        # and populate the 'current' dictionary with column names as keys and values as data
        # This creates a flat dictionary of all current market data fields
        for column in current_row.index:
            # Store each column's value from current_row into the 'current' sub-dictionary
            # Example: self.market_data['current']['Close'] = 150.25
            self.market_data['current'][column] = current_row[column]
        
        # Add metadata: store the actual timestamp of the current data point
        # This helps with debugging and provides temporal context for the data
        self.market_data['current']['timestamp'] = current_timestamp
        
        # Add metadata: store the integer index position in the time series
        # Useful for calculations that depend on data point position (e.g., progress tracking)
        self.market_data['current']['index'] = current_index
        
        # Calculate the actual number of lookback periods available
        # Use the minimum of requested lookback_period and current_index to avoid
        # attempting to access data before the start of the dataset
        # Example: if current_index=3 and lookback_period=10, available_lookbacks=3
        available_lookbacks = min(self.lookback_period, current_index)
        
        # Iterate through each historical period from t-1 to t-N (where N=available_lookbacks)
        # range(1, available_lookbacks + 1) generates [1, 2, 3, ..., available_lookbacks]
        # i represents how many periods back we're looking (1=previous period, 2=two periods back, etc.)
        for i in range(1, available_lookbacks + 1):
            # Calculate the timestamp for the i-th previous period
            # Subtract i from current_index to go backward in time
            # Example: if current_index=10 and i=2, retrieves timestamp at position 8
            previous_timestamp = timestamps[current_index - i]
            
            # Verify that the calculated previous_timestamp exists in the DataFrame index
            # This check handles cases where data might be missing or index is sparse
            if previous_timestamp not in vbp_chart_data_df.index:
                # Log a warning using lazy % formatting (Pylint best practice)
                # This avoids string formatting overhead if logging level filters out warnings
                logging.warning("Missing data for timestamp: %s", previous_timestamp)
                # Skip to the next iteration without adding data for this period
                continue
            
            # Retrieve the data row corresponding to the previous timestamp
            # .loc returns either a Series (single row) or DataFrame (multiple rows with same index)
            # Returns: pd.Series or pd.DataFrame depending on index uniqueness
            previous_data = vbp_chart_data_df.loc[previous_timestamp]
            
            # Validate and normalize the data structure to ensure it's a pandas Series
            # This handles the case where .loc might return a DataFrame if timestamp isn't unique
            if isinstance(previous_data, pd.DataFrame) and not previous_data.empty:
                # If it's a DataFrame with rows, take the first row to get a Series
                # .iloc[0] extracts the first row as a Series
                previous_data = previous_data.iloc[0]
            elif isinstance(previous_data, pd.Series):
                # If it's already a Series, no action needed - this is the expected case
                # The pass statement explicitly shows we've handled this case
                pass
            else:
                # If data is neither a valid DataFrame nor Series (e.g., None or scalar),
                # log a warning using lazy % formatting and skip this period
                logging.warning("Invalid data structure for timestamp: %s", previous_timestamp)
                # Continue to next iteration without adding data for this period
                continue
            
            # Initialize a new dictionary entry for this historical period
            # Key format: 't-1', 't-2', 't-3', etc. (t-i where i is periods back)
            # Start with just the timestamp to establish the time reference
            self.market_data[f't-{i}'] = {'timestamp': previous_timestamp}
            
            # Iterate through all columns in the previous_data Series
            # and populate the historical period dictionary with all market data fields
            for column in previous_data.index:
                # Store each column's historical value in the period's dictionary
                # Example: self.market_data['t-1']['Close'] = 148.50
                self.market_data[f't-{i}'][column] = previous_data[column]
        
        # Return the fully populated market_data dictionary containing current and historical data
        # This dictionary can now be used for feature engineering, analysis, or ML model input
        return self.market_data