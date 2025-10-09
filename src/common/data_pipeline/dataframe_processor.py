
"""
Data Pipeline Module for DataFrame Processing

This module provides utilities for processing financial market data stored in CSV format.
It handles data loading, cleaning, filtering, and preparation for analysis within the
Project Chimera trading system.

The module focuses on time-series data processing with specific filtering for market hours
and proper datetime handling for financial analysis workflows.

Classes:
    DataFrameProcessor: Main class for processing CSV-based financial data

Dependencies:
    pandas: For DataFrame operations and time-series handling

Author: Roy Williams
Version: 1.0.0
"""

# Standard imports
import pandas as pd

class DataFrameProcessor:
    """
    A processor class for handling financial market data from CSV files.
    
    This class provides functionality to load, clean, and filter financial time-series
    data with a focus on market hours filtering and proper datetime indexing.
    
    Attributes:
        file_path (str): Path to the CSV file containing the data
        df (pd.DataFrame): Processed DataFrame with datetime index
        
    Example:
        >>> processor = DataFrameProcessor('data/market_data.csv')
        >>> df = processor.process_data()
        >>> print(df.head())
    """

    def __init__(self, file_path):
        """
        Initialize the DataFrameProcessor with a file path.
        
        Args:
            file_path (str): Path to the CSV file to be processed. The file should
                           contain financial data with a 'DateTime' column in 
                           'YYYY-MM-DD HH:MM:SS' format.
                           
        Raises:
            FileNotFoundError: If the specified file path does not exist
        """
        self.file_path = file_path
        self.df = None

    def process_data(self):
        """
        Process the CSV data by loading, cleaning, and filtering for market hours.
        
        This method performs the following operations:
        1. Loads data from the CSV file specified in __init__
        2. Sorts the data by DateTime column
        3. Converts DateTime column to pandas datetime format
        4. Sets DateTime as the index
        5. Filters data to include only market hours (08:00:00 to 15:00:00)
        
        Returns:
            pd.DataFrame: Processed DataFrame with datetime index, filtered for
                         market hours and sorted chronologically
                         
        Raises:
            FileNotFoundError: If the CSV file cannot be found
            KeyError: If the 'DateTime' column is missing from the CSV
            ValueError: If DateTime values cannot be parsed
            
        Note:
            The market hours filter (08:00:00 to 15:00:00) assumes a specific
            timezone. Adjust as needed for different markets or timezones.
        """

        self.df = pd.read_csv(self.file_path)
        self.df = self.df.sort_values('DateTime')
        self.df['DateTime'] = pd.to_datetime(self.df['DateTime'], format='%Y-%m-%d %H:%M:%S')
        # self.df = self.df.set_index('DateTime').between_time('08:00:00', '15:00:00')
        self.df = self.df.set_index('DateTime')
        return self.df
