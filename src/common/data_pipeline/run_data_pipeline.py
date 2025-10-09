"""
Data Pipeline Runner Module

This module provides the main pipeline orchestration for processing financial market data
within the Project Chimera trading system. It handles both live data streams and file-based
data processing workflows.

Classes:
    DataPipelineRunner: Main orchestrator for data processing pipelines

Dependencies:
    pandas: For DataFrame operations
    logging: For system logging and monitoring

Author: Roy Williams
Version: 1.0.0
"""

# System Libraries
import logging
from typing import Dict, Any, Optional
import pandas as pd

# Local imports
from .dataframe_processor import DataFrameProcessor


class DataPipelineRunner:
    """
    Main orchestrator for data processing pipelines.
    
    This class handles the initialization and execution of data processing workflows,
    supporting both live data streams and file-based data sources. It provides
    comprehensive logging and error handling for robust pipeline execution.
    
    Attributes:
        config (Dict[str, Any]): Configuration dictionary containing pipeline parameters
        file_path (Optional[str]): Path to data file if using file-based processing
        df (Optional[pd.DataFrame]): DataFrame for live data processing
        data_source (Optional[str]): Source type indicator ('live' or 'file')
        processor (Optional[DataFrameProcessor]): Processor instance for file-based data
        logger (logging.Logger): Logger instance for this class
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize the DataPipelineRunner with configuration.
        
        Args:
            config (Dict[str, Any]): Configuration dictionary that must contain either
                'file_path' for file-based processing or 'df' for live data processing.
                Supported keys:
                - file_path (str, optional): Path to CSV file for processing
                - df (pd.DataFrame, optional): DataFrame for live data processing
        
        Raises:
            TypeError: If config is not a dictionary
            
        Example:
            >>> config = {'file_path': 'data/market_data.csv'}
            >>> pipeline = DataPipelineRunner(config)
        """
        if not isinstance(config, dict):
            raise TypeError("Config must be a dictionary")

        self.config = config
        self.file_path: Optional[str] = config.get('file_path')
        self.df: Optional[pd.DataFrame] = config.get('df')
        self.data_source: Optional[str] = None
        self.processor: Optional[DataFrameProcessor] = None

        # Set up module-specific logging
        self.logger = logging.getLogger(__name__)
        self.logger.info("DataPipelineRunner initialized with config keys: %s", list(config.keys()))
        self.logger.debug("Initializing DataPipelineRunner instance at memory address: %s", id(self))

    def process_data(self) -> None:
        """
        Process data from the configured source.
        
        This method handles data processing from either a live DataFrame or a file path.
        It automatically detects the data source type and applies the appropriate
        processing workflow.
        
        Raises:
            ValueError: If neither DataFrame nor file_path is provided in configuration
            FileNotFoundError: If file_path is provided but file doesn't exist
            Exception: For any other processing errors
            
        Side Effects:
            - Sets self.df to the processed DataFrame
            - Sets self.data_source to indicate the source type
            - Creates self.processor if using file-based processing
        """
        try:
            # Check if a DataFrame is provided directly
            if self.df is not None:
                self.logger.info("Using the provided DataFrame for data processing.")
                self.data_source = 'live'  # Mark the data source as live data stream

            # If no DataFrame is provided, try to load data from the file path
            elif self.file_path is not None:
                self.logger.info("Processing data from file path: %s", self.file_path)
                self.processor = DataFrameProcessor(self.file_path)
                self.df = self.processor.process_data()
                self.data_source = 'file'  # Mark the data source as file path

            # If neither a DataFrame nor a file path is provided, raise an error
            else:
                error_msg = "Data source not specified in configuration. Provide either 'df' or 'file_path'."
                self.logger.error(error_msg)
                raise ValueError(error_msg)

            # Log the successful data processing with the source and shape of the DataFrame
            if self.df is not None:
                self.logger.info(
                    "Data processing completed. Data source: %s, DataFrame shape: %s", 
                    self.data_source, 
                    self.df.shape
                )
            else:
                self.logger.warning("Data processing completed but DataFrame is None")

        except FileNotFoundError as e:
            self.logger.error("File not found at path '%s': %s", self.file_path, e)
            raise
        except Exception as e:
            self.logger.error("Error during data processing: %s", str(e))
            raise

    def run_pipeline(self) -> pd.DataFrame:
        """
        Execute the complete data pipeline.
        
        This method orchestrates the entire data processing workflow from
        initialization through completion, providing comprehensive logging
        and error handling.
        
        Returns:
            pd.DataFrame: The processed DataFrame ready for analysis
            
        Raises:
            ValueError: If data source configuration is invalid
            Exception: For any processing errors
            
        Example:
            >>> config = {'file_path': 'data/market_data.csv'}
            >>> pipeline = DataPipelineRunner(config)
            >>> processed_df = pipeline.run_pipeline()
        """
        try:
            self.logger.info("Starting data pipeline execution...")
            self.process_data()
            self.logger.info("Data pipeline execution completed successfully.")
            
            if self.df is None:
                raise ValueError("Pipeline completed but no data was processed")
                
            return self.df
            
        except Exception as e:
            self.logger.error("Data pipeline execution failed: %s", str(e))
            raise
    
    def get_data_info(self) -> Dict[str, Any]:
        """
        Get information about the processed data.
        
        Returns:
            Dict[str, Any]: Dictionary containing data information including:
                - data_source: Source type ('live' or 'file')
                - shape: Tuple of (rows, columns) if data is available
                - file_path: Path to source file if applicable
                - columns: List of column names if data is available
                
        Example:
            >>> info = pipeline.get_data_info()
            >>> print(f"Data shape: {info['shape']}")
        """
        info = {
            'data_source': self.data_source,
            'file_path': self.file_path,
            'shape': self.df.shape if self.df is not None else None,
            'columns': list(self.df.columns) if self.df is not None else None
        }
        return info
