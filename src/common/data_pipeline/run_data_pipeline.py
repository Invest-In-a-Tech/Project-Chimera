"""
Data Pipeline Runner Module

This module provides the main pipeline orchestration for processing financial market data
within the Project Chimera trading system. It handles both live data streams and file-based
data processing workflows.

The pipeline supports two distinct modes:
1. Training Mode: File-based processing for ML model training and backtesting
2. Live Mode: Real-time data processing from Sierra Chart for live trading

Classes:
    DataPipelineRunner: Main orchestrator for data processing pipelines

Dependencies:
    pandas: For DataFrame operations
    logging: For system logging and monitoring

Author: Roy Williams
Version: 1.1.0
"""

# System Libraries
import logging
from typing import Dict, Any, Optional
from enum import Enum
import pandas as pd

# Local imports
from .dataframe_processor import DataFrameProcessor


class PipelineMode(Enum):
    """Pipeline operation modes."""
    TRAINING = "training"  # File-based processing for ML training
    LIVE = "live"         # Real-time processing from Sierra Chart
    AUTO = "auto"         # Auto-detect based on config


class DataPipelineRunner:
    """
    Main orchestrator for data processing pipelines.
    
    This class handles the initialization and execution of data processing workflows,
    supporting both training (file-based) and live (Sierra Chart) data sources. 
    It provides comprehensive logging and error handling for robust pipeline execution.
    
    Pipeline Modes:
        - TRAINING: File-based processing for ML model training and backtesting
        - LIVE: Real-time data processing from Sierra Chart for live trading
        - AUTO: Automatically detect mode based on configuration
    
    Attributes:
        config (Dict[str, Any]): Configuration dictionary containing pipeline parameters
        mode (PipelineMode): Explicit pipeline operation mode
        file_path (Optional[str]): Path to data file if using training mode
        df (Optional[pd.DataFrame]): DataFrame for live data processing
        data_source (Optional[str]): Source type indicator ('training' or 'live')
        processor (Optional[DataFrameProcessor]): Processor instance for file-based data
        logger (logging.Logger): Logger instance for this class
    """

    def __init__(self, config: Dict[str, Any], mode: PipelineMode = PipelineMode.AUTO) -> None:
        """
        Initialize the DataPipelineRunner with configuration and mode.
        
        Args:
            config (Dict[str, Any]): Configuration dictionary that must contain either
                'file_path' for training mode or 'df' for live mode.
                Supported keys:
                - file_path (str, optional): Path to CSV file for training mode
                - df (pd.DataFrame, optional): DataFrame for live data processing
                - sierra_chart_config (dict, optional): Sierra Chart connection config
            mode (PipelineMode): Explicit pipeline mode selection
                - TRAINING: Force file-based processing
                - LIVE: Force live data processing  
                - AUTO: Auto-detect based on config (default)
        
        Raises:
            TypeError: If config is not a dictionary
            ValueError: If mode doesn't match available data sources
            
        Examples:
            >>> # Training mode for ML model development
            >>> config = {'file_path': 'data/historical_data.csv'}
            >>> pipeline = DataPipelineRunner(config, PipelineMode.TRAINING)
            
            >>> # Live mode for real-time trading
            >>> config = {'df': live_dataframe}
            >>> pipeline = DataPipelineRunner(config, PipelineMode.LIVE)
            
            >>> # Auto-detect mode (backwards compatible)
            >>> config = {'file_path': 'data/market_data.csv'}
            >>> pipeline = DataPipelineRunner(config)  # Uses AUTO mode
        """
        if not isinstance(config, dict):
            raise TypeError("Config must be a dictionary")

        self.config = config
        self.mode = mode
        self.file_path: Optional[str] = config.get('file_path')
        self.df: Optional[pd.DataFrame] = config.get('df')
        self.sierra_chart_config: Optional[Dict[str, Any]] = config.get('sierra_chart_config')
        self.data_source: Optional[str] = None
        self.processor: Optional[DataFrameProcessor] = None
        self._effective_mode: Optional[PipelineMode] = None  # Cache effective mode after determination

        # Validate mode compatibility
        self._validate_mode_config()

        # Set up module-specific logging
        self.logger = logging.getLogger(__name__)
        self.logger.info("DataPipelineRunner initialized - Mode: %s, Config keys: %s", 
                        self.mode.value, list(config.keys()))
        self.logger.debug("Initializing DataPipelineRunner instance at memory address: %s", id(self))

    def _validate_mode_config(self) -> None:
        """
        Validate that the selected mode is compatible with the provided configuration.
        
        Raises:
            ValueError: If mode doesn't match available data sources
        """
        if self.mode == PipelineMode.TRAINING:
            if self.file_path is None:
                raise ValueError("Training mode requires 'file_path' in configuration")
        elif self.mode == PipelineMode.LIVE:
            if self.df is None and self.sierra_chart_config is None:
                raise ValueError("Live mode requires either 'df' or 'sierra_chart_config' in configuration")
        # AUTO mode doesn't require validation - it will auto-detect
        
    def get_effective_mode(self) -> PipelineMode:
        """
        Get the effective pipeline mode after auto-detection.
        
        Returns:
            PipelineMode: The actual mode being used for processing
        """
        if self._effective_mode is not None:
            return self._effective_mode
            
        if self.mode == PipelineMode.AUTO:
            # Auto-detect based on available data sources
            if self.df is not None or self.sierra_chart_config is not None:
                self._effective_mode = PipelineMode.LIVE
            elif self.file_path is not None:
                self._effective_mode = PipelineMode.TRAINING
            else:
                raise ValueError("No valid data source found for auto-detection")
        else:
            self._effective_mode = self.mode
            
        return self._effective_mode

    def process_data(self) -> None:
        """
        Process data from the configured source based on pipeline mode.
        
        This method handles data processing according to the selected pipeline mode:
        - TRAINING mode: Load and process data from file for ML training/backtesting
        - LIVE mode: Process real-time data from Sierra Chart or provided DataFrame
        - AUTO mode: Auto-detect and process based on available data sources
        
        Raises:
            ValueError: If no valid data source is found or mode is incompatible
            FileNotFoundError: If file_path is provided but file doesn't exist
            Exception: For any other processing errors
            
        Side Effects:
            - Sets self.df to the processed DataFrame
            - Sets self.data_source to indicate the source type ('training' or 'live')
            - Creates self.processor if using file-based processing
        """
        try:
            effective_mode = self.get_effective_mode()
            self.logger.info("Processing data in %s mode", effective_mode.value)
            
            if effective_mode == PipelineMode.TRAINING:
                # Training mode: File-based processing for ML training
                self.logger.info("Training mode: Processing data from file path: %s", self.file_path)
                self.processor = DataFrameProcessor(self.file_path)
                self.df = self.processor.process_data()
                self.data_source = 'training'
                
            elif effective_mode == PipelineMode.LIVE:
                # Live mode: Real-time data processing
                if self.df is not None:
                    self.logger.info("Live mode: Using provided DataFrame for real-time processing")
                    self.data_source = 'live'
                elif self.sierra_chart_config is not None:
                    self.logger.info("Live mode: Connecting to Sierra Chart for real-time data")
                    # NOTE: Sierra Chart live data connection will be implemented in future release
                    # This would integrate with the Sierra Chart bridge for real-time data
                    self.logger.warning("Sierra Chart live connection not yet implemented")
                    raise NotImplementedError("Sierra Chart live connection feature coming soon")
                else:
                    raise ValueError("Live mode requires either 'df' or 'sierra_chart_config'")
            
            # Log the successful data processing with the source and shape of the DataFrame
            if self.df is not None:
                self.logger.info(
                    "Data processing completed. Mode: %s, Data source: %s, DataFrame shape: %s", 
                    effective_mode.value,
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
        Get information about the processed data and pipeline mode.
        
        Returns:
            Dict[str, Any]: Dictionary containing data information including:
                - mode: Pipeline mode ('training', 'live', or 'auto')
                - effective_mode: Actual mode being used after auto-detection
                - data_source: Source type ('training' or 'live')
                - shape: Tuple of (rows, columns) if data is available
                - file_path: Path to source file if applicable
                - columns: List of column names if data is available
                
        Example:
            >>> info = pipeline.get_data_info()
            >>> print(f"Running in {info['effective_mode']} mode with {info['shape'][0]} rows")
        """
        try:
            effective_mode = self.get_effective_mode()
        except ValueError:
            effective_mode = None
            
        info = {
            'mode': self.mode.value,
            'effective_mode': effective_mode.value if effective_mode else None,
            'data_source': self.data_source,
            'file_path': self.file_path,
            'shape': self.df.shape if self.df is not None else None,
            'columns': list(self.df.columns) if self.df is not None else None
        }
        return info
