"""
Data Pipeline Runner Module.

This module provides the main pipeline orchestration for processing financial market data
within the Project Chimera trading system. It handles both live data streams and file-based
data processing workflows, serving as the central coordinator for data flow from sources
through processing to analysis-ready format.

The pipeline architecture supports flexibility and adaptability:
- Training Mode: Historical data from files for model development and backtesting
- Live Mode: Real-time streaming data from Sierra Chart for active trading
- Auto Mode: Intelligent detection of appropriate mode based on configuration

This design pattern enables:
- Consistent processing logic across training and production environments
- Easy switching between development (file-based) and production (live) data
- Robust error handling and comprehensive logging for both modes
- Configuration-driven operation for flexible deployment scenarios

The pipeline supports two distinct operational modes:
1. Training Mode: File-based processing for ML model training, backtesting, and analysis
   - Reads historical data from CSV files or other persistent storage
   - Ideal for model development, feature engineering, and strategy testing
   - Provides reproducible results with static datasets
   
2. Live Mode: Real-time data processing from Sierra Chart for live trading
   - Processes streaming data from active market connections
   - Supports real-time feature engineering and model inference
   - Integrates with trading systems for decision-making

3. Auto Mode: Intelligent mode selection based on configuration
   - Automatically detects appropriate mode from available data sources
   - Maintains backward compatibility with existing configurations
   - Simplifies deployment by reducing manual configuration

Classes:
    PipelineMode: Enum defining available pipeline operation modes
    DataPipelineRunner: Main orchestrator for data processing pipelines with mode support

Dependencies:
    pandas: DataFrame operations and time-series data manipulation
    logging: System logging for monitoring, debugging, and audit trails
    typing: Type hints for improved code quality and IDE support
    enum: Enumeration support for mode definitions

Example:
    ```python
    from common.data_pipeline.run_data_pipeline import DataPipelineRunner, PipelineMode
    
    # Training mode for model development
    config = {'file_path': 'data/historical_vbp_data.csv'}
    pipeline = DataPipelineRunner(config, PipelineMode.TRAINING)
    training_data = pipeline.run_pipeline()
    
    # Live mode for real-time trading
    config = {'df': realtime_dataframe}
    pipeline = DataPipelineRunner(config, PipelineMode.LIVE)
    live_data = pipeline.run_pipeline()
    
    # Auto mode (detects from config)
    config = {'file_path': 'data/market_data.csv'}
    pipeline = DataPipelineRunner(config)  # AUTO mode by default
    data = pipeline.run_pipeline()
    ```

Author: Roy Williams
Version: 1.1.0
"""

# Standard library imports
# Import logging module for comprehensive system logging, debugging, and monitoring
# Provides structured logging with levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
import logging

# Import type hinting primitives for documenting function signatures and data structures
# Dict: typed dictionary container, Any: accepts any type, Optional: value or None
from typing import Dict, Any, Optional

# Import Enum base class for creating enumerated constants
# Used to define PipelineMode with fixed set of valid values
from enum import Enum

# Third-party imports
# Import pandas for DataFrame operations and time-series data manipulation
# Core data structure for all pipeline processing operations
import pandas as pd

# Local imports
# Import DataFrameProcessor for handling file-based data loading and processing
# This processor reads CSV files and performs initial data transformations
from .dataframe_processor import DataFrameProcessor


class PipelineMode(Enum):
    """
    Enumeration of available pipeline operation modes.
    
    This enum defines the three operational modes supported by the data pipeline,
    providing type-safe mode selection and clear intent in code. Using an enum
    prevents invalid mode strings and enables IDE autocomplete support.
    
    Attributes:
        TRAINING (str): File-based processing mode for ML training and backtesting.
            Use this mode when working with historical data files, developing models,
            or running backtests. Data is loaded from persistent storage (CSV files).
            
        LIVE (str): Real-time processing mode for live trading with Sierra Chart.
            Use this mode when processing streaming data from active market connections.
            Data comes from real-time sources like Sierra Chart bridge or DataFrames.
            
        AUTO (str): Automatic mode detection based on configuration.
            The pipeline automatically determines the appropriate mode by examining
            the configuration. If 'df' or 'sierra_chart_config' exists, uses LIVE mode.
            If 'file_path' exists, uses TRAINING mode. This is the default mode.
    
    Example:
        >>> # Explicit mode selection
        >>> mode = PipelineMode.TRAINING
        >>> print(mode.value)  # 'training'
        >>> 
        >>> # Mode comparison
        >>> if pipeline.mode == PipelineMode.LIVE:
        ...     print("Running in live mode")
        >>> 
        >>> # Get all available modes
        >>> for mode in PipelineMode:
        ...     print(mode.name, mode.value)
    """
    # File-based processing mode for ML model training, backtesting, and historical analysis
    # Data source: CSV files or other persistent storage containing historical market data
    TRAINING = "training"
    
    # Real-time processing mode for live trading and streaming data from Sierra Chart
    # Data source: Live DataFrame updates or Sierra Chart bridge connections
    LIVE = "live"
    
    # Automatic mode detection based on configuration keys present
    # Intelligently selects TRAINING or LIVE based on available data sources
    AUTO = "auto"


class DataPipelineRunner:
    """
    Main orchestrator for data processing pipelines with multi-mode support.
    
    This class handles the complete lifecycle of data processing workflows, from
    initialization through execution to completion. It provides a unified interface
    for both training (file-based) and live (streaming) data sources, enabling
    seamless transitions between development and production environments.
    
    The orchestrator responsibilities include:
    - Configuration validation and mode selection
    - Data source initialization (files or live streams)
    - Processing workflow execution with error handling
    - Comprehensive logging and monitoring
    - Result validation and delivery
    
    Pipeline Modes:
        - TRAINING: File-based processing for ML model training and backtesting
          Loads historical data from CSV files for reproducible analysis
          
        - LIVE: Real-time data processing from Sierra Chart or live DataFrames
          Processes streaming market data for active trading decisions
          
        - AUTO: Automatically detect mode based on configuration keys
          Intelligent selection based on presence of 'file_path', 'df', or 'sierra_chart_config'
    
    Attributes:
        config (Dict[str, Any]): Configuration dictionary containing pipeline parameters.
            Controls data sources, processing options, and pipeline behavior.
            
        mode (PipelineMode): Explicit pipeline operation mode (TRAINING, LIVE, or AUTO).
            Determines how the pipeline interprets and processes configuration.
            
        file_path (Optional[str]): Path to data file if using training mode.
            Points to CSV or other data file for historical data loading.
            
        df (Optional[pd.DataFrame]): DataFrame for live data processing.
            Contains real-time or pre-loaded data for immediate processing.
            
        sierra_chart_config (Optional[Dict[str, Any]]): Sierra Chart connection configuration.
            Settings for establishing live data connection (future implementation).
            
        data_source (Optional[str]): Source type indicator ('training' or 'live').
            Set during processing to track which data source is active.
            
        processor (Optional[DataFrameProcessor]): Processor instance for file-based data.
            Handles CSV loading and initial transformations for training mode.
            
        logger (logging.Logger): Logger instance for this class.
            Provides structured logging for monitoring and debugging.
            
        _effective_mode (Optional[PipelineMode]): Cached effective mode after auto-detection.
            Stores the resolved mode to avoid repeated detection logic.
    
    Example:
        >>> # Training pipeline for model development
        >>> config = {'file_path': 'data/vbp_historical.csv'}
        >>> pipeline = DataPipelineRunner(config, PipelineMode.TRAINING)
        >>> df = pipeline.run_pipeline()
        >>> 
        >>> # Live pipeline for real-time trading
        >>> config = {'df': streaming_dataframe}
        >>> pipeline = DataPipelineRunner(config, PipelineMode.LIVE)
        >>> df = pipeline.run_pipeline()
        >>> 
        >>> # Get pipeline information
        >>> info = pipeline.get_data_info()
        >>> print(f"Mode: {info['effective_mode']}, Shape: {info['shape']}")
    
    Note:
        - Configuration must contain appropriate data source for selected mode
        - TRAINING mode requires 'file_path'
        - LIVE mode requires 'df' or 'sierra_chart_config'
        - AUTO mode selects based on available keys
        - Sierra Chart live connection is planned for future release
    """

    def __init__(self, config: Dict[str, Any], mode: PipelineMode = PipelineMode.AUTO) -> None:
        """
        Initialize the DataPipelineRunner with configuration and operational mode.
        
        Sets up the pipeline orchestrator by validating configuration, extracting
        data source parameters, and preparing logging infrastructure. Performs
        upfront validation to catch configuration errors before processing begins.
        
        Args:
            config (Dict[str, Any]): Configuration dictionary that must contain appropriate
                data source parameters for the selected mode.
                
                Supported keys:
                - file_path (str, optional): Absolute or relative path to CSV file.
                  Required for TRAINING mode. Used to load historical market data.
                  Example: 'data/raw/historical_vbp_data.csv'
                  
                - df (pd.DataFrame, optional): Pre-loaded DataFrame for immediate processing.
                  Used in LIVE mode for processing real-time or pre-fetched data.
                  Should contain market data with appropriate columns and index.
                  
                - sierra_chart_config (dict, optional): Configuration for Sierra Chart connection.
                  Used in LIVE mode for establishing streaming data connection.
                  Future implementation for direct Sierra Chart integration.
                  Example: {'host': 'localhost', 'port': 11099}
                  
            mode (PipelineMode, optional): Explicit pipeline mode selection.
                Defaults to PipelineMode.AUTO for automatic detection.
                
                - TRAINING: Force file-based processing (requires file_path)
                - LIVE: Force live data processing (requires df or sierra_chart_config)
                - AUTO: Auto-detect based on config (recommended for flexibility)
        
        Raises:
            TypeError: If config is not a dictionary type.
            ValueError: If mode doesn't match available data sources in configuration.
                For example, TRAINING mode without file_path, or LIVE mode without
                df or sierra_chart_config.
            
        Examples:
            >>> # Training mode with explicit file path
            >>> config = {'file_path': 'data/historical_data.csv'}
            >>> pipeline = DataPipelineRunner(config, PipelineMode.TRAINING)
            >>> 
            >>> # Live mode with DataFrame
            >>> import pandas as pd
            >>> df = pd.read_csv('realtime_data.csv')
            >>> config = {'df': df}
            >>> pipeline = DataPipelineRunner(config, PipelineMode.LIVE)
            >>> 
            >>> # Auto mode (backward compatible, recommended)
            >>> config = {'file_path': 'data/market_data.csv'}
            >>> pipeline = DataPipelineRunner(config)  # Uses AUTO mode by default
            >>> 
            >>> # Future: Sierra Chart live connection
            >>> config = {'sierra_chart_config': {'host': 'localhost', 'port': 11099}}
            >>> pipeline = DataPipelineRunner(config, PipelineMode.LIVE)
        
        Note:
            - Validation happens immediately during initialization
            - Logger is configured to use module name for traceability
            - Effective mode is determined lazily on first access (for AUTO mode)
            - Configuration is stored as-is without modification
        """
        # Validate that config parameter is a dictionary type
        # This upfront check prevents confusing errors later in processing
        # Type validation ensures config.get() calls will work properly
        if not isinstance(config, dict):
            raise TypeError("Config must be a dictionary")

        # Store the complete configuration dictionary for access throughout pipeline
        # Contains all parameters needed for data loading and processing
        # Preserved as-is without modification for transparency
        self.config = config
        
        # Store the explicitly selected or default pipeline mode
        # This controls how the pipeline interprets configuration and processes data
        # AUTO mode will trigger lazy mode detection on first use
        self.mode = mode
        
        # Extract file path from configuration if present
        # Used in TRAINING mode to locate and load historical data files
        # None if not provided (valid for LIVE mode)
        self.file_path: Optional[str] = config.get('file_path')
        
        # Extract DataFrame from configuration if present
        # Used in LIVE mode for immediate processing of pre-loaded data
        # None if not provided (valid for TRAINING mode)
        self.df: Optional[pd.DataFrame] = config.get('df')
        
        # Extract Sierra Chart configuration if present
        # Used in LIVE mode for establishing streaming data connections
        # None if not provided (feature planned for future implementation)
        self.sierra_chart_config: Optional[Dict[str, Any]] = config.get('sierra_chart_config')
        
        # Initialize data source indicator to None
        # Will be set to 'training' or 'live' during processing
        # Tracks which data source is actively being used
        self.data_source: Optional[str] = None
        
        # Initialize processor to None
        # Will be instantiated in TRAINING mode to handle file loading
        # Remains None in LIVE mode (not needed for DataFrame processing)
        self.processor: Optional[DataFrameProcessor] = None
        
        # Initialize effective mode cache to None
        # Stores the resolved mode after auto-detection in AUTO mode
        # Prevents repeated mode detection logic on subsequent accesses
        self._effective_mode: Optional[PipelineMode] = None

        # Validate that selected mode is compatible with provided configuration
        # Ensures TRAINING has file_path and LIVE has df or sierra_chart_config
        # Raises ValueError if validation fails, preventing invalid pipeline execution
        self._validate_mode_config()

        # Set up module-specific logger using the module's __name__
        # Logger name includes full module path for traceable log messages
        # Example: "common.data_pipeline.run_data_pipeline"
        self.logger = logging.getLogger(__name__)
        
        # Log initialization with mode and configuration keys for audit trail
        # Uses lazy % formatting (Pylint best practice) for efficiency
        # Helps track when and how pipelines are created
        self.logger.info("DataPipelineRunner initialized - Mode: %s, Config keys: %s", 
                        self.mode.value, list(config.keys()))
        
        # Log debug message with memory address for advanced troubleshooting
        # Useful for tracking object lifecycle and identifying memory leaks
        # id(self) returns unique object identifier in memory
        self.logger.debug("Initializing DataPipelineRunner instance at memory address: %s", id(self))

    def _validate_mode_config(self) -> None:
        """
        Validate that the selected mode is compatible with the provided configuration.
        
        This private method performs upfront validation to ensure the configuration
        contains appropriate data sources for the selected pipeline mode. Early
        validation prevents cryptic errors during processing and provides clear
        feedback about configuration problems.
        
        Validation rules:
        - TRAINING mode: Must have 'file_path' in configuration
        - LIVE mode: Must have either 'df' or 'sierra_chart_config'
        - AUTO mode: No validation (will detect or fail gracefully later)
        
        Raises:
            ValueError: If mode requirements are not met by configuration.
                Includes specific message indicating which key is missing.
        
        Example:
            >>> # This would raise ValueError
            >>> config = {'df': some_dataframe}
            >>> pipeline = DataPipelineRunner(config, PipelineMode.TRAINING)
            >>> # ValueError: Training mode requires 'file_path' in configuration
        
        Note:
            - Called automatically during __init__
            - Validation is strict for explicit modes (TRAINING, LIVE)
            - AUTO mode defers validation to get_effective_mode()
            - Helps catch configuration errors before processing begins
        """
        # Check if TRAINING mode was explicitly selected
        if self.mode == PipelineMode.TRAINING:
            # TRAINING mode requires a file path to load historical data
            # Verify that file_path key exists and has a non-None value
            if self.file_path is None:
                # Raise ValueError with clear message about missing requirement
                raise ValueError("Training mode requires 'file_path' in configuration")
                
        # Check if LIVE mode was explicitly selected
        elif self.mode == PipelineMode.LIVE:
            # LIVE mode requires either a DataFrame or Sierra Chart config
            # Both None means no data source is available for live processing
            if self.df is None and self.sierra_chart_config is None:
                # Raise ValueError explaining the two valid options for LIVE mode
                raise ValueError("Live mode requires either 'df' or 'sierra_chart_config' in configuration")
                
        # AUTO mode doesn't require validation at initialization
        # Mode detection will happen later in get_effective_mode()
        # This allows flexible configuration without premature failures
        
    def get_effective_mode(self) -> PipelineMode:
        """
        Get the effective pipeline mode after auto-detection.
        
        This method resolves the actual mode being used, performing auto-detection
        if AUTO mode was selected. The result is cached to avoid repeated detection
        logic on subsequent calls. This lazy evaluation pattern optimizes performance
        while maintaining flexibility.
        
        Auto-detection logic:
        1. If mode is LIVE or TRAINING, return it directly (no detection needed)
        2. If mode is AUTO, examine configuration keys:
           - If 'df' or 'sierra_chart_config' exists → LIVE mode
           - If 'file_path' exists → TRAINING mode
           - If no valid source exists → raise ValueError
        3. Cache the detected mode for future calls
        
        Returns:
            PipelineMode: The actual mode being used for processing.
                Either the explicitly set mode or the auto-detected mode.
        
        Raises:
            ValueError: If AUTO mode is used but no valid data source is found
                in the configuration. Indicates invalid or incomplete configuration.
        
        Example:
            >>> # Explicit mode (no detection)
            >>> config = {'file_path': 'data.csv'}
            >>> pipeline = DataPipelineRunner(config, PipelineMode.TRAINING)
            >>> mode = pipeline.get_effective_mode()  # Returns TRAINING
            >>> 
            >>> # Auto-detection from config
            >>> config = {'df': dataframe}
            >>> pipeline = DataPipelineRunner(config, PipelineMode.AUTO)
            >>> mode = pipeline.get_effective_mode()  # Returns LIVE (detected)
            >>> 
            >>> # Subsequent calls use cached value
            >>> mode = pipeline.get_effective_mode()  # Returns cached LIVE
        
        Note:
            - First call may perform detection; subsequent calls use cache
            - Cache is stored in self._effective_mode
            - Detection only happens for AUTO mode
            - Explicit modes (TRAINING, LIVE) bypass detection
        """
        # Check if effective mode has already been determined and cached
        # If cache exists, return it immediately without re-detection
        # This optimization prevents redundant detection logic on repeated calls
        if self._effective_mode is not None:
            return self._effective_mode
        
        # Check if AUTO mode was selected (requires detection)
        if self.mode == PipelineMode.AUTO:
            # Auto-detect based on available data sources in configuration
            # Priority: df/sierra_chart_config (LIVE) > file_path (TRAINING)
            
            # Check for LIVE mode indicators (df or Sierra Chart config)
            # If either exists, this is a live data processing scenario
            if self.df is not None or self.sierra_chart_config is not None:
                # Set effective mode to LIVE and cache it
                self._effective_mode = PipelineMode.LIVE
                
            # Check for TRAINING mode indicator (file_path)
            # If file path exists and no live sources, this is training scenario
            elif self.file_path is not None:
                # Set effective mode to TRAINING and cache it
                self._effective_mode = PipelineMode.TRAINING
                
            # No valid data source found in configuration
            # This indicates incomplete or invalid configuration
            else:
                # Raise ValueError explaining the problem
                # User must provide at least one valid data source
                raise ValueError("No valid data source found for auto-detection")
        else:
            # Mode was explicitly set (TRAINING or LIVE), no detection needed
            # Simply cache the explicit mode for consistency
            self._effective_mode = self.mode
        
        # Return the determined effective mode (either detected or explicit)
        # Value is now cached for future calls
        return self._effective_mode

    def process_data(self) -> None:
        """
        Process data from the configured source based on pipeline mode.
        
        This method orchestrates the actual data processing workflow, loading data
        from the appropriate source (file or live) based on the effective mode.
        It handles mode-specific initialization, data loading, and validation,
        with comprehensive error handling and logging throughout.
        
        Processing workflow:
        1. Determine effective mode (with auto-detection if needed)
        2. Branch to appropriate processing logic (TRAINING or LIVE)
        3. Load/process data from the selected source
        4. Validate results and set data_source indicator
        5. Log completion status and data characteristics
        
        This method handles data processing according to the selected pipeline mode:
        - TRAINING mode: Load and process data from file for ML training/backtesting
          Creates DataFrameProcessor, loads CSV file, performs transformations
          
        - LIVE mode: Process real-time data from Sierra Chart or provided DataFrame
          Uses pre-loaded DataFrame or establishes Sierra Chart connection
          
        - AUTO mode: Auto-detect and process based on available data sources
          Intelligently selects appropriate processing path
        
        Raises:
            ValueError: If no valid data source is found or mode is incompatible.
                Also raised if processing completes but no DataFrame was created.
                
            FileNotFoundError: If file_path is provided but file doesn't exist at
                that location. Includes original exception message for debugging.
                
            NotImplementedError: If Sierra Chart live connection is requested.
                Feature is planned for future implementation.
                
            Exception: For any other processing errors. Original exception is
                logged and re-raised for proper error propagation.
            
        Side Effects:
            - Sets self.df to the processed DataFrame
            - Sets self.data_source to indicate the source type ('training' or 'live')
            - Creates self.processor if using file-based processing
            - Logs multiple messages tracking processing progress
        
        Example:
            >>> # File-based processing
            >>> config = {'file_path': 'data/market_data.csv'}
            >>> pipeline = DataPipelineRunner(config)
            >>> pipeline.process_data()  # Loads and processes file
            >>> print(pipeline.df.shape)
            >>> 
            >>> # Live DataFrame processing
            >>> config = {'df': realtime_dataframe}
            >>> pipeline = DataPipelineRunner(config)
            >>> pipeline.process_data()  # Uses provided DataFrame
            >>> print(pipeline.data_source)  # 'live'
        
        Note:
            - This method modifies instance state (self.df, self.data_source, self.processor)
            - Use run_pipeline() for complete workflow with error handling
            - All exceptions are logged before being re-raised
            - Processing is idempotent (safe to call multiple times)
        """
        try:
            # Determine the effective mode (handles AUTO mode detection if needed)
            # This resolves which processing path to take
            # Returns cached value if already determined, otherwise performs detection
            effective_mode = self.get_effective_mode()
            
            # Log the mode being used for this processing run
            # Helps track which processing path is executed in log files
            # Uses lazy % formatting for Pylint compliance
            self.logger.info("Processing data in %s mode", effective_mode.value)
            
            # Check if effective mode is TRAINING (file-based processing)
            if effective_mode == PipelineMode.TRAINING:
                # Training mode: File-based processing for ML training and backtesting
                # This path loads historical data from persistent storage
                
                # Validate that file_path is not None before proceeding
                # This should always be true due to _validate_mode_config(), but check for type safety
                # Type checker needs explicit confirmation that file_path is str, not str | None
                if self.file_path is None:
                    # This should never happen if validation worked correctly
                    # Raise error to maintain type safety and catch configuration bugs
                    raise ValueError("TRAINING mode requires file_path but it is None")
                
                # Log the file being processed for audit trail
                # Useful for debugging file path issues or tracking data sources
                self.logger.info("Training mode: Processing data from file path: %s", self.file_path)
                
                # Create DataFrameProcessor instance to handle file loading and processing
                # Processor encapsulates CSV reading, parsing, and initial transformations
                # Raises FileNotFoundError if file doesn't exist at specified path
                # file_path is now confirmed to be str (not None) due to check above
                self.processor = DataFrameProcessor(self.file_path)
                
                # Execute the processing workflow to load and transform data
                # Returns a processed DataFrame ready for analysis
                # May apply various transformations depending on processor configuration
                self.df = self.processor.process_data()
                
                # Set data source indicator to 'training' for tracking
                # This helps distinguish training data from live data in downstream logic
                self.data_source = 'training'
                
            # Check if effective mode is LIVE (real-time processing)
            elif effective_mode == PipelineMode.LIVE:
                # Live mode: Real-time data processing from active sources
                # This path handles streaming or pre-loaded live data
                
                # Check if DataFrame was pre-provided in configuration
                if self.df is not None:
                    # DataFrame already exists, no loading needed
                    # This is the current implementation for live data
                    self.logger.info("Live mode: Using provided DataFrame for real-time processing")
                    
                    # Set data source indicator to 'live' for tracking
                    # Distinguishes this from training/historical data
                    self.data_source = 'live'
                    
                # Check if Sierra Chart configuration was provided
                elif self.sierra_chart_config is not None:
                    # Sierra Chart live connection requested
                    # This is a planned feature for future implementation
                    self.logger.info("Live mode: Connecting to Sierra Chart for real-time data")
                    
                    # Log warning that this feature is not yet available
                    # Helps users understand current limitations
                    self.logger.warning("Sierra Chart live connection not yet implemented")
                    
                    # Raise NotImplementedError to indicate planned feature
                    # Clear message informs users this is coming in future release
                    raise NotImplementedError("Sierra Chart live connection feature coming soon")
                    
                else:
                    # Neither df nor sierra_chart_config provided in LIVE mode
                    # This shouldn't happen if validation worked, but check anyway
                    raise ValueError("Live mode requires either 'df' or 'sierra_chart_config'")
            
            # Processing completed, log success with data details
            # Check if DataFrame was successfully created/loaded
            if self.df is not None:
                # Log comprehensive completion message with mode, source, and shape
                # Shape provides quick validation that data was loaded correctly
                # Uses lazy % formatting for Pylint compliance
                self.logger.info(
                    "Data processing completed. Mode: %s, Data source: %s, DataFrame shape: %s", 
                    effective_mode.value,
                    self.data_source, 
                    self.df.shape
                )
            else:
                # DataFrame is None after processing, which indicates a problem
                # This shouldn't normally happen but log warning if it does
                self.logger.warning("Data processing completed but DataFrame is None")

        # Handle file not found errors specifically for better error messages
        except FileNotFoundError as e:
            # Log error with file path and original exception message
            # Helps users quickly identify which file is missing
            self.logger.error("File not found at path '%s': %s", self.file_path, e)
            # Re-raise exception to propagate error up the call stack
            raise
            
        # Handle all other exceptions generically
        except Exception as e:
            # Log error with exception message for debugging
            # Captures any unexpected errors during processing
            self.logger.error("Error during data processing: %s", str(e))
            # Re-raise exception to propagate error up the call stack
            raise

    def run_pipeline(self) -> pd.DataFrame:
        """
        Execute the complete data pipeline workflow.
        
        This is the main entry point for running the entire data processing pipeline
        from start to finish. It orchestrates all pipeline stages including data
        loading, processing, validation, and result delivery, with comprehensive
        logging and error handling at each stage.
        
        Pipeline execution flow:
        1. Log pipeline start
        2. Call process_data() to load and process data
        3. Validate that processing produced a DataFrame
        4. Log successful completion
        5. Return processed DataFrame
        
        This method wraps process_data() with additional logging and validation,
        making it the recommended way to execute the pipeline rather than calling
        process_data() directly.
        
        Returns:
            pd.DataFrame: The processed DataFrame ready for analysis, feature
                engineering, or model training. Contains all loaded data with
                appropriate transformations applied.
            
        Raises:
            ValueError: If data source configuration is invalid, or if pipeline
                completes but no DataFrame was produced (unexpected state).
                
            FileNotFoundError: If TRAINING mode file path doesn't exist.
            
            NotImplementedError: If Sierra Chart connection is requested.
            
            Exception: For any other processing errors. All exceptions are logged
                before being re-raised for proper error propagation.
            
        Example:
            >>> # Training pipeline with file
            >>> config = {'file_path': 'data/market_data.csv'}
            >>> pipeline = DataPipelineRunner(config)
            >>> df = pipeline.run_pipeline()
            >>> print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
            >>> 
            >>> # Live pipeline with DataFrame
            >>> config = {'df': streaming_data}
            >>> pipeline = DataPipelineRunner(config, PipelineMode.LIVE)
            >>> df = pipeline.run_pipeline()
            >>> print(f"Processing live data: {df.shape}")
            >>> 
            >>> # Use processed data for analysis
            >>> df = pipeline.run_pipeline()
            >>> analysis_results = analyze_market_data(df)
        
        Note:
            - This is the recommended method for executing the pipeline
            - Provides comprehensive logging for monitoring and debugging
            - Validates results before returning
            - All errors are logged and re-raised
            - Safe to call multiple times (idempotent)
        """
        try:
            # Log the start of pipeline execution
            # Marks the beginning of the complete processing workflow
            # Helps track timing and identify pipeline runs in logs
            self.logger.info("Starting data pipeline execution...")
            
            # Execute the data processing workflow
            # This loads data from the configured source and applies transformations
            # Sets self.df and self.data_source as side effects
            # May raise various exceptions if errors occur
            self.process_data()
            
            # Log successful completion of pipeline
            # Indicates all processing stages completed without errors
            # Success message helps quickly identify successful runs in logs
            self.logger.info("Data pipeline execution completed successfully.")
            
            # Validate that processing produced a DataFrame
            # This should always be true after successful process_data() call
            # Check catches unexpected states where processing "succeeds" but produces no data
            if self.df is None:
                # Raise ValueError indicating unexpected state
                # This helps catch bugs in processing logic
                raise ValueError("Pipeline completed but no data was processed")
            
            # Return the processed DataFrame to the caller
            # Contains all loaded and transformed data ready for use
            # This is the primary output of the pipeline
            return self.df
            
        # Catch all exceptions and log before re-raising
        except Exception as e:
            # Log error with exception message for debugging
            # Provides visibility into pipeline failures
            # Uses lazy % formatting for Pylint compliance
            self.logger.error("Data pipeline execution failed: %s", str(e))
            # Re-raise exception to propagate error up the call stack
            # Allows caller to handle or propagate error as appropriate
            raise
    
    def get_data_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about the processed data and pipeline state.
        
        This method returns a dictionary containing metadata about the pipeline's
        configuration, mode, data source, and processed data characteristics. Useful
        for debugging, logging, monitoring, and understanding pipeline state without
        accessing internal attributes directly.
        
        The info dictionary provides:
        - Configuration mode (user-specified)
        - Effective mode (after auto-detection)
        - Data source type (training vs live)
        - Data location (file path if applicable)
        - Data dimensions (shape if data loaded)
        - Available columns (if data loaded)
        
        Returns:
            Dict[str, Any]: Dictionary containing comprehensive pipeline information:
            
                - mode (str): User-specified pipeline mode ('training', 'live', or 'auto').
                  Shows what mode was requested during initialization.
                  
                - effective_mode (str | None): Actual mode being used after auto-detection.
                  For AUTO mode, shows the detected mode. For explicit modes, same as 'mode'.
                  None if mode detection hasn't run or failed.
                  
                - data_source (str | None): Source type indicator ('training' or 'live').
                  Set during processing. None if process_data() hasn't been called.
                  
                - file_path (str | None): Path to source file if using TRAINING mode.
                  None for LIVE mode or if not specified.
                  
                - shape (tuple | None): Tuple of (rows, columns) if data has been loaded.
                  Example: (10000, 15). None if process_data() hasn't been called.
                  
                - columns (list | None): List of column names if data has been loaded.
                  Example: ['DateTime', 'Close', 'Volume', 'RVOL']. None if no data.
                
        Example:
            >>> # Get pipeline info
            >>> config = {'file_path': 'data/vbp_data.csv'}
            >>> pipeline = DataPipelineRunner(config)
            >>> info = pipeline.get_data_info()
            >>> print(f"Mode: {info['mode']}")  # 'auto'
            >>> print(f"Effective: {info['effective_mode']}")  # 'training'
            >>> 
            >>> # After processing
            >>> pipeline.run_pipeline()
            >>> info = pipeline.get_data_info()
            >>> print(f"Shape: {info['shape']}")  # (5000, 12)
            >>> print(f"Columns: {info['columns']}")  # ['DateTime', 'Open', ...]
            >>> 
            >>> # Use for logging
            >>> info = pipeline.get_data_info()
            >>> logger.info(f"Pipeline status: {info}")
        
        Note:
            - Safe to call at any time (before or after processing)
            - Values are None if corresponding data isn't available yet
            - Effective mode may be None if auto-detection hasn't run
            - Does not modify pipeline state (read-only operation)
            - Useful for debugging and monitoring pipeline status
        """
        # Attempt to get the effective mode (may trigger auto-detection)
        # Wrap in try/except to handle case where detection fails
        try:
            # Get effective mode after detection (if AUTO mode)
            # Returns cached value if already determined
            effective_mode = self.get_effective_mode()
        except ValueError:
            # Auto-detection failed (no valid data source found)
            # Set effective_mode to None to indicate unavailable
            effective_mode = None
        
        # Build and return information dictionary with all available metadata
        # Each key provides specific aspect of pipeline state
        info = {
            # User-specified mode from initialization
            # Value is the enum value string ('training', 'live', 'auto')
            'mode': self.mode.value,
            
            # Actual mode after auto-detection (if applicable)
            # Extract .value if mode exists, otherwise None
            # Shows resolved mode for AUTO, same as mode for explicit
            'effective_mode': effective_mode.value if effective_mode else None,
            
            # Data source type set during processing ('training' or 'live')
            # None if process_data() hasn't been called yet
            'data_source': self.data_source,
            
            # File path for TRAINING mode
            # None for LIVE mode or if not specified
            'file_path': self.file_path,
            
            # Shape tuple (rows, columns) if data has been loaded
            # Example: (10000, 15) means 10,000 rows and 15 columns
            # None if no data loaded yet
            'shape': self.df.shape if self.df is not None else None,
            
            # List of column names if data has been loaded
            # Convert Index to list for JSON serialization compatibility
            # None if no data loaded yet
            'columns': list(self.df.columns) if self.df is not None else None
        }
        
        # Return the complete information dictionary
        # Contains all available metadata about pipeline state
        return info
