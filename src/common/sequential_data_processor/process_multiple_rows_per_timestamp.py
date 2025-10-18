"""
Sequential Data Processor Module for Multiple Rows Per Timestamp.

This module provides functionality for processing Volume by Price (VBP) chart data
that may contain multiple rows per timestamp. It handles sequential processing of
time-series data with duplicate detection, callback notifications, and robust error
handling for real-time or batch data processing workflows.

Classes:
    ProcessMultipleRowsPerTimestamp: Main class for processing multiple rows per timestamp
        with deduplication and callback support.

Example:
    >>> def my_callback(data):
    ...     print(f"Received data: {data}")
    >>> processor = ProcessMultipleRowsPerTimestamp(data_callback=my_callback)
    >>> processor.process_multiple_rows(new_vbp_dataframe)
"""

# Standard library imports
import logging
from typing import Optional, Callable, Any, Set, Union, cast
from datetime import datetime

# Third-party imports
import pandas as pd

# Local imports
from common.sequential_data_processor.process_each_row import ProcessEachRow

class ProcessMultipleRowsPerTimestamp:
    """
    Process multiple rows of VBP data per timestamp with deduplication.

    This class manages sequential processing of Volume by Price (VBP) chart data,
    handling cases where multiple data rows may exist for the same timestamp.
    It maintains state across processing calls, deduplicates data to prevent
    reprocessing, and provides callback notifications for downstream consumers.

    Attributes:
        vbp_chart_data_df (pd.DataFrame): Accumulated VBP chart data across all
            processing calls. Serves as the historical context for lookback operations.
        processed_rows_hashes (Set[int]): Set of hash values for rows that have
            already been processed. Used for deduplication to prevent redundant processing.
        row_processor (ProcessEachRow): Instance of row processor that handles
            individual row processing logic and feature extraction.
        data_callback (Optional[Callable[[Any], None]]): Optional callback function
            invoked with processed data. Enables real-time data streaming to subscribers.

    Example:
        >>> def handle_data(processed_data):
        ...     print(f"New data received: {processed_data}")
        >>> processor = ProcessMultipleRowsPerTimestamp(data_callback=handle_data)
        >>> new_data = pd.DataFrame(...)  # VBP data
        >>> processor.process_multiple_rows(new_data)
    """

    def __init__(self, data_callback: Optional[Callable[[Any], None]] = None) -> None:
        """
        Initialize the ProcessMultipleRowsPerTimestamp instance.

        Sets up the data structures and processors needed for sequential processing
        of VBP chart data. Initializes an empty DataFrame for accumulating data,
        a hash set for tracking processed rows, and configures the callback function
        for downstream notifications.

        Args:
            data_callback (Optional[Callable[[Any], None]], optional): Callback function
                that receives processed data as its argument. Called for each successfully
                processed row. If None, no callbacks are made. Defaults to None.

        Returns:
            None

        Example:
            >>> processor = ProcessMultipleRowsPerTimestamp()  # No callback
            >>> processor_with_cb = ProcessMultipleRowsPerTimestamp(
            ...     data_callback=lambda data: print(data)
            ... )
        """
        # Initialize an empty DataFrame to accumulate VBP chart data across processing calls
        # This DataFrame serves as the growing historical context for lookback operations
        # Structure: Index is DatetimeIndex, columns include Open, High, Low, Close, Volume, etc.
        self.vbp_chart_data_df: pd.DataFrame = pd.DataFrame()

        # Initialize an empty set to track hashes of processed rows for deduplication
        # Each hash represents a unique row's data content (not including index/timestamp)
        # This prevents reprocessing identical rows that may appear in different batches
        self.processed_rows_hashes: Set[int] = set()

        # Initialize the row processor that handles individual row processing logic
        # This processor performs feature extraction, market data processing, and formatting
        self.row_processor: ProcessEachRow = ProcessEachRow()

        # Store the callback function for notifying downstream consumers of processed data
        # Can be None if no notification is needed (e.g., batch processing mode)
        self.data_callback: Optional[Callable[[Any], None]] = data_callback

        # Log successful initialization with info level for debugging and monitoring
        logging.info("ProcessMultipleRowsPerTimestamp initialized.")

    def process_multiple_rows(self, new_vbp_data: pd.DataFrame) -> None:
        """
        Process new VBP chart data sequentially, handling multiple rows per timestamp.

        This method is the main entry point for processing batches of VBP data. It handles:
        1. Validation and normalization of incoming data (datetime index conversion)
        2. Accumulation of new data into the historical context DataFrame
        3. Deduplication to remove redundant data points
        4. Sequential row-by-row processing with hash-based duplicate detection
        5. Callback notifications for successfully processed rows
        6. Robust error handling with detailed logging

        The method maintains state across calls, accumulating all processed data into
        `self.vbp_chart_data_df` and tracking processed rows in `self.processed_rows_hashes`.
        This enables continuous, real-time processing of streaming data while building
        historical context for lookback operations.

        Args:
            new_vbp_data (pd.DataFrame): New VBP chart data to process. Should contain
                market data columns (Open, High, Low, Close, Volume, etc.) with a
                datetime-compatible index. Can have duplicate timestamps (multiple rows
                per timestamp). Empty DataFrames are handled gracefully.

        Returns:
            None: Results are delivered via callback function if configured.

        Raises:
            Exception: Errors during individual row processing are caught, logged,
                and do not stop processing of remaining rows.

        Side Effects:
            - Updates self.vbp_chart_data_df with new data
            - Updates self.processed_rows_hashes with processed row hashes
            - Invokes self.data_callback for each successfully processed row

        Example:
            >>> processor = ProcessMultipleRowsPerTimestamp(data_callback=my_handler)
            >>> new_data = pd.DataFrame({
            ...     'Open': [100, 101],
            ...     'Close': [102, 103]
            ... }, index=pd.DatetimeIndex(['2025-01-01', '2025-01-02']))
            >>> processor.process_multiple_rows(new_data)

        Note:
            - Duplicate rows (same data values) are automatically skipped
            - Index is automatically converted to DatetimeIndex if needed
            - Processing continues even if individual rows fail
            - Empty DataFrames trigger a warning but don't raise errors
        """
        # Check if the incoming DataFrame contains any rows to process
        # .empty returns True if DataFrame has no rows, False otherwise
        if not new_vbp_data.empty:
            # Optional logging for detailed debugging: track receipt of new data
            # Commented out to reduce log verbosity in production
            # logging.info("Received new VBP data with %d rows", len(new_vbp_data))
            # logging.info("Timestamp range: %s to %s",
            #              new_vbp_data.index.min(), new_vbp_data.index.max())

            # Validate and normalize the index to ensure it's a DatetimeIndex
            # This is crucial for time-based operations like lookback and sorting
            # isinstance checks if index is already a DatetimeIndex
            # to avoid unnecessary conversion
            if not isinstance(new_vbp_data.index, pd.DatetimeIndex):
                # Convert index to DatetimeIndex using pd.to_datetime()
                # This handles various datetime formats (strings, integers, etc.)
                new_vbp_data.index = pd.to_datetime(new_vbp_data.index)
                # Optional debug logging for tracking index conversions
                # logging.debug("Converted new_vbp_data index to DatetimeIndex.")

            # Append new data to the existing accumulated DataFrame using pd.concat
            # This builds the historical context needed for lookback operations
            # pd.concat returns a new DataFrame combining both DataFrames vertically
            # Note: This preserves all rows including potential duplicates (handled next)
            self.vbp_chart_data_df = pd.concat([self.vbp_chart_data_df, new_vbp_data])

            # Remove duplicate rows from the accumulated DataFrame
            # drop_duplicates() compares all columns and keeps only the first occurrence
            # This prevents data redundancy from overlapping batches or repeated data feeds
            # Returns: DataFrame with duplicate rows removed, maintaining original order
            self.vbp_chart_data_df = self.vbp_chart_data_df.drop_duplicates()

            # Iterate through each row in the new data batch for sequential processing
            # iterrows() returns an iterator yielding (index, Series) tuples
            # idx: timestamp from the DatetimeIndex
            # (type: Hashable, but we know it's pd.Timestamp)
            # row: Series containing all column values for this timestamp
            for idx, row in new_vbp_data.iterrows():
                # Cast idx to proper timestamp type since we've already validated
                # the index is DatetimeIndex. This satisfies type checking while
                # being safe due to our earlier index validation.
                # The cast tells the type checker that idx is
                # Union[pd.Timestamp, datetime] not just Hashable
                timestamp: Union[pd.Timestamp, datetime] = cast(
                    Union[pd.Timestamp, datetime], idx
                )

                # Create a unique hash for this row based on its data values (not index)
                # tuple(row.values) converts Series values to an immutable tuple
                # hash() generates an integer hash code for efficient lookup in set
                # This hash identifies identical data rows regardless of their timestamp
                row_hash: int = hash(tuple(row.values))

                # Check if this row has already been processed
                # using hash-based deduplication
                # The 'not in' operator performs O(1) lookup in the set
                # This prevents reprocessing identical rows from different batches
                if row_hash not in self.processed_rows_hashes:
                    # Wrap processing in try-except to handle errors gracefully
                    # This ensures one failing row doesn't stop processing
                    # of remaining rows
                    try:
                        # Process the individual row using the row processor instance
                        # Pass three arguments:
                        # 1. row: pd.Series with market data columns
                        # 2. timestamp: Union[pd.Timestamp, datetime] indicating
                        #    when this data occurred
                        # 3. self.vbp_chart_data_df: accumulated historical data
                        #    for lookback
                        # Returns: Dict with processed/transformed data
                        # or None if processing fails
                        processed_data: Any = self.row_processor.process_rows(
                            row,
                            timestamp,  # Pass the properly typed timestamp variable
                            self.vbp_chart_data_df
                        )

                        # Check if processing returned valid data (not None)
                        # process_rows may return None if row doesn't meet processing criteria
                        if processed_data is not None:
                            # Check if a callback function has been configured for notifications
                            # self.data_callback can be None in batch processing scenarios
                            if self.data_callback is not None:
                                # Invoke the callback function with the processed data
                                # This enables real-time streaming of results to subscribers
                                # The callback is responsible for handling/storing the data
                                self.data_callback(processed_data)

                        # Mark this row as processed by adding its hash
                        # to the tracking set
                        # This ensures we won't reprocess this identical row
                        # in future batches
                        # Uses O(1) set.add() operation for efficient tracking
                        self.processed_rows_hashes.add(row_hash)

                    # Catch any exception that occurs during row processing
                    # Using broad Exception to ensure processing continues
                    # regardless of error type
                    # pylint: disable=broad-exception-caught
                    # Justification: We want to catch all errors to prevent
                    # one failing row from stopping the entire batch processing
                    except Exception as e:
                        # Log the error with the timestamp and error message
                        # for debugging
                        # Use lazy % formatting (not f-strings) for Pylint compliance
                        # Include both the timestamp and error details
                        # for troubleshooting
                        logging.error("Error processing row at index %s: %s", timestamp, e)
                        # Continue to next iteration without adding
                        # to processed_rows_hashes
                        # This allows retry if the same row appears again
                        # (error might be transient)

                else:
                    # This row hash already exists in processed_rows_hashes
                    # Log informational message that row is being skipped
                    # (prevents duplicate processing)
                    # This is normal behavior and indicates
                    # the deduplication system is working
                    logging.info("Row at index %s already processed.", timestamp)

            # Optional logging for tracking completion of batch processing
            # Commented out to reduce log verbosity in production environments
            # logging.info("Finished processing new rows.")
            # logging.info("**************************************************")

        else:
            # Handle the case where an empty DataFrame was passed to the method
            # Log warning to alert that no processing occurred
            # (may indicate upstream issue)
            # This is non-fatal - method completes successfully
            # without processing anything
            logging.warning("No new VBP data to process.")
