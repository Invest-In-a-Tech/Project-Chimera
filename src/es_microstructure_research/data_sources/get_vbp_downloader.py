"""
Volume by Price (VBP) Chart Data Downloader Module.

This module provides a simple interface for downloading and saving Volume by Price (VBP) 
chart data from Sierra Chart. It imports and uses the GetVbpData class from the 
sc_py_bridge package to ensure consistency across the codebase.

Author: Roy Williams
Version: 2.0.0
Last Updated: November 2025
"""

# Standard library imports
import os
import logging

# Local imports
from src.sc_py_bridge.get_vbp_chart_data import GetVbpData

# Module-level logger
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    # Example usage when running this module as a script
    vbp_data = GetVbpData()
    result_df = vbp_data.get_vbp_chart_data()

    # Save the DataFrame to the specified directory using os.path
    output_dir = os.path.join('data', 'raw', 'dataframes')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'volume_by_price_testdata2.csv')
    result_df.to_csv(output_path)
    # logger.info(f"Volume by Price data saved to: {output_path}")
    print(f"Volume by Price data saved to: {output_path}")
