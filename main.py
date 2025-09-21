"""Entry point for Project Chimera.

At the moment, I'm just using this to test different
methods and classes.
"""

# Local imports
from src.sc_py_bridge.get_vbp_chart_data import GetVbpData

def main():
    """Main function to demonstrate fetching VBP chart data."""
    vbp_chart_data_processor = GetVbpData()
    vbp_chart_data_df = vbp_chart_data_processor.get_vbp_chart_data()
    print(vbp_chart_data_df.tail())

if __name__ == "__main__":
    main()
