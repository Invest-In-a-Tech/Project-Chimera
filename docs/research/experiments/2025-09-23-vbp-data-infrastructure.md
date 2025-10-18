# Experiment: VBP Data Infrastructure Setup

- Date: 2025-09-23
- Owner: Roy Williams
- Hypothesis: Volume by Price data can be reliably extracted from Sierra Chart using trade29-scpy bridge, providing foundation for microstructure analysis
- Metric(s): Successful extraction of 15+ years of ES Futures VBP data with complete price/volume breakdown
- Data: ES Futures 30-minute bars, historical data back to ~2010, Sierra Chart feed
- Method: Implement GetVbpData class with Sierra Chart bridge integration, test data extraction and processing

## Run
- Code/Notebook: `src/project_chimera/data_sources/get_vbp_downloader.py`
- Params:
  - historical_bars: 1,000,000 (default)
  - include_volume_by_price: True
  - Studies: RVOL (ID6.SG1), Today OHLC (ID4.SG1-3), Large Trades (ID9.SG1-4)
- Environment: Python 3.13, trade29-scpy==1.0.2, pandas==2.3.2

## Results
- Findings:
  - Successfully extracted VBP data with price-level volume breakdown
  - VBP profiles contain: Price, BidVol, AskVol, TotalVolume, NumOfTrades
  - Integration with OHLCV and market indicators (RVOL, Today levels)
  - Data processing pipeline converts nested VBP structure to flat DataFrame
  - Column normalization provides consistent naming conventions
  - WARNING: Identified duplicate GetVbpData classes (needs consolidation)
- Plots/Tables: CSV output saved to `data/raw/dataframes/volume_by_price_15years.csv`
- Outcome: **Accepted** - VBP data infrastructure is operational

## Notes
- Surprises: VBP data structure more complex than expected (nested arrays per bar)
- Next questions:
  - What VBP features best capture auction behavior?
  - How does VBP profile shape correlate with price movement?
  - What's the optimal way to aggregate VBP data across timeframes?
- Follow-ups:
  - Consolidate duplicate GetVbpData implementations
  - Build VBP profile analysis tools
  - Create visualization for VBP data exploration
  - Define first hypothesis for VBP-based edge discovery

## Technical Implementation Notes
- Bridge successfully connects to Sierra Chart and retrieves data
- VBP processing flattens nested structure using concat with keys
- Column mapping standardizes Sierra Chart field names
- DateTime indexing enables time-series analysis
- CSV export provides persistent storage for analysis

## Data Quality Assessment
- **Coverage**: Full historical data available
- **Completeness**: All requested fields present
- **Accuracy**: Data matches Sierra Chart display
- **Volume**: ~15 years of 1-minute ES Futures bars
- **Format**: Clean, analysis-ready DataFrame structure

## Architecture Decision
Current implementation has VBP classes in two locations:
1. `src/project_chimera/data_sources/get_vbp_downloader.py` (downloader script)
2. `src/sc_py_bridge/get_vbp_chart_data.py` (reusable class)

**Decision**: Consolidate into single `sc_py_bridge` implementation for consistency.