# Data Sources API Documentation

## Overview

The data sources module provides interfaces for extracting and processing Volume by Price (VBP) data from Sierra Chart. Currently implements historical data extraction with real-time subscription capabilities.

## Architecture

```
project_chimera.data_sources/
├── get_vbp_downloader.py     # Historical VBP data extraction
└── __init__.py

sc_py_bridge/
├── get_vbp_chart_data.py     # Core VBP fetcher class
└── subscribe_to_vbp_chart_data.py  # Real-time VBP subscription
```

## Classes

### `GetVbpData` (Historical Data)

**Location**: `src/project_chimera/data_sources/get_vbp_downloader.py`

Primary class for extracting historical Volume by Price data from Sierra Chart.

#### Constructor

```python
GetVbpData(
    bridge: Optional[SCBridge] = None,
    columns_to_drop: Optional[List[str]] = None,
    historical_bars: int = 1000000
)
```

**Parameters:**
- `bridge`: Optional SCBridge instance (creates new if None)
- `columns_to_drop`: Columns to remove from final DataFrame (default: ['IsBarClosed'])
- `historical_bars`: Number of historical bars to request (default: 1M)

#### Key Methods

##### `get_vbp_chart_data() -> pd.DataFrame`
Main method that fetches and processes VBP data.

**Returns:** Processed DataFrame with VBP data and market indicators

##### `fetch_vbp_chart_data() -> pd.DataFrame`
Fetches raw VBP chart data from Sierra Chart.

**Data Requested:**
- Base data: OHLCV
- Study ID 6, Subgraph 1: Relative Volume (RVOL)
- Study ID 4, Subgraphs 1-3: Today's Open/High/Low
- Study ID 9, Subgraphs 1-4: Large Trade indicators

##### `process_vbp_chart_data(df: pd.DataFrame) -> pd.DataFrame`
Processes raw chart data into analysis-friendly format.

**Processing Steps:**
1. Expands VolumeByPrice nested data into tabular format
2. Merges with bar-level OHLCV data
3. Normalizes column names to standard conventions
4. Sets DateTime as index

**Column Mapping:**
- `Last` → `Close`
- `ID6.SG1` → `RVOL` (Relative Volume)
- `ID4.SG1` → `TodayOpen`
- `ID4.SG2` → `TodayHigh`
- `ID4.SG3` → `TodayLow`
- `ID9.SG1` → `LTMaxVol` (Large Trade Max Volume)
- `ID9.SG2` → `LTTotalVol` (Large Trade Total Volume)
- `ID9.SG3` → `LTBidVol` (Large Trade Bid Volume)
- `ID9.SG4` → `LTAskVol` (Large Trade Ask Volume)

## Output Data Schema

### Final DataFrame Structure

**Index:** `DateTime` (bar timestamps)

**VBP Columns (per price level):**
- `Price`: Price level
- `BidVol`: Volume on bid side
- `AskVol`: Volume on ask side
- `TotalVolume`: Total volume at price level
- `NumOfTrades`: Number of trades at price level

**Bar-level Market Data:**
- `Open`, `High`, `Low`, `Close`: OHLC prices
- `Volume`: Total bar volume
- `RVOL`: Relative volume indicator
- `TodayOpen`, `TodayHigh`, `TodayLow`: Daily levels
- `LTMaxVol`, `LTTotalVol`, `LTBidVol`, `LTAskVol`: Large trade metrics

## Usage Examples

### Basic Historical Data Extraction

```python
from project_chimera.data_sources.get_vbp_downloader import GetVbpData

# Initialize with default settings
vbp_fetcher = GetVbpData()

# Extract VBP data
df = vbp_fetcher.get_vbp_chart_data()

# Save to CSV
df.to_csv('data/raw/dataframes/vbp_data.csv')

# Clean up
vbp_fetcher.stop_bridge()
```

### Custom Configuration

```python
from trade29.sc import SCBridge

# Use custom bridge and settings
bridge = SCBridge()
vbp_fetcher = GetVbpData(
    bridge=bridge,
    historical_bars=500000,  # Last 500k bars
    columns_to_drop=['IsBarClosed', 'unwanted_col']
)

df = vbp_fetcher.get_vbp_chart_data()
```

## Data Quality Notes

### Expected Data Volume
- **ES Futures 30-minute bars**: ~500k bars ≈ 1 year of data
- **15 years of data**: ~7.5M bars (adjust `historical_bars` accordingly)

### Data Integrity Checks
- Verify Sierra Chart connection before large extractions
- Check for missing bars during market hours
- Validate VBP data completeness (some bars may have empty VBP profiles)

### Performance Considerations
- Large historical requests (1M+ bars) may take several minutes
- Sierra Chart memory usage increases with request size
- Consider batching very large extractions

## Error Handling

### Common Issues
1. **Sierra Chart not running**: Check SC is active with data feed connected
2. **Bridge connection failed**: Verify firewall and SC bridge settings
3. **Empty VBP data**: Some bars may not have VBP profiles (normal)
4. **Memory errors**: Reduce `historical_bars` for very large requests

### Debugging
Enable logging for detailed diagnostics:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Future Enhancements

### Planned Features
- [ ] Batch processing for very large historical periods
- [ ] Data validation and quality checks
- [ ] Automatic retry logic for failed requests
- [ ] Progress indicators for long-running extractions
- [ ] Integration with data storage backends (database, parquet)

### Consolidation Notes
Currently there are duplicate `GetVbpData` classes in:
- `src/project_chimera/data_sources/get_vbp_downloader.py`
- `src/sc_py_bridge/get_vbp_chart_data.py`

**Planned**: Consolidate into single implementation in `sc_py_bridge` module.