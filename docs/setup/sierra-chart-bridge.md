# Sierra Chart Bridge Setup

This guide covers the complete installation and configuration of the `trade29-scpy` bridge that enables Python to communicate with Sierra Chart for VBP data extraction.

## Overview

The Sierra Chart bridge consists of two components:
1. **Python Library** (`trade29_scpy-1.0.2.tar.gz`) - Provides Python API
2. **Sierra Chart DLL** - Custom study that handles data communication

Both components must be installed for the bridge to function properly.

## Requirements

- **Windows**: 10/11 recommended
- **Sierra Chart**: Version 2626 or higher
- **Python**: 3.10+ (ES Microstructure Research uses 3.13+)
- **Data Feed**: Active market data connection in Sierra Chart

## Installation Process

### Step 1: Install Python Library

The Python library is included in the project as `trade29_scpy-1.0.2.tar.gz`.

```powershell
# Navigate to project root
cd "d:\Workbooks\PRISM\Systems\ES Microstructure Research"

# Install the bridge library
pip install trade29_scpy-1.0.2.tar.gz

# Verify installation
python -c "import trade29; print('Bridge library installed successfully')"
```

**Expected Output:**
```
Bridge library installed successfully
```

### Step 2: Install Sierra Chart DLL

#### A. Locate Sierra Chart Data Folder
1. Open Sierra Chart
2. Go to `Global Settings > General Settings > Data Files Folder`
3. Note the path (typically `C:\SierraChart\Data`)

#### B. Download and Install DLL
1. Visit the [Trade29 Downloads](https://drive.google.com/drive/folders/1FHiWCoHNNu09bkVjMt6oPI9AFe1TJwso)
2. Download the latest DLL file for your version
3. Copy the DLL to your Sierra Chart `Data` folder
4. Restart Sierra Chart completely

#### C. Verify DLL Installation
1. In Sierra Chart, open any chart
2. Right-click chart → `Studies > Add Custom Study`
3. Look for "Trade29" or "SC-Py" in the study list
4. If present, the DLL is installed correctly

![Study List Example](https://artemis-docs.trade29.com/_images/studylist.png)

### Step 3: Configure Sierra Chart

#### Enable External Connections
1. Go to `Global Settings > General Settings`
2. Find network/connection settings
3. Ensure external API connections are allowed
4. Note any port configurations (default should work)

#### Set Up Data Feed
Ensure you have:
- Active market data subscription
- ES Futures data available
- Real-time or historical data access

## Testing the Installation

### Quick Connection Test
```python
from trade29.sc import SCBridge

# Test bridge initialization
try:
    bridge = SCBridge()
    print("Bridge connection successful!")
    bridge.stop()
except Exception as e:
    print(f"Bridge connection failed: {e}")
```

### VBP Data Test
```python
# Test VBP data extraction
from src.es_microstructure_research.data_sources.get_vbp_downloader import GetVbpData

try:
    vbp_fetcher = GetVbpData(historical_bars=10)  # Small test
    df = vbp_fetcher.get_vbp_chart_data()
    print(f"Successfully extracted {len(df)} rows of VBP data")
    vbp_fetcher.stop_bridge()
except Exception as e:
    print(f"VBP data extraction failed: {e}")
```

## Troubleshooting

### Common Issues

#### 1. "Cannot import trade29" Error
- **Cause**: Python library not installed correctly
- **Solution**: Reinstall with `pip install trade29_scpy-1.0.2.tar.gz --force-reinstall`

#### 2. "Bridge connection failed" Error
- **Cause**: Sierra Chart DLL not installed or SC not running
- **Solution**:
  - Ensure Sierra Chart is running
  - Verify DLL is in correct Data folder
  - Restart Sierra Chart after DLL installation

#### 3. "No data received" Error
- **Cause**: Data feed issues or symbol not found
- **Solution**:
  - Verify market data subscription is active
  - Check if ES Futures data is available
  - Ensure Sierra Chart can display the data normally

#### 4. Permission/Firewall Issues
- **Cause**: Windows firewall blocking connections
- **Solution**:
  - Allow Sierra Chart and Python through Windows Firewall
  - Check antivirus software isn't blocking the bridge

### Debug Mode

Enable detailed logging for troubleshooting:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Your bridge code here
```

## Bridge Architecture

### How It Works
1. **Python** calls `trade29.sc` functions
2. **Python library** communicates with Sierra Chart via the DLL
3. **Sierra Chart DLL** extracts data from SC's internal data structures
4. **Data** flows back through the bridge to Python as pandas DataFrames

### Data Flow
```
Sierra Chart Data → Custom DLL → Python Library → Your Code
                   ↑
                Bridge Components
```

## Version Compatibility

| ES Microstructure Research | trade29-scpy | Sierra Chart | Python |
|-----------------|--------------|--------------|--------|
| Current         | 1.0.2        | 2626+        | 3.10+  |

## Support Resources

- **Official Documentation**: [SC-Py Artemis Docs](https://artemis-docs.trade29.com/)
- **Downloads**: [Trade29 Google Drive](https://drive.google.com/drive/folders/1FHiWCoHNNu09bkVjMt6oPI9AFe1TJwso)
- **Getting Started**: [User Guide](https://artemis-docs.trade29.com/userguides/gettingstarted.html)

## Next Steps

After successful installation:

1. **Test the setup** using the commands above
2. **Run VBP data extraction**: `uv run main.py download-vbp`
3. **Verify data output** in `data/raw/dataframes/`
4. **Start your first experiment** using the research framework

---

*This bridge enables the core data infrastructure for ES Microstructure Research's trading research pipeline. Proper installation is essential for all VBP analysis capabilities.*