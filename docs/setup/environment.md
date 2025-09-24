# Environment Setup

## Prerequisites

### 1. Sierra Chart
- **Version**: Latest stable release
- **Data Feed**: ES Futures data (CME Group)
- **Configuration**: Ensure VBP studies are enabled
- **Bridge**: Must allow external connections for `trade29-scpy`

### 2. Python Environment
- **Python**: 3.13+
- **Package Manager**: `uv` (recommended) or `pip`
- **OS**: Windows (required for `pywin32` dependency)

## Installation Steps

### Option 1: Using uv (Recommended)
```powershell
# Install uv if not already installed
pip install uv

# Clone and setup
git clone <repo-url>
cd Project-Chimera
uv sync
```

### Option 2: Using pip
```powershell
git clone <repo-url>
cd Project-Chimera
pip install -r requirements.txt
```

## Verification

Test your setup by running the VBP data extractor:
```powershell
uv run src\project_chimera\data_sources\get_vbp_downloader.py
```

Expected output: CSV file created in `data/raw/dataframes/`

## Troubleshooting

### Sierra Chart Connection Issues
- Verify Sierra Chart is running
- Check firewall settings
- Ensure `trade29-scpy` bridge is properly configured

### Python Environment Issues  
- Confirm Python 3.13+ is installed
- On Windows, ensure `pywin32` is properly installed
- Try reinstalling dependencies with `--force-reinstall`