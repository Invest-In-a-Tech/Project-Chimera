# Environment Setup

## Prerequisites

### 1. Sierra Chart
- **Version**: 2626 or higher (required for bridge compatibility)
- **OS**: Windows 10/11 recommended
- **Data Feed**: ES Futures data (CME Group)
- **Configuration**: Ensure VBP studies are enabled
- **Bridge**: Must allow external connections for `trade29-scpy`

### 2. Python Environment
- **Python**: 3.10+ (3.13+ recommended for this project)
- **Package Manager**: `uv` (recommended) or `pip`
- **OS**: Windows (required for `pywin32` dependency and Sierra Chart integration)

## Installation Steps

### Step 1: Install Sierra Chart Bridge

The `trade29-scpy` bridge consists of two components that must be installed:

#### A. Install the Python Library
```powershell
# Navigate to the project directory
cd Project-Chimera

# Install the bridge from the included tarball
pip install trade29_scpy-1.0.2.tar.gz

# Verify installation
python -c "import trade29; print('Bridge installed successfully')"
```

#### B. Install Sierra Chart DLL
1. **Locate your Sierra Chart Data folder**:
   - Usually in your Sierra Chart root folder
   - Check `Global Settings > General Settings > Data Files Folder` if not found

2. **Copy the custom study DLL**:
   - Download the DLL from [Trade29 downloads](https://drive.google.com/drive/folders/1FHiWCoHNNu09bkVjMt6oPI9AFe1TJwso)
   - Copy it to your Sierra Chart `Data` folder
   - Restart Sierra Chart

3. **Verify DLL installation**:
   - The custom study should appear in Sierra Chart's study list
   - Look for "Trade29" or "SC-Py" in the custom studies

### Step 2: Install Project Dependencies

#### Option 1: Using uv (Recommended)
```powershell
# Install uv if not already installed
pip install uv

# Clone and setup project
git clone <repo-url>
cd Project-Chimera

# Install all dependencies including the bridge
uv sync
```

#### Option 2: Using pip
```powershell
# Clone project
git clone <repo-url>
cd Project-Chimera

# Install bridge first
pip install trade29_scpy-1.0.2.tar.gz

# Install other dependencies
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