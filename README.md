# Project Chimera — The Pursuit of Statistical Edge

*"In the pursuit of excellence, there is no finish line."*
—Roy Williams

---

## Why This Exists
I now realize that finding a statistical edge in trading is a data science problem. If I'm wrong, please correct me—I'm a student first, expert second.

It makes no sense to predict the **closing price** of an asset—the close is just the final print of an **auction** and won't make you money in a game dominated by **probabilities and statistical edges**. That's why I treat this as a **data science problem**: before solving anything, define the problem through three lenses—**data** (what's actually there), **domain** (how the auction works), and **business** (what outcome matters).

I've also learned I was asking the wrong questions (a skill I learned while studying Thinking Strategies) when building tech to "become profitable." Now I'm focused on the right ones: **Where are the repeatable patterns that create high-probability setups, and how do we validate them?**

**My two edges:**
1) I can build and evaluate AI/ML tools.
2) I understand market mechanics and microstructure.

Project Chimera combines these edges to find patterns worth trading. It starts **not** as a fully automated system (and may never be). It's a research pipeline where **human + AI** collaborate to surface and test edges.

## What I'm Bringing
- 5+ years trading U.S. equities/futures
- AI/ML engineering and systems research
- Working knowledge of market microstructure and mechanics

## Goal
Discover repeatable, high-probability setups in ES Futures by combining:
1) structured market data,
2) microstructure context, and
3) practical ML/AI tooling.

## Approach (Keep It Simple)
- **Frame the problem** in business, domain, and data terms.
- **Extract features** that reflect auction behavior (not just predicting a close).
- **Test hypotheses** fast with small, falsifiable experiments.
- **Keep the human in the loop**—AI assists decision-making instead of replacing it (at least early on).

## What This Is (for now)
- A research pipeline for **edge discovery**.
- Human-guided trading supported by analytics and lightweight models.

## What This Isn't (yet)
- A fully automated trading system.

---

## Current Status & Setup

### Data Infrastructure
- **Volume by Price (VBP) Data Pipeline**: Complete Sierra Chart bridge integration
- **Real-time & Historical Data**: 15+ years of ES Futures VBP data extraction
- **Data Format**: Structured pandas DataFrames with price, volume, bid/ask breakdown

### Core Components

#### 1. Data Sources (`src/project_chimera/data_sources/`)
- `get_vbp_downloader.py`: Historical VBP data extraction and CSV export
- Fetches OHLCV data + Volume by Price profiles
- Includes relative volume (RVOL) and large trade indicators

#### 2. Sierra Chart Bridge (`src/sc_py_bridge/`)
- `get_vbp_chart_data.py`: Core VBP data fetcher class
- `subscribe_to_vbp_chart_data.py`: Real-time VBP data subscription
- Bridge to Sierra Chart via `trade29-scpy` library

#### 3. Data Pipeline (`src/common/data_pipeline/`)
- `run_data_pipeline.py`: Main data processing orchestrator with comprehensive logging
- `dataframe_processor.py`: CSV data processing and transformation utilities
- Handles both live data streams and file-based processing workflows
- Provides structured logging and error handling for robust pipeline execution

#### 4. Feature Engineering (`src/project_chimera/features/`)
- **Status**: Ready for implementation
- **Planned**: VBP profile analysis, auction behavior features, volume distribution metrics

### Tech Stack
- **Python 3.13** with `uv` package management
- **pandas** for data manipulation and analysis
- **trade29-scpy** for Sierra Chart data bridge
- **Sierra Chart** as primary data source for ES Futures

### Dependencies
```
trade29-scpy==1.0.2    # Sierra Chart bridge
pandas==2.3.2          # Data analysis
fastavro==1.12.0        # Serialization
pywin32==311           # Windows integration
```

---

## Quick Start

### Prerequisites
1. **Sierra Chart** (version 2626+) with ES Futures data feed
2. **Python 3.13+**
3. **uv** package manager (recommended) or pip
4. **Windows 10/11** (required for Sierra Chart bridge)

### Installation

#### 1. Install Sierra Chart Bridge
The project requires the `trade29-scpy` bridge for data extraction:

```powershell
# Clone the repository first
git clone <your-repo-url>
cd Project-Chimera

# Install the Sierra Chart bridge
pip install trade29_scpy-1.0.2.tar.gz

# Verify bridge installation
python -c "import trade29; print('Bridge installed')"
```

**Important**: You must also install the Sierra Chart DLL component. See [Sierra Chart Bridge Setup](docs/setup/sierra-chart-bridge.md) for complete instructions.

#### 2. Install Project Dependencies
```powershell
# Install dependencies with uv (recommended)
uv sync

# Or with pip
pip install -r requirements.txt
```

### Extract Historical VBP Data
```powershell
# Option 1: Use the CLI interface (recommended)
uv run main.py download-vbp

# Option 2: Run the VBP data downloader directly
uv run src\project_chimera\data_sources\get_vbp_downloader.py

# Data saved to: data/raw/dataframes/volume_by_price_data.csv
```

### Process Data Through the Pipeline
```powershell
# Process existing VBP data through the data pipeline
uv run main.py process-data

# Process a specific input file
uv run main.py process-data --input "data/raw/dataframes/your_data.csv"

# Process and save to a specific output file
uv run main.py process-data --output "data/processed/processed_data.csv"

# Process with both custom input and output
uv run main.py process-data --input "data/custom.csv" --output "data/results.csv"
```

### Check Project Status
```powershell
# View project status and available data files
uv run main.py status
```

### CLI Help
```powershell
# View all available commands
uv run main.py --help

# Get help for a specific command
uv run main.py process-data --help
```

### Verify Setup
The data extraction will create CSV files containing:
- **DateTime**: Bar timestamps (indexed)
- **OHLCV Data**: Open, High, Low, Close, Volume for each time period
- **VBP Profile Data**:
  - **Price**: Each price level in the VBP profile
  - **BidVol/AskVol**: Volume breakdown by market side
  - **TotalVolume**: Total volume at each price
  - **NumOfTrades**: Number of trades at each price
- **Market Indicators**: RVOL, Today's range data, large trade indicators

**Example verification:**
```powershell
# Check what data files are available
uv run main.py status

# Process and preview data structure
uv run main.py process-data --input "data/raw/dataframes/1.volume_by_price_15years.csv"
```

This will display data shape, columns, date range, and preview the first 5 rows.

---

## CLI Commands Reference

Project Chimera provides a comprehensive command-line interface for common operations:

### Available Commands

#### `download-vbp`
Download historical Volume by Price data from Sierra Chart:
```powershell
uv run main.py download-vbp [--output PATH]
```
- `--output, -o`: Custom output CSV file path (optional)
- Default output: `data/raw/dataframes/volume_by_price_data.csv`

#### `process-data`
Process data through the data pipeline with multiple modes:
```powershell
uv run main.py process-data [--input PATH] [--output PATH] [--mode MODE]
```
- `--input, -i`: Input CSV file path (optional, defaults to latest VBP data)
- `--output, -o`: Output CSV file path (optional, displays to console if not specified)
- `--mode, -m`: Pipeline mode - `training`, `live`, or `auto` (default: `auto`)

**Pipeline Modes:**
- **training**: File-based processing for model training and backtesting
- **live**: Process real-time data from external sources (see [Live Mode Architecture](docs/setup/live-mode-architecture.md))
- **auto**: Automatically detect mode based on configuration

**What it does:**
- Loads data using the DataPipelineRunner
- Applies data processing and transformation
- Provides comprehensive logging and error handling
- Returns processed DataFrame ready for analysis

**Live Mode Example:**
```powershell
# Start real-time Sierra Chart data streaming
uv run main.py process-data --mode live

# This will:
# 1. Connect to Sierra Chart
# 2. Fetch initial historical data
# 3. Display real-time updates as they arrive
# 4. Optionally save to file with --output
```

#### `status`
Display project status and available data files:
```powershell
uv run main.py status
```
Shows:
- Available data files and their sizes
- Documentation structure
- Suggested next steps

### Example Workflows

#### Complete Data Processing Workflow
```powershell
# 1. Download fresh VBP data
uv run main.py download-vbp

# 2. Process the data through the pipeline
uv run main.py process-data

# 3. Process and save for further analysis
uv run main.py process-data --output "data/processed/analyzed_vbp.csv"

# 4. Check project status
uv run main.py status
```

#### Working with Existing Data
```powershell
# Process existing data file
uv run main.py process-data --input "data/raw/dataframes/1.volume_by_price_15years.csv"

# Process and save to specific location
uv run main.py process-data \
    --input "data/raw/dataframes/1.volume_by_price_15years.csv" \
    --output "data/processed/historical_analysis.csv"
```

---

### Data Pipeline Architecture

The data pipeline provides a robust, modular approach to processing financial data with support for both historical and real-time data sources.

### Pipeline Modes

The pipeline supports three operational modes:

1. **Training Mode**: File-based processing for ML model training, backtesting, and historical analysis
   - Reads historical data from CSV files
   - Reproducible results with static datasets
   - Ideal for model development and strategy testing

2. **Live Mode**: Real-time data processing with external data sources
   - Modular architecture with separated subscription management
   - Clean integration with Sierra Chart via SierraChartSubscriptionManager
   - Pipeline focuses on feature engineering only
   - Production-ready for active trading systems
   - See [Live Mode Architecture](docs/setup/live-mode-architecture.md) for detailed usage

3. **Auto Mode**: Intelligent mode detection based on configuration
   - Automatically selects training or live mode
   - Simplifies deployment and configuration

### Key Components

1. **DataPipelineRunner**: Feature engineering pipeline
   - Processes DataFrame inputs for ML model consumption
   - Supports multiple pipeline modes (training, live, auto)
   - Comprehensive logging and error handling
   - Type-safe configuration management
   - Works with external data sources (does NOT manage subscriptions)

2. **DataFrameProcessor**: Specialized CSV data processor
   - Financial data-specific transformations
   - Time-series handling and filtering
   - Market hours and datetime processing

3. **SubscribeToVbpChartData**: Real-time Sierra Chart integration
   - Persistent subscription to VBP data streams
   - Configurable update frequency (bar-close or tick-by-tick)
   - Proper resource management and cleanup

### Pipeline Features

- **Flexible Input Sources**: File paths, DataFrames, or Sierra Chart streams
- **Multi-Mode Support**: Training, live, or auto-detection
- **Robust Error Handling**: Detailed error messages and graceful failure
- **Comprehensive Logging**: Full audit trail of processing steps
- **Type Safety**: Full type hints and validation
- **Modular Design**: Easy to extend and customize

### Usage in Code

**Training Mode (Historical Data):**
```python
from src.common.data_pipeline.run_data_pipeline import DataPipelineRunner, PipelineMode

# Configure for file processing
config = {'file_path': 'data/raw/dataframes/market_data.csv'}
pipeline = DataPipelineRunner(config, PipelineMode.TRAINING)

# Run the pipeline
processed_df = pipeline.run_pipeline()

# Get processing information
info = pipeline.get_data_info()
print(f"Processed {info['shape'][0]} rows from {info['data_source']} source")
```

**Live Mode (Real-Time Sierra Chart):**
```python
from src.common.sierra_chart_manager import SierraChartSubscriptionManager, ResponseProcessor
from src.common.data_pipeline.run_data_pipeline import DataPipelineRunner, PipelineMode

# Initialize components with separated concerns
manager = SierraChartSubscriptionManager()
processor = ResponseProcessor()
pipeline = DataPipelineRunner({}, PipelineMode.LIVE)

try:
    # Subscribe to Sierra Chart VBP data
    request_id = manager.subscribe_vbp_chart_data(
        historical_bars=50,
        realtime_bars=1,
        on_bar_close=True
    )
    
    # Process real-time updates
    while True:
        response = manager.get_next_response()  # Blocks until new data
        
        # Transform raw response to DataFrame
        df = processor.process_vbp_response(response)
        
        # Engineer features through pipeline
        config = {'df': df}
        processed_df = pipeline.run_pipeline(config)
        
        print(f"New bar @ {processed_df.index[-1]}: Close={processed_df['Close'].iloc[-1]}")
        
        # Your ML model inference here
        
except KeyboardInterrupt:
    print("Stopping...")
finally:
    manager.stop_all_subscriptions()  # Always clean up
```

For complete live mode documentation with architecture diagrams and examples, see [docs/setup/live-mode-architecture.md](docs/setup/live-mode-architecture.md).

---

## Project Structure

```
Project-Chimera/
├── src/
│   ├── project_chimera/          # Main research modules
│   │   ├── data_sources/         # Data extraction & preprocessing
│   │   └── features/             # Feature engineering (planned)
│   ├── common/                   # Shared utilities
│   │   ├── data_pipeline/        # Data processing pipeline
│   │   │   ├── run_data_pipeline.py    # Main pipeline orchestrator
│   │   │   └── dataframe_processor.py  # Data processing utilities
│   │   ├── sierra_chart_manager/ # Sierra Chart subscription management
│   │   │   ├── subscription_manager.py # Manages SC subscriptions
│   │   │   └── response_processor.py   # Processes SC responses
│   │   ├── sequential_data_processor/  # Granular row processing
│   │   └── market_data_processor/      # Market data utilities
│   └── sc_py_bridge/             # Sierra Chart integration
│       ├── get_vbp_chart_data.py     # Historical data fetcher
│       └── subscribe_to_vbp_chart_data.py  # Real-time subscriber
├── data/
│   ├── raw/dataframes/           # Extracted datasets
│   └── processed/                # Pipeline-processed data (created as needed)
├── docs/
│   ├── logbook/                  # Daily research logs
│   ├── setup/                    # Setup guides (environment, Sierra Chart, live mode)
│   └── templates/                # Experiment templates
├── notebooks/                    # Jupyter analysis notebooks
└── main.py                       # CLI entry point (includes live mode)
```

---

## Troubleshooting

### Common Issues

#### "Default VBP data file not found"
```
ERROR - Default VBP data file not found: data\raw\dataframes\volume_by_price_data.csv
INFO - Run 'uv run main.py download-vbp' first, or specify --input path
```

**Solutions:**
1. Download VBP data first: `uv run main.py download-vbp`
2. Use existing data file: `uv run main.py process-data --input "data/raw/dataframes/1.volume_by_price_15years.csv"`
3. Check available files: `uv run main.py status`

#### "Cannot import DataPipelineRunner class"
```
ERROR - Cannot import DataPipelineRunner class
```

**Solutions:**
1. Ensure you're in the correct directory: `cd Project-Chimera`
2. Install dependencies: `uv sync`
3. Verify Python path includes src directory

#### File Path Issues on Windows
- Use forward slashes or escape backslashes in file paths
- Wrap paths with spaces in quotes: `"data/my file.csv"`

### Getting Help

```powershell
# View all available commands
uv run main.py --help

# Get specific command help
uv run main.py process-data --help
uv run main.py download-vbp --help
```

---

## Research Methodology

This project follows a **systematic research approach**:

### 1. Hypothesis-Driven Experiments
- Each research question becomes a documented experiment
- Clear metrics for success/failure
- Reproducible methodology

### 2. Documentation Structure
- **Daily Logbook**: `docs/logbook/` - Research progress and decisions
- **Experiment Templates**: `docs/templates/experiment.md` - Structured experiment format
- **Findings**: Consolidated insights and discoveries

### 3. Data-First Approach
- Start with market microstructure data (VBP profiles)
- Build features that reflect auction behavior
- Test hypotheses with statistical rigor

---

## Documentation Philosophy

### Research Journal Format (Recommended)

This project uses a **research journal/lab notebook approach** rather than traditional software documentation because:

1. **Iterative Discovery**: Research findings build incrementally
2. **Hypothesis Testing**: Each experiment needs clear documentation
3. **Thought Process**: Decision-making rationale is as important as results
4. **Reproducibility**: Future researchers (including future you) need context

### Recommended Documentation Structure

```
docs/
├── setup/                      # Installation & configuration
│   ├── environment.md
│   ├── sierra-chart-bridge.md
│   └── dependencies.md
├── research/                   # Research documentation
│   ├── experiments/           # Individual experiment logs
│   ├── findings/              # Key insights & discoveries
│   └── methodology.md         # Research approach
├── api/                       # Code documentation
│   ├── data-sources.md
│   ├── features.md
│   └── bridges.md
├── logbook/                   # Daily progress (current)
└── assets/                    # Plots, diagrams, screenshots
```

### How to Document Your Thought Process

1. **Daily Logbook** (`docs/logbook/YYYY-MM-DD.md`):
   - What you worked on
   - Decisions made and why
   - Questions that arose
   - Next steps

2. **Experiment Documentation** (`docs/research/experiments/`):
   - Use your existing template in `docs/templates/experiment.md`
   - Hypothesis → Method → Results → Insights
   - Include code paths, parameters, and data versions

3. **Code Documentation**:
   - Docstrings explain *what* and *how*
   - Comments explain *why* (especially for domain-specific decisions)
   - README files for each major module

---

## How I'll Work
- Ship small iterations, measure, and refine.
- Document assumptions, data, tests, and results—wins and losses.
- Keep the human in the loop—AI assists decision-making rather than replacing it.

## Commitment
I'll clearly document my thought process and findings as the project evolves, treating this as a research journal where both successes and failures contribute to understanding.

---

*This project represents an ongoing research effort into quantitative trading edge discovery. The focus is on methodological rigor, reproducible research, and maintaining the human trader's intuition while leveraging data science and ML tooling.*