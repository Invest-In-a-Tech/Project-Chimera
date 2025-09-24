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

#### 3. Feature Engineering (`src/project_chimera/features/`)
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
1. **Sierra Chart** installed and configured with ES Futures data feed
2. **Python 3.13+** 
3. **uv** package manager (recommended) or pip

### Installation
```powershell
# Clone the repository
git clone <your-repo-url>
cd Project-Chimera

# Install dependencies with uv (recommended)
uv sync

# Or with pip
pip install -r requirements.txt
```

### Extract Historical VBP Data
```powershell
# Run the VBP data downloader (direct script)
uv run src\project_chimera\data_sources\get_vbp_downloader.py

# Or use the CLI interface
uv run main.py download-vbp

# Data saved to: data/raw/dataframes/volume_by_price_15years.csv
```

### Verify Setup
The script will create a CSV file containing:
- **DateTime**: Bar timestamps
- **Price**: Each price level in the VBP profile  
- **BidVol/AskVol**: Volume breakdown by market side
- **TotalVolume**: Total volume at each price
- **NumOfTrades**: Number of trades at each price
- **Market indicators**: OHLC, RVOL, Today's range data

---

## Project Structure

```
Project-Chimera/
├── src/
│   ├── project_chimera/          # Main research modules
│   │   ├── data_sources/         # Data extraction & preprocessing
│   │   └── features/             # Feature engineering (planned)
│   └── sc_py_bridge/             # Sierra Chart integration
│       ├── get_vbp_chart_data.py     # Historical data fetcher
│       └── subscribe_to_vbp_chart_data.py  # Real-time subscriber
├── data/
│   └── raw/dataframes/           # Extracted datasets
├── docs/
│   ├── logbook/                  # Daily research logs  
│   └── templates/                # Experiment templates
├── notebooks/                    # Jupyter analysis notebooks
└── main.py                       # CLI entry point (planned)
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