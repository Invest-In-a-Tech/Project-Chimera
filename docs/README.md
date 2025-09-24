# Project Documentation Index

## Getting Started
- **[Main README](../README.md)** - Project overview, philosophy, and current status
- **[Environment Setup](setup/environment.md)** - Installation and configuration guide
- **[Sierra Chart Bridge Setup](setup/sierra-chart-bridge.md)** - Complete bridge installation guide
- **[Project Structure](#project-structure)** - How the codebase is organized

## Research Documentation

### Methodology & Approach
- **[Research Methodology](research/methodology.md)** - Systematic approach to edge discovery
- **[Experiment Template](research/experiments/template.md)** - Standard format for research experiments

### Active Experiments
- **[VBP Data Infrastructure](research/experiments/2025-09-23-vbp-data-infrastructure.md)** - Sierra Chart bridge setup and validation

### Key Findings
- **[Findings](research/findings/)** - Consolidated insights and discoveries *(Coming Soon)*

## Technical Documentation

### API Reference
- **[Data Sources](api/data-sources.md)** - VBP data extraction and processing classes
- **[Features](api/features.md)** - Feature engineering modules *(Coming Soon)*
- **[Bridges](api/bridges.md)** - Sierra Chart integration layer *(Coming Soon)*

### Architecture
- **[System Architecture](assets/architecture/)** - High-level system design *(Coming Soon)*
- **[Data Flow](assets/architecture/)** - How data moves through the system *(Coming Soon)*

## Daily Progress
- **[Research Logbook](logbook/)** - Daily progress and decision log
  - [2025-09-23](logbook/2025-09-23.md) - Documentation setup and VBP downloader

## Project Structure

```
Project-Chimera/
├── src/
│   ├── project_chimera/          # Main research modules
│   │   ├── data_sources/         # Data extraction & preprocessing
│   │   └── features/             # Feature engineering (planned)
│   └── sc_py_bridge/             # Sierra Chart integration
├── data/
│   └── raw/dataframes/           # Extracted datasets
├── docs/                         # All documentation (this folder)
│   ├── setup/                    # Installation guides
│   ├── research/                 # Research methodology & experiments
│   ├── api/                      # Code documentation
│   ├── logbook/                  # Daily progress logs
│   └── assets/                   # Diagrams, plots, screenshots
├── notebooks/                    # Jupyter analysis notebooks
└── README.md                     # Main project documentation
```

## Documentation Standards

### For Researchers
- **Experiments**: Use the [template](research/experiments/template.md) for all formal experiments
- **Daily Logs**: Record progress in `logbook/YYYY-MM-DD.md`
- **Findings**: Document key insights in `research/findings/`

### For Developers
- **Code**: Include comprehensive docstrings and type hints
- **APIs**: Document interfaces in `api/` folder
- **Changes**: Update relevant docs when modifying code

### For Analysis
- **Notebooks**: Save analysis notebooks with clear documentation
- **Plots**: Store visualizations in `assets/experiments/`
- **Data**: Document data sources and transformations

## Quick Navigation

| I want to... | Go to... |
|---------------|----------|
| Set up the project | [Environment Setup](setup/environment.md) |
| Understand the research approach | [Research Methodology](research/methodology.md) |
| Run VBP data extraction | [Data Sources API](api/data-sources.md) |
| Create a new experiment | [Experiment Template](research/experiments/template.md) |
| Check daily progress | [Research Logbook](logbook/) |
| Understand the codebase | [API Documentation](api/) |

---

*This documentation follows a research journal approach, emphasizing both the discovery process and technical implementation. Both successes and failures are documented as part of the learning process.*