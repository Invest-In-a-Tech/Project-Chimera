# Research Methodology

## Core Philosophy

This project treats trading edge discovery as a **data science research problem** with the following principles:

### 1. Hypothesis-Driven Research
- Every investigation starts with a clear, testable hypothesis
- Define success metrics upfront
- Document both positive and negative results

### 2. Three-Lens Problem Definition
For every research question, examine through:
- **Data Lens**: What information is actually available?
- **Domain Lens**: How do market mechanics work?
- **Business Lens**: What outcome creates trading value?

### 3. Systematic Experimentation
- Small, focused experiments with clear scope
- Reproducible methodology
- Statistical rigor in validation

## Research Workflow

### Phase 1: Problem Framing
1. **Observation**: Notice a pattern or ask a question
2. **Hypothesis**: Form a testable prediction
3. **Scope**: Define data requirements and success metrics
4. **Documentation**: Create experiment document

### Phase 2: Data Exploration
1. **Collection**: Gather relevant historical data
2. **Quality Check**: Validate data completeness and accuracy
3. **Initial Analysis**: Exploratory data analysis
4. **Feature Engineering**: Create relevant indicators

### Phase 3: Testing
1. **Implementation**: Build test framework
2. **Execution**: Run experiments with statistical controls
3. **Analysis**: Evaluate results against hypothesis
4. **Iteration**: Refine based on findings

### Phase 4: Documentation
1. **Results**: Document findings, both positive and negative
2. **Insights**: Extract actionable intelligence
3. **Next Steps**: Identify follow-up research questions
4. **Archive**: Store for future reference

## Documentation Standards

### Experiment Records
Use the template in `docs/templates/experiment.md` for all formal experiments.

Required elements:
- Clear hypothesis statement
- Success/failure metrics
- Data sources and time periods
- Methodology steps
- Results with statistical significance
- Conclusions and next steps

### Daily Logbook
Track daily progress in `docs/logbook/YYYY-MM-DD.md`:
- What you worked on
- Decisions made and rationale
- Questions that emerged
- Obstacles encountered
- Next day's priorities

### Code Documentation
- **Docstrings**: Explain what functions do and their parameters
- **Comments**: Explain *why* decisions were made, especially domain-specific ones
- **Type Hints**: Use comprehensive type annotations
- **Examples**: Include usage examples in docstrings

## Quality Standards

### Statistical Rigor
- Use appropriate sample sizes
- Apply proper statistical tests
- Account for multiple testing corrections
- Document confidence intervals
- Validate out-of-sample when possible

### Reproducibility
- Version control all code and major data
- Document software versions and dependencies
- Include random seeds for stochastic processes
- Provide clear instructions for replication

### Domain Knowledge Integration
- Validate findings against market microstructure theory
- Consider market regime changes
- Account for trading costs and implementation constraints
- Separate statistical significance from economic significance

## Research Areas

### Current Focus: Volume by Price Analysis
1. **VBP Profile Characteristics**: Shape, volume distribution, concentration
2. **Auction Behavior**: Value area formation, acceptance/rejection levels
3. **Temporal Patterns**: Intraday, daily, weekly cycles in VBP data
4. **Predictive Power**: VBP features for directional forecasting

### Future Research Directions
1. **Multi-timeframe Analysis**: VBP patterns across different timeframes
2. **Market Microstructure**: Order flow, large trade impact
3. **Regime Detection**: Identifying changing market conditions
4. **Risk Management**: Position sizing based on VBP insights