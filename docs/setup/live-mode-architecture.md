# Live Mode Architecture - Separation of Concerns

## Overview

The live mode has been refactored to properly separate concerns, making the codebase scalable for machine learning projects with multiple subscriptions (VBP data, account info, position info, etc.).

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Sierra Chart (SC)                         │
│  - VBP Chart Data                                             │
│  - Account Data                                               │
│  - Position Data                                              │
└────────────────────┬────────────────────────────────────────┘
                     │ DTC Protocol
                     ▼
┌─────────────────────────────────────────────────────────────┐
│         SierraChartSubscriptionManager                       │
│  - Manages all SC subscriptions                              │
│  - Routes responses by request_id                            │
│  - Lifecycle management (start/stop)                         │
└────────────────────┬────────────────────────────────────────┘
                     │ Raw Responses
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              ResponseProcessor                                │
│  - Transforms SC responses to DataFrames                     │
│  - Handles VBP/account/position formats                      │
│  - Prepares data for pipeline                                │
└────────────────────┬────────────────────────────────────────┘
                     │ Clean DataFrames
                     ▼
┌─────────────────────────────────────────────────────────────┐
│           DataPipelineRunner (LIVE mode)                     │
│  - Feature engineering                                        │
│  - Data transformations                                       │
│  - NO subscription management                                │
└────────────────────┬────────────────────────────────────────┘
                     │ Engineered Features
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                 ML Model / Trading Logic                     │
│  - Make predictions                                           │
│  - Execute trades                                             │
│  - Portfolio management                                       │
└─────────────────────────────────────────────────────────────┘
```

## Key Components

### 1. SierraChartSubscriptionManager
**Location:** `src/common/sierra_chart_manager/subscription_manager.py`

**Responsibilities:**
- Manage multiple Sierra Chart subscriptions
- Maintain registry of subscription IDs
- Route responses to handlers
- Lifecycle management (start/stop all subscriptions)

**Usage:**
```python
from src.common.sierra_chart_manager import SierraChartSubscriptionManager
from src.common.sierra_chart_manager.subscription_manager import SubscriptionType

# Initialize manager
manager = SierraChartSubscriptionManager()

# Subscribe to VBP data
vbp_id = manager.subscribe_vbp_chart_data({
    'historical_init_bars': 50,
    'on_bar_close': True
})

# Subscribe to account data (future)
# account_id = manager.subscribe_account_data()

# Subscribe to position data (future)
# position_id = manager.subscribe_position_data()

# Get responses
response = manager.get_next_response(SubscriptionType.VBP_CHART_DATA)

# Cleanup
manager.stop_all_subscriptions()
```

### 2. ResponseProcessor
**Location:** `src/common/sierra_chart_manager/response_processor.py`

**Responsibilities:**
- Transform raw SC responses to clean DataFrames
- Handle different response types (VBP, account, position)
- Normalize data formats
- Prepare data for pipeline

**Usage:**
```python
from src.common.sierra_chart_manager import ResponseProcessor

processor = ResponseProcessor()

# Process VBP response
vbp_df = processor.process_vbp_response(response)

# Process account response (future)
# account_data = processor.process_account_response(response)

# Process position response (future)
# position_data = processor.process_position_response(response)
```

### 3. DataPipelineRunner (LIVE mode)
**Location:** `src/common/data_pipeline/run_data_pipeline.py`

**Responsibilities:**
- Feature engineering on live data
- Data transformations
- Statistical calculations
- **NO subscription management** (removed)

**Usage:**
```python
from src.common.data_pipeline.run_data_pipeline import DataPipelineRunner, PipelineMode

# Pipeline accepts DataFrame only (no SC config)
config = {'df': vbp_df}
pipeline = DataPipelineRunner(config, PipelineMode.LIVE)

# Run feature engineering
features = pipeline.run_pipeline()
```

## Complete Example

```python
import logging
from src.common.sierra_chart_manager import (
    SierraChartSubscriptionManager,
    ResponseProcessor
)
from src.common.sierra_chart_manager.subscription_manager import SubscriptionType
from src.common.data_pipeline.run_data_pipeline import DataPipelineRunner, PipelineMode

logging.basicConfig(level=logging.INFO)

# 1. Setup Sierra Chart subscriptions
sc_manager = SierraChartSubscriptionManager()
response_processor = ResponseProcessor()

vbp_id = sc_manager.subscribe_vbp_chart_data({'historical_init_bars': 50})
# account_id = sc_manager.subscribe_account_data()
# position_id = sc_manager.subscribe_position_data()

# 2. Get initial data
initial_response = sc_manager.get_next_response(SubscriptionType.VBP_CHART_DATA)
initial_vbp = response_processor.process_vbp_response(initial_response)

# 3. Engineer features
pipeline = DataPipelineRunner({'df': initial_vbp}, PipelineMode.LIVE)
features = pipeline.run_pipeline()

# 4. Load ML model
# model = joblib.load('models/my_model.pkl')

# 5. Real-time processing loop
try:
    while True:
        # Get response (could be VBP, account, or position)
        response = sc_manager.get_next_response(SubscriptionType.VBP_CHART_DATA)
        request_id = vbp_id
        
        # Route based on request_id
        if request_id == vbp_id:
            # Process VBP data
            vbp_update = response_processor.process_vbp_response(response)
            
            # Engineer features
            pipeline = DataPipelineRunner({'df': vbp_update}, PipelineMode.LIVE)
            features = pipeline.run_pipeline()
            
            # Make prediction
            # prediction = model.predict(features.iloc[-1].values.reshape(1, -1))
            
            # Your trading logic here
            latest = features.iloc[-1]
            if latest.get('RVOL', 0) > 2.0:
                print("HIGH VOLUME ALERT!")
        
        # elif request_id == account_id:
        #     account_data = response_processor.process_account_response(response)
        #     # Update account state
        
        # elif request_id == position_id:
        #     position_data = response_processor.process_position_response(response)
        #     # Update position state

except KeyboardInterrupt:
    print("Stopping...")
finally:
    sc_manager.stop_all_subscriptions()
```

## Benefits of This Architecture

### 1. **Separation of Concerns**
- **SierraChartSubscriptionManager**: Knows about SC, doesn't know about pipelines
- **ResponseProcessor**: Knows about data formats, doesn't know about subscriptions
- **DataPipelineRunner**: Knows about features, doesn't know about SC
- **ML Model**: Knows about predictions, doesn't know about data sources

### 2. **Scalability**
Easy to add new subscriptions:
```python
# Add new subscription type
def subscribe_order_updates(self, config):
    # Implementation...
    pass

# Add new processor
def process_order_response(self, response):
    # Implementation...
    pass
```

### 3. **Testability**
Each component can be tested independently:
```python
# Test response processor with mock data
processor = ResponseProcessor()
mock_response = create_mock_vbp_response()
df = processor.process_vbp_response(mock_response)
assert df.shape[0] > 0
```

### 4. **Reusability**
Components can be used in different contexts:
```python
# Use subscription manager for data collection
manager = SierraChartSubscriptionManager()
# Collect data for backtesting...

# Use response processor for batch processing
processor = ResponseProcessor()
# Process historical responses...

# Use pipeline for both live and training
pipeline = DataPipelineRunner(config, PipelineMode.TRAINING)
```

### 5. **Clear Interfaces**
Each component has a clear, focused API:
```python
# Subscription Manager
manager.subscribe_vbp_chart_data(config) -> int
manager.get_next_response(type) -> Response
manager.stop_all_subscriptions() -> None

# Response Processor
processor.process_vbp_response(response) -> DataFrame
processor.process_account_response(response) -> Dict
processor.process_position_response(response) -> DataFrame

# Pipeline
pipeline.run_pipeline() -> DataFrame
pipeline.get_data_info() -> Dict
```

## Migration Guide

### Old Way (Tightly Coupled)
```python
# Pipeline knew about Sierra Chart ❌
config = {'sierra_chart_config': {...}}
pipeline = DataPipelineRunner(config, PipelineMode.LIVE)
data = pipeline.run_pipeline()
update = pipeline.get_live_update()  # Blocking call inside pipeline
pipeline.stop_live_subscription()
```

### New Way (Loosely Coupled)
```python
# Separate concerns ✅
manager = SierraChartSubscriptionManager()
processor = ResponseProcessor()

manager.subscribe_vbp_chart_data(config)
response = manager.get_next_response(SubscriptionType.VBP_CHART_DATA)
df = processor.process_vbp_response(response)

pipeline = DataPipelineRunner({'df': df}, PipelineMode.LIVE)
features = pipeline.run_pipeline()

manager.stop_all_subscriptions()
```

## Future Extensibility

### Adding Account Data Subscription
```python
# In subscription_manager.py
def subscribe_account_data(self, config):
    subscriber = SubscribeToAccountData(**config)
    self.subscriptions['account_data'] = subscriber
    self.subscription_ids['account_data'] = subscriber.account_id
    return subscriber.account_id

# In response_processor.py
def process_account_response(self, response):
    return {
        'buying_power': response.buying_power,
        'cash_balance': response.cash_balance,
        # ...
    }

# In your trading code
account_id = manager.subscribe_account_data()
response = manager.get_next_response(SubscriptionType.ACCOUNT_DATA)
account_data = processor.process_account_response(response)
```

### Adding Multiple Models
```python
# Easy to add multiple models
vbp_model = joblib.load('models/vbp_model.pkl')
momentum_model = joblib.load('models/momentum_model.pkl')
ensemble_model = joblib.load('models/ensemble_model.pkl')

# Process features through all models
vbp_pred = vbp_model.predict(features)
momentum_pred = momentum_model.predict(features)
ensemble_pred = ensemble_model.predict(features)

# Combine predictions
final_decision = voting_logic(vbp_pred, momentum_pred, ensemble_pred)
```

## See Also

- [main.py](../../main.py) - Complete live mode implementation (run with `uv run main.py process-data --mode live`)
- [src/common/sierra_chart_manager/](../../src/common/sierra_chart_manager/) - Manager and processor implementation
- [Sierra Chart Bridge Setup](sierra-chart-bridge.md) - Bridge installation and configuration
