"""
Sierra Chart Manager Package.

This package handles all Sierra Chart bridge interactions and response processing,
keeping the concerns separate from the data pipeline and ML components.

Modules:
    subscription_manager: Manages multiple Sierra Chart subscriptions
    response_processor: Processes responses from different subscription types
"""

from .subscription_manager import SierraChartSubscriptionManager
from .response_processor import ResponseProcessor

__all__ = [
    'SierraChartSubscriptionManager',
    'ResponseProcessor',
]
