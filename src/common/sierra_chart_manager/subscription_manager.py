"""
Sierra Chart Subscription Manager.

This module manages multiple Sierra Chart subscriptions (VBP data, account data,
position data, etc.) and routes responses to the appropriate processors.

The manager centralizes subscription lifecycle management and ensures clean
separation of concerns between data acquisition and processing.

Classes:
    SierraChartSubscriptionManager: Central manager for all SC subscriptions

Author: Roy Williams
Version: 1.0.0
"""

import logging
from typing import Optional, Dict, Any, Callable
from enum import Enum

# Import Sierra Chart bridge components
try:
    from src.sc_py_bridge.subscribe_to_vbp_chart_data import SubscribeToVbpChartData
except ImportError:
    SubscribeToVbpChartData = None

logger = logging.getLogger(__name__)


class SubscriptionType(Enum):
    """Enumeration of available subscription types."""
    VBP_CHART_DATA = "vbp_chart_data"
    ACCOUNT_DATA = "account_data"
    POSITION_DATA = "position_data"
    # Add more as needed


class SierraChartSubscriptionManager:
    """
    Central manager for Sierra Chart subscriptions.

    This class handles creation, lifecycle management, and response routing
    for all Sierra Chart subscriptions. It maintains a registry of active
    subscriptions and their associated request IDs.

    The manager pattern keeps subscription logic separate from business logic
    and allows easy addition of new subscription types.

    Attributes:
        subscriptions (Dict[str, Any]): Registry of active subscription instances
        subscription_ids (Dict[str, int]): Mapping of subscription types to request IDs
        response_handlers (Dict[int, Callable]): Response handlers by request ID

    Example:
        >>> manager = SierraChartSubscriptionManager()
        >>>
        >>> # Subscribe to VBP data
        >>> vbp_config = {'historical_init_bars': 50, 'on_bar_close': True}
        >>> manager.subscribe_vbp_chart_data(vbp_config)
        >>>
        >>> # Register response handler
        >>> def handle_vbp(response):
        ...     df = response.as_df()
        ...     # Process df...
        >>>
        >>> manager.register_response_handler(
        ...     SubscriptionType.VBP_CHART_DATA,
        ...     handle_vbp
        ... )
        >>>
        >>> # Process responses
        >>> while True:
        ...     response = manager.get_next_response()
        ...     manager.process_response(response)
    """

    def __init__(self):
        """Initialize the subscription manager."""
        # Registry of active subscription instances
        self.subscriptions: Dict[str, Any] = {}

        # Mapping of subscription types to their request IDs
        self.subscription_ids: Dict[str, int] = {}

        # Response handlers registered for each subscription type
        self.response_handlers: Dict[int, Callable] = {}

        logger.info("Sierra Chart Subscription Manager initialized")

    def subscribe_vbp_chart_data(
        self,
        config: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Subscribe to VBP chart data from Sierra Chart.

        Args:
            config: Configuration dictionary with keys:
                - historical_init_bars (int): Initial historical bars (default: 50)
                - realtime_update_bars (int): Bars per update (default: 1)
                - on_bar_close (bool): Update on bar close vs tick (default: True)

        Returns:
            int: Request ID for this subscription

        Raises:
            ImportError: If SubscribeToVbpChartData is not available
        """
        if SubscribeToVbpChartData is None:
            raise ImportError(
                "SubscribeToVbpChartData not available. "
                "Install Sierra Chart bridge dependencies."
            )

        # Extract config with defaults
        config = config or {}
        historical_bars = config.get('historical_init_bars', 50)
        realtime_bars = config.get('realtime_update_bars', 1)
        on_bar_close = config.get('on_bar_close', True)

        logger.info(
            "Subscribing to VBP chart data - Historical: %d, "
            "Realtime: %d, On bar close: %s",
            historical_bars, realtime_bars, on_bar_close
        )

        # Create subscription instance
        subscriber = SubscribeToVbpChartData(
            historical_init_bars=historical_bars,
            realtime_update_bars=realtime_bars,
            on_bar_close=on_bar_close
        )

        # Store subscription
        subscription_type = SubscriptionType.VBP_CHART_DATA.value
        self.subscriptions[subscription_type] = subscriber
        self.subscription_ids[subscription_type] = subscriber.chart_data_id

        logger.info(
            "VBP chart data subscription created with ID: %d",
            subscriber.chart_data_id
        )

        return subscriber.chart_data_id

    def subscribe_account_data(self, config: Optional[Dict[str, Any]] = None) -> int:
        """
        Subscribe to account data from Sierra Chart.

        Args:
            config: Configuration dictionary for account subscription

        Returns:
            int: Request ID for this subscription

        Note:
            Placeholder for future implementation
        """
        raise NotImplementedError(
            "Account data subscription not yet implemented"
        )

    def subscribe_position_data(self, config: Optional[Dict[str, Any]] = None) -> int:
        """
        Subscribe to position data from Sierra Chart.

        Args:
            config: Configuration dictionary for position subscription

        Returns:
            int: Request ID for this subscription

        Note:
            Placeholder for future implementation
        """
        raise NotImplementedError(
            "Position data subscription not yet implemented"
        )

    def register_response_handler(
        self,
        subscription_type: SubscriptionType,
        handler: Callable
    ) -> None:
        """
        Register a response handler for a subscription type.

        Args:
            subscription_type: Type of subscription
            handler: Callable that processes responses for this subscription

        Example:
            >>> def handle_vbp(response):
            ...     df = response.as_df()
            ...     # Process...
            >>>
            >>> manager.register_response_handler(
            ...     SubscriptionType.VBP_CHART_DATA,
            ...     handle_vbp
            ... )
        """
        subscription_id = self.subscription_ids.get(subscription_type.value)
        if subscription_id is None:
            raise ValueError(
                f"No active subscription for type: {subscription_type.value}"
            )

        self.response_handlers[subscription_id] = handler
        logger.debug(
            "Registered handler for subscription type: %s (ID: %d)",
            subscription_type.value, subscription_id
        )

    def get_next_response(self, subscription_type: SubscriptionType) -> Any:
        """
        Get the next response from a specific subscription.

        This method blocks until a response is available from the specified
        subscription's queue.

        Args:
            subscription_type: Type of subscription to get response from

        Returns:
            Response object from Sierra Chart

        Raises:
            ValueError: If subscription type not found
        """
        subscription = self.subscriptions.get(subscription_type.value)
        if subscription is None:
            raise ValueError(
                f"No active subscription for type: {subscription_type.value}"
            )

        # VBP chart data subscription
        if subscription_type == SubscriptionType.VBP_CHART_DATA:
            return subscription.get_subscribed_vbp_chart_data()

        # Add other subscription types here as implemented
        raise NotImplementedError(
            f"Response retrieval not implemented for: {subscription_type.value}"
        )

    def process_response(self, response: Any, request_id: int) -> Any:
        """
        Process a response by routing it to the registered handler.

        Args:
            response: Response object from Sierra Chart
            request_id: Request ID to identify which subscription this is for

        Returns:
            Result from the response handler, if any

        Example:
            >>> response = manager.get_next_response(SubscriptionType.VBP_CHART_DATA)
            >>> result = manager.process_response(
            ...     response,
            ...     manager.subscription_ids['vbp_chart_data']
            ... )
        """
        handler = self.response_handlers.get(request_id)
        if handler is None:
            logger.warning(
                "No handler registered for request ID: %d",
                request_id
            )
            return None

        try:
            return handler(response)
        # pylint: disable=broad-exception-caught
        # Justification: We re-raise immediately after logging for debugging
        except Exception as e:
            logger.error(
                "Error in response handler for request ID %d: %s",
                request_id, e
            )
            raise

    def stop_subscription(self, subscription_type: SubscriptionType) -> None:
        """
        Stop a specific subscription and clean up resources.

        Args:
            subscription_type: Type of subscription to stop
        """
        subscription = self.subscriptions.get(subscription_type.value)
        if subscription is None:
            logger.debug(
                "No active subscription to stop for type: %s",
                subscription_type.value
            )
            return

        # Stop VBP chart data subscription
        if subscription_type == SubscriptionType.VBP_CHART_DATA:
            subscription.stop_bridge()

        # Remove from registries
        del self.subscriptions[subscription_type.value]
        request_id = self.subscription_ids.pop(subscription_type.value, None)
        if request_id and request_id in self.response_handlers:
            del self.response_handlers[request_id]

        logger.info("Stopped subscription: %s", subscription_type.value)

    def stop_all_subscriptions(self) -> None:
        """Stop all active subscriptions and clean up resources."""
        logger.info("Stopping all subscriptions...")

        subscription_types = list(self.subscriptions.keys())
        for sub_type_str in subscription_types:
            try:
                sub_type = SubscriptionType(sub_type_str)
                self.stop_subscription(sub_type)
            except ValueError:
                logger.warning("Unknown subscription type: %s", sub_type_str)

        logger.info("All subscriptions stopped")

    def get_subscription_info(self) -> Dict[str, Any]:
        """
        Get information about active subscriptions.

        Returns:
            Dictionary with subscription status and IDs
        """
        return {
            'active_subscriptions': list(self.subscriptions.keys()),
            'subscription_ids': self.subscription_ids.copy(),
            'registered_handlers': len(self.response_handlers)
        }
