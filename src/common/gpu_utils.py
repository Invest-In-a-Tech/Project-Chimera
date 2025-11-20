#!/usr/bin/env python3
"""
GPU Utilities - Device Configuration and Management for PyTorch

This module provides utilities for configuring and managing GPU devices in PyTorch,
including automatic device selection, device validation, and tensor/model transfer.
It ensures consistent GPU usage across the entire application while gracefully
falling back to CPU when GPU is unavailable.

Key Features:
- Automatic device detection (GPU/CPU)
- Manual device override via environment variables
- Device validation and availability checking
- Convenient helper functions for tensor/model device transfer
- Comprehensive logging for device status

Usage:
    # Basic usage - automatic device selection
    from src.common.gpu_utils import get_device, move_to_device
    
    device = get_device()  # Returns 'cuda' or 'cpu'
    model = move_to_device(model, device)  # Move model to device
    
    # Environment variable override
    # Set CUDA_VISIBLE_DEVICES="" to force CPU
    # Set CUDA_VISIBLE_DEVICES="0" to use GPU 0

Environment Variables:
    CUDA_VISIBLE_DEVICES: Control GPU visibility (empty string forces CPU)
    
Example:
    >>> from src.common.gpu_utils import get_device, move_to_device
    >>> device = get_device()
    >>> print(f"Using device: {device}")
    Using device: cuda
    >>> 
    >>> # Move model to GPU
    >>> model = MyModel()
    >>> model = move_to_device(model, device)
    >>> 
    >>> # Move tensor to GPU
    >>> import torch
    >>> tensor = torch.randn(10, 5)
    >>> tensor = move_to_device(tensor, device)
"""

# Standard library imports
import logging
from typing import Union, Optional

# Third-party imports
# Import PyTorch - required for device management and GPU operations
# This is the core dependency for deep learning and GPU acceleration
import torch

# Module-level logger for GPU utilities
# Uses __name__ to create hierarchical logger (e.g., 'src.common.gpu_utils')
logger = logging.getLogger(__name__)


def get_device(force_cpu: bool = False, device_id: Optional[int] = None) -> torch.device:
    """
    Get the appropriate PyTorch device for computation (GPU or CPU).
    
    This function intelligently selects the best available device for PyTorch
    operations, with the following priority:
    1. If force_cpu=True, always returns CPU
    2. If CUDA is available and not disabled, returns GPU device
    3. Otherwise, falls back to CPU
    
    The function also logs the selected device for debugging and monitoring
    purposes, making it easy to verify GPU usage in production.
    
    Args:
        force_cpu (bool, optional): If True, forces CPU usage even if GPU is
            available. Useful for debugging or comparing GPU vs CPU performance.
            Defaults to False.
        device_id (int, optional): Specific GPU device ID to use (0, 1, 2, etc.).
            Only relevant when CUDA is available and force_cpu=False.
            If None, uses default device (usually device 0).
            Defaults to None.
    
    Returns:
        torch.device: PyTorch device object representing 'cuda' or 'cpu'.
            Can be used directly with tensor.to(device) or model.to(device).
    
    Raises:
        No exceptions are raised - function always returns a valid device
    
    Example:
        >>> # Automatic device selection
        >>> device = get_device()
        >>> print(device)
        cuda
        >>> 
        >>> # Force CPU usage
        >>> device = get_device(force_cpu=True)
        >>> print(device)
        cpu
        >>> 
        >>> # Use specific GPU
        >>> device = get_device(device_id=1)
        >>> print(device)
        cuda:1
    
    Note:
        - The function respects CUDA_VISIBLE_DEVICES environment variable
        - Logs device selection at INFO level for monitoring
        - Safe to call multiple times (idempotent)
    """
    # Check if user explicitly requested CPU mode
    # force_cpu=True bypasses all GPU detection and forces CPU usage
    # Useful for debugging, comparing performance, or when GPU memory is full
    if force_cpu:
        # Log the forced CPU selection at INFO level
        # This helps users understand why CPU is being used
        logger.info("GPU device selection: CPU (forced by user)")
        # Return CPU device object
        # torch.device('cpu') creates a device representing CPU
        return torch.device('cpu')
    # Check if CUDA (GPU) is available via PyTorch
    # torch.cuda.is_available() returns True only if:
    # - CUDA-capable GPU is present
    # - NVIDIA drivers are installed and functional
    # - CUDA runtime is accessible
    # - CUDA_VISIBLE_DEVICES is not set to empty string
    if torch.cuda.is_available():
        # Construct the device string based on whether specific GPU was requested
        # If device_id is provided, use 'cuda:N' format (e.g., 'cuda:0', 'cuda:1')
        # If device_id is None, use 'cuda' which defaults to device 0
        if device_id is not None:
            # Validate device_id is within available GPU range
            # torch.cuda.device_count() returns number of available GPUs
            device_count: int = torch.cuda.device_count()
            if device_id >= device_count:
                # Log warning if requested device doesn't exist
                # Fall back to default device instead of raising error
                logger.warning(
                    "Requested GPU device %d, but only %d device(s) available. Using default GPU.",
                    device_id,
                    device_count
                )
                # Use default device string (just 'cuda')
                device_str: str = 'cuda'
            else:
                # Construct device string with specific GPU ID
                device_str: str = f'cuda:{device_id}'
        else:
            # No specific device requested - use default
            device_str: str = 'cuda'
        # Create PyTorch device object from device string
        device: torch.device = torch.device(device_str)
        # Get GPU name for informative logging
        # This helps users verify they're using the expected GPU
        try:
            # Get the actual device index from the device object
            # If device is 'cuda', this returns 0 (default)
            # If device is 'cuda:1', this returns 1
            actual_device_id: int = device.index if device.index is not None else 0
            # Get human-readable GPU name (e.g., "NVIDIA GeForce RTX 4090")
            gpu_name: str = torch.cuda.get_device_name(actual_device_id)
            # Log successful GPU selection with device name
            logger.info("GPU device selection: %s (%s)", device_str, gpu_name)
        except (RuntimeError, OSError) as gpu_error:
            # If we can't get GPU name, log device without name
            # This shouldn't prevent GPU usage, just affects logging
            logger.info("GPU device selection: %s (name unavailable: %s)", device_str, gpu_error)
        # Return the GPU device object
        return device
    # CUDA not available - fall back to CPU
    # This happens when: no GPU, no drivers, or CUDA_VISIBLE_DEVICES=""
    # Log the fallback to CPU at INFO level
    # This helps users understand why CPU is being used
    logger.info("GPU device selection: CPU (CUDA not available)")
    # Return CPU device object
    return torch.device('cpu')


def move_to_device(
    obj: Union[torch.Tensor, torch.nn.Module],
    device: torch.device
) -> Union[torch.Tensor, torch.nn.Module]:
    """
    Move a PyTorch tensor or model to the specified device.
    
    This is a convenience wrapper around PyTorch's .to(device) method that
    provides consistent error handling and logging. It works with both
    tensors and neural network modules (models).
    
    Args:
        obj (Union[torch.Tensor, torch.nn.Module]): PyTorch tensor or model
            to move to the specified device. Can be any PyTorch object that
            supports the .to() method.
        device (torch.device): Target device ('cuda' or 'cpu'). Should be
            obtained from get_device() function.
    
    Returns:
        Union[torch.Tensor, torch.nn.Module]: The tensor or model moved to
            the specified device. Same type as input.
    
    Raises:
        RuntimeError: If device transfer fails (e.g., out of GPU memory)
        
    Example:
        >>> # Move tensor to GPU
        >>> device = get_device()
        >>> tensor = torch.randn(10, 5)
        >>> tensor = move_to_device(tensor, device)
        >>> print(tensor.device)
        cuda:0
        >>> 
        >>> # Move model to GPU
        >>> model = MyNeuralNetwork()
        >>> model = move_to_device(model, device)
        >>> next(model.parameters()).device
        cuda:0
    
    Note:
        - Moving to GPU requires sufficient GPU memory
        - Moving between GPU and CPU creates a copy
        - For models, this moves all parameters and buffers
    """
    # Move the object to the specified device
    # .to(device) returns a new tensor/model on the target device
    # For tensors: creates a copy on the new device
    # For modules: moves all parameters and buffers to the device
    try:
        # Perform the device transfer
        # This is the core operation that actually moves data/model to GPU/CPU
        return obj.to(device)
    except RuntimeError as transfer_error:
        # Catch runtime errors (usually GPU out of memory)
        # Log the error with context about what failed
        logger.error(
            "Failed to move object to device %s: %s",
            device,
            transfer_error
        )
        # Re-raise the exception so caller can handle it
        # This prevents silent failures that could cause confusion
        raise


def is_gpu_available() -> bool:
    """
    Check if GPU is available for PyTorch operations.
    
    Simple wrapper around torch.cuda.is_available() for consistent API.
    Useful for conditional logic that depends on GPU availability.
    
    Returns:
        bool: True if CUDA GPU is available, False otherwise.
    
    Example:
        >>> if is_gpu_available():
        ...     print("GPU acceleration enabled")
        ... else:
        ...     print("Running on CPU")
        GPU acceleration enabled
    """
    # Return CUDA availability status
    # torch.cuda.is_available() checks GPU, drivers, and CUDA runtime
    return torch.cuda.is_available()


def get_device_name(device: torch.device) -> str:
    """
    Get the human-readable name of a PyTorch device.
    
    Retrieves the hardware name for GPU devices (e.g., "NVIDIA GeForce RTX 4090")
    or returns "CPU" for CPU devices.
    
    Args:
        device (torch.device): PyTorch device to get name for.
    
    Returns:
        str: Human-readable device name.
    
    Example:
        >>> device = get_device()
        >>> print(get_device_name(device))
        NVIDIA GeForce RTX 4090
    """
    # Check if this is a CUDA device
    if device.type == 'cuda':
        try:
            # Get device index (0, 1, 2, etc.)
            # If device.index is None, assume device 0
            device_id: int = device.index if device.index is not None else 0
            # Get and return GPU name
            return torch.cuda.get_device_name(device_id)
        except (RuntimeError, OSError):
            # If we can't get the name, return generic string
            return f"CUDA Device {device.index if device.index is not None else 0}"
    # Not a CUDA device - return "CPU"
    return "CPU"
