#!/usr/bin/env python3
"""
GPU Check Utility - Verify PyTorch CUDA/GPU Availability and Functionality

This module provides a comprehensive check of PyTorch's GPU capabilities including:
- CUDA availability detection
- GPU device information (name, CUDA version)
- Tensor allocation testing on GPU
- Error handling for common CUDA runtime issues

The implementation is intentionally defensive to avoid static analysis warnings
while providing robust error handling for runtime CUDA operations.

Usage:
    uv run gpu_check.py          # Run as standalone script
    uv run main.py check-gpu     # Run via CLI command

Example Output:
    GPU is available: True
    PyTorch CUDA version: 12.1
    GPU Name: NVIDIA GeForce RTX 4090
    
    Testing tensor allocation on GPU...
    Test tensor successfully created on GPU:
    tensor([[0.1234, 0.5678, 0.9012],
            ...])
"""

# Standard library imports
from typing import Optional

# Third-party imports
# Import PyTorch - required for GPU/CUDA detection and tensor operations
# This is the core dependency for deep learning and GPU acceleration
import torch


def check_gpu_availability() -> None:
    """
    Check PyTorch GPU availability and print comprehensive GPU/CUDA details.

    This function performs a complete GPU diagnostic workflow:
    1. Checks if CUDA is available via PyTorch
    2. If available, retrieves CUDA version and GPU device information
    3. Tests GPU functionality by allocating a test tensor
    4. Handles common runtime errors gracefully with informative messages

    The function uses defensive programming patterns to avoid static analysis
    warnings while maintaining robust error handling for CUDA operations.

    Returns:
        None: Outputs diagnostic information directly to console

    Raises:
        No exceptions are raised - all errors are caught and logged to console

    Example:
        >>> check_gpu_availability()
        GPU is available: True
        PyTorch CUDA version: 12.1
        GPU Name: NVIDIA GeForce RTX 4090
    """
    # Step 1: Check if PyTorch can access the GPU
    # torch.cuda.is_available() returns True if CUDA-capable GPU is detected
    # This is the primary check that determines if GPU acceleration is possible
    # Returns False if: no GPU, no CUDA drivers, or CUDA runtime not found
    is_available: bool = torch.cuda.is_available()
    # Print the availability status using f-string for clear output
    # This is the most critical piece of information for users
    print(f"GPU is available: {is_available}")

    # Step 2: If GPU is available, provide detailed information
    # Only proceed with additional checks if is_available is True
    # This avoids runtime errors when trying to access GPU on CPU-only systems
    if is_available:
        # Retrieve the CUDA version that PyTorch was compiled with
        # Use getattr with fallback to None to avoid static analysis warnings
        # First getattr gets torch.version (or None if not present)
        # Second getattr gets the cuda attribute from torch.version (or None)
        # This defensive pattern prevents AttributeError in type checkers
        cuda_version: Optional[str] = getattr(
            getattr(torch, "version", None),
            "cuda", 
            None
        )
        # Check if CUDA version was successfully retrieved
        # cuda_version will be None if attribute chain failed
        # Otherwise it contains a string like "12.1" or "11.8"
        if cuda_version:
            # Print the CUDA version for user verification
            # This helps users ensure their PyTorch installation matches their GPU
            print(f"PyTorch CUDA version: {cuda_version}")
        else:
            # Handle case where CUDA version attribute is not present
            # This can happen with custom PyTorch builds or version mismatches
            # Inform user but don't fail - GPU may still work
            print("PyTorch CUDA version: unknown (attribute not present in runtime stubs)")

        # Retrieve the name of the current GPU device
        # torch.cuda.current_device() returns the index of the current device (usually 0)
        # torch.cuda.get_device_name() converts device index to human-readable name
        # Wrap in try-except to handle potential runtime errors gracefully
        try:
            # Get current device index - defaults to device 0 if not explicitly set
            current_device_idx: int = torch.cuda.current_device()
            # Get the human-readable name of the GPU (e.g., "NVIDIA GeForce RTX 4090")
            # This makes it easy for users to verify they're using the expected GPU
            gpu_name: str = torch.cuda.get_device_name(current_device_idx)
            # Print GPU name to console for user confirmation
            print(f"GPU Name: {gpu_name}")
        # Catch specific exceptions that may occur during GPU name retrieval
        # RuntimeError: CUDA runtime errors (driver issues, initialization failures)
        # OSError: Operating system level errors (missing libraries, permissions)
        except (RuntimeError, OSError) as gpu_name_error:
            # Print error message but don't crash - this is diagnostic info only
            # User can still proceed even if GPU name can't be retrieved
            print(f"Could not get GPU name: {gpu_name_error}")

        # Step 3: Test GPU functionality by allocating a tensor
        # This verifies that we can actually use the GPU for computation
        # Print status message to inform user we're testing GPU operations
        print("\nTesting tensor allocation on GPU...")
        # Attempt to create and allocate a test tensor on the GPU
        # This is the definitive test of GPU functionality
        try:
            # Create a random 5x3 tensor and move it to the GPU
            # torch.rand(5, 3) creates a CPU tensor with random values
            # .to("cuda") moves the tensor to the GPU (device 0 by default)
            # This operation will fail if GPU is not actually functional
            test_tensor: torch.Tensor = torch.rand(5, 3).to("cuda")
            # Print success message to confirm GPU allocation worked
            print("Test tensor successfully created on GPU:")
            # Print the tensor values to show it's actually on the GPU
            # PyTorch will display the device in the tensor representation
            print(test_tensor)
        # Catch specific exceptions that may occur during tensor allocation
        # RuntimeError: CUDA errors (out of memory, invalid device, runtime failures)
        # OSError: System-level errors (driver crashes, resource limits)
        except (RuntimeError, OSError) as tensor_error:
            # Print error message with context about what failed
            # This helps users diagnose GPU issues (driver problems, memory, etc.)
            print(f"An error occurred during tensor test: {tensor_error}")


# Standard Python idiom to check if script is being run directly
# __name__ == "__main__" is True when script is executed directly
# __name__ == "module_name" when imported as a module
# This prevents code execution when gpu_check is imported in main.py
if __name__ == "__main__":
    # Call the main GPU check function to run diagnostics
    # This executes the complete GPU availability and functionality check
    check_gpu_availability()
