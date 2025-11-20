#!/usr/bin/env python3
"""
GPU Diagnostics Utility - Comprehensive PyTorch/CUDA Environment Diagnostics

This module provides verbose diagnostic information for troubleshooting PyTorch
and CUDA configuration issues. It goes beyond basic availability checks to provide:
- Python interpreter path and version information
- PyTorch version and build configuration
- CUDA version that PyTorch was compiled against
- GPU device count and availability status
- Individual GPU device information (index, name, properties)
- Driver and runtime compatibility verification

Use this diagnostic tool when:
- gpu_check.py succeeds but you need more detailed information
- gpu_check.py fails and you need to troubleshoot driver/build mismatches
- Setting up a new environment and verifying PyTorch installation
- Debugging CUDA runtime errors or version conflicts
- Confirming you're using the correct virtual environment

Usage:
    uv run gpu_diag.py           # Run as standalone script
    uv run main.py diagnose-gpu  # Run via CLI command

Example Output:
    python executable: C:\\Users\\...\\venv\\Scripts\\python.exe
    torch.__version__: 2.1.0+cu121
    torch.version.cuda: 12.1
    torch.cuda.is_available(): True
    torch.cuda.device_count(): 1
    current_device: 0
    device_name: NVIDIA GeForce RTX 4090
"""

# Standard library imports
import sys  # Required for sys.executable to display Python interpreter path
from typing import Optional  # For type hints on potentially None values

# Third-party imports
# Import PyTorch - required for GPU/CUDA detection and device information
# This is the core dependency for deep learning and GPU acceleration
import torch


def diagnose_gpu_environment() -> None:
    """
    Print comprehensive PyTorch/CUDA diagnostics for the local environment.

    This function performs an exhaustive diagnostic check of the PyTorch and
    CUDA environment, printing detailed information about:
    1. Python interpreter location (to verify virtual environment)
    2. PyTorch version (to check compatibility)
    3. CUDA build version (to verify PyTorch CUDA support)
    4. CUDA availability status (basic functionality check)
    5. GPU device count (number of available GPUs)
    6. Current device index and name (active GPU information)

    The function uses defensive error handling to continue diagnostics even
    when individual checks fail, providing maximum information for troubleshooting.

    Returns:
        None: Outputs diagnostic information directly to console

    Raises:
        No exceptions are raised - all errors are caught and reported

    Example:
        >>> diagnose_gpu_environment()
        python executable: /path/to/venv/bin/python
        torch.__version__: 2.1.0+cu121
        torch.version.cuda: 12.1
        torch.cuda.is_available(): True
        torch.cuda.device_count(): 2
        current_device: 0
        device_name: NVIDIA GeForce RTX 4090
    """
    # Step 1: Display Python interpreter path
    # sys.executable contains the absolute path to the Python interpreter
    # This is critical for verifying you're in the correct virtual environment
    # Helps diagnose issues where wrong Python/packages are being used
    # Common mistake: running system Python instead of venv Python
    print('python executable:', sys.executable)

    # Step 2: Display installed PyTorch version
    # torch.__version__ returns a string like "2.1.0+cu121" or "2.1.0+cpu"
    # The "+cu121" suffix indicates CUDA 12.1 build
    # The "+cpu" suffix indicates CPU-only build (no CUDA support)
    # This helps verify you installed the correct PyTorch variant
    print('torch.__version__:', torch.__version__)

    # Step 3: Display the CUDA version PyTorch was compiled against
    # Use defensive getattr pattern to avoid static analysis warnings
    # First getattr retrieves torch.version object (or None if missing)
    # Second getattr retrieves the cuda attribute (or None if not present)
    # Returns None for CPU-only PyTorch builds
    # Returns string like "12.1" for CUDA-enabled builds
    cuda_version: Optional[str] = getattr(
        getattr(torch, "version", None),
        "cuda",
        None
    )
    # Print the CUDA version string (or None for CPU builds)
    # This is crucial for diagnosing driver/PyTorch version mismatches
    # NVIDIA drivers must be >= the CUDA version shown here
    print('torch.version.cuda:', cuda_version)

    # Step 4: Check high-level CUDA availability
    # torch.cuda.is_available() performs comprehensive checks:
    # - Is CUDA-capable hardware present?
    # - Are NVIDIA drivers installed and functional?
    # - Is the CUDA runtime library accessible?
    # - Can PyTorch initialize CUDA context?
    # Returns True only if all checks pass
    is_available: bool = torch.cuda.is_available()
    # Print availability status - this is the first critical diagnostic
    # If False, subsequent GPU operations will fail
    print('torch.cuda.is_available():', is_available)
    # Step 5: Attempt detailed GPU device queries
    # Wrap in try-except to handle runtime errors gracefully
    # Even if availability check passed, device queries can fail due to:
    # - Driver state changes between checks
    # - Insufficient permissions
    # - GPU hardware errors
    # - CUDA runtime initialization failures
    try:
        # Query the number of CUDA-capable devices visible to PyTorch
        # torch.cuda.device_count() returns an integer >= 0
        # Returns 0 if: no GPUs, drivers missing, or CUDA unavailable
        # Multi-GPU systems will return 2, 3, 4, etc.
        device_count: int = torch.cuda.device_count()
        # Print device count for user verification
        # Helps confirm PyTorch can see all expected GPUs
        # Common issue: PCIe errors causing GPUs to disappear
        print('torch.cuda.device_count():', device_count)
        # Step 6: If GPUs are present, query current device details
        # Only proceed if device_count > 0 to avoid errors
        # device_count == 0 means no GPUs available for queries
        if device_count > 0:
            # Get the index of the currently selected CUDA device
            # torch.cuda.current_device() returns 0-based integer index
            # By default, PyTorch selects device 0 unless explicitly changed
            # Returns the device that new tensors will be allocated on
            current_device_idx: int = torch.cuda.current_device()
            # Print current device index for user confirmation
            # Helps verify which GPU is active in multi-GPU systems
            print('current_device:', current_device_idx)
            # Get the human-readable name of device 0
            # torch.cuda.get_device_name(0) returns string like "NVIDIA GeForce RTX 4090"
            # We query device 0 specifically as it's typically the primary GPU
            # In multi-GPU setups, you can query other devices (1, 2, etc.)
            device_name: str = torch.cuda.get_device_name(0)
            # Print device name for easy identification
            # Users can verify this matches their expected GPU hardware
            print('device_name:', device_name)
    # Catch specific exceptions that commonly occur during CUDA probing
    # RuntimeError: CUDA runtime errors (driver issues, initialization failures)
    # OSError: Operating system level errors (missing libraries, permissions)
    # By catching these specifically, we avoid hiding unrelated bugs
    except (RuntimeError, OSError) as cuda_probe_error:
        # Print error message with context for troubleshooting
        # This is informational only - doesn't crash the diagnostic script
        # Helps users identify specific driver or configuration issues
        # Common errors: "CUDA driver version insufficient", "no CUDA-capable device"
        print('cuda probe error:', cuda_probe_error)


# Standard Python idiom to check if script is being run directly
# __name__ == "__main__" is True when script is executed directly
# __name__ == "module_name" when imported as a module
# This prevents code execution when gpu_diag is imported in main.py
if __name__ == "__main__":
    # Call the main diagnostic function to run comprehensive checks
    # This executes the complete GPU environment diagnostic workflow
    diagnose_gpu_environment()
