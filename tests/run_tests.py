#!/usr/bin/env python
"""
Test runner that suppresses Numba CUDA logging errors.
"""

import atexit
import logging
import os
import sys
import warnings

# Set environment variables before importing anything else
os.environ["NUMBA_CUDA_LOG_LEVEL"] = "CRITICAL"
os.environ["NUMBA_DISABLE_PERFORMANCE_WARNINGS"] = "1"
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["CUPY_CACHE_CLEAR_ON_DEVICE_DESTRUCTION"] = "1"

# Disable specific loggers
logging.getLogger("numba.cuda.cudadrv.driver").disabled = True
logging.getLogger("numba").setLevel(logging.CRITICAL)

# Suppress all warnings
warnings.filterwarnings("ignore")


def cleanup_cuda():
    """Clean up CUDA resources before exit."""
    try:
        import cupy as cp

        cp.cuda.Stream.null.synchronize()
        mempool = cp.get_default_memory_pool()
        mempool.free_all_blocks()
    except:
        pass

    try:
        from numba import cuda

        cuda.close()
    except:
        pass


# Register cleanup
atexit.register(cleanup_cuda)

# Run pytest
import pytest

if __name__ == "__main__":
    # Run tests
    exit_code = pytest.main(sys.argv[1:])

    # Force cleanup before exit
    cleanup_cuda()

    # Suppress stderr for exit
    if exit_code == 0:
        # Tests passed, suppress exit errors
        sys.stderr = open(os.devnull, "w")

    sys.exit(exit_code)
