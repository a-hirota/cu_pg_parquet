"""
Pytest configuration and fixtures for GPU PostgreSQL Parser tests.
"""

import atexit
import logging
import os
import sys

import psycopg2
import pytest

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Suppress Numba CUDA logging errors at exit
def cleanup_numba_logging():
    """Disable Numba CUDA logging to prevent errors during Python exit."""
    try:
        # Get Numba logger and disable it
        numba_logger = logging.getLogger("numba.cuda.cudadrv.driver")
        numba_logger.disabled = True
        numba_logger.handlers.clear()

        # Clean up CuPy resources first
        try:
            import cupy as cp

            # Synchronize all streams
            cp.cuda.Stream.null.synchronize()
            # Free memory pools
            mempool = cp.get_default_memory_pool()
            mempool.free_all_blocks()
            pinned_mempool = cp.get_default_pinned_memory_pool()
            pinned_mempool.free_all_blocks()
        except:
            pass

        # Also try to clean up any CUDA contexts
        try:
            import numba.cuda

            numba.cuda.close()
        except:
            pass
    except:
        pass


# Register cleanup function
atexit.register(cleanup_numba_logging)

# Also set Numba environment variables
os.environ.setdefault("NUMBA_CUDA_LOG_LEVEL", "ERROR")
os.environ.setdefault("NUMBA_DISABLE_PERFORMANCE_WARNINGS", "1")

# Suppress CuPy warnings
os.environ.setdefault("CUPY_CACHE_CLEAR_ON_DEVICE_DESTRUCTION", "1")
os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")


@pytest.fixture(scope="session")
def test_db_setup():
    """Set up test database once per test session."""
    # Run database setup
    from tests.setup.create_test_db import main as setup_db

    result = setup_db()
    if result != 0:
        pytest.exit("Failed to set up test database")

    yield

    # Cleanup can be added here if needed


@pytest.fixture(scope="session")
def test_db_connection(test_db_setup):
    """Provide test database connection for the session."""
    try:
        from tests.test_config import get_test_dsn

        conn = psycopg2.connect(get_test_dsn())
        yield conn
        conn.close()
    except ImportError:
        pytest.skip("Test database not configured. Run setup/create_test_db.py first.")
    except Exception as e:
        pytest.exit(f"Failed to connect to test database: {e}")


@pytest.fixture
def db_connection(test_db_connection):
    """Provide a fresh database connection for each test."""
    # Create a new connection from the same DSN
    from tests.test_config import get_test_dsn

    conn = psycopg2.connect(get_test_dsn())
    yield conn
    # Rollback any uncommitted changes
    conn.rollback()
    conn.close()


@pytest.fixture
def temp_output_dir(tmp_path):
    """Provide temporary directory for test outputs."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return output_dir


@pytest.fixture(scope="session")
def gpu_available():
    """Check if GPU is available for testing."""
    try:
        import cupy as cp

        cp.cuda.Device(0).compute_capability
        return True
    except Exception:
        return False


@pytest.fixture
def basic_test_table(db_connection):
    """Create and populate a basic test table."""
    from tests.setup.generate_test_data import create_basic_test_table

    table_name = create_basic_test_table(db_connection, rows=100)
    yield table_name
    # Cleanup
    cur = db_connection.cursor()
    cur.execute(f"DROP TABLE IF EXISTS {table_name}")
    db_connection.commit()


# Markers for test categorization
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "gpu: marks tests that require GPU")
    config.addinivalue_line("markers", "integration: marks integration tests")
    config.addinivalue_line("markers", "slow: marks slow tests")
    config.addinivalue_line("markers", "datatypes: marks data type specific tests")


# Skip GPU tests if no GPU available
def pytest_collection_modifyitems(config, items):
    """Skip GPU tests if no GPU is available."""
    try:
        import cupy as cp

        cp.cuda.Device(0).compute_capability
        gpu_available = True
    except Exception:
        gpu_available = False

    if not gpu_available:
        skip_gpu = pytest.mark.skip(reason="GPU not available")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)


def pytest_sessionfinish(session, exitstatus):
    """Clean up at the end of the test session."""
    # Force cleanup of Numba CUDA resources
    cleanup_numba_logging()

    # Additional cleanup for logging handlers
    logging.shutdown()

    # Force garbage collection
    import gc

    gc.collect()

    # Explicitly disable CuPy module unloading warnings
    import warnings

    warnings.filterwarnings("ignore", category=Warning, module="cupy")
