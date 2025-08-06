# Suggested Commands for GPU PostgreSQL Parser Development

## Essential Environment Setup
```bash
# Activate conda environment
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate cudf_dev

# Set PostgreSQL connection
export GPUPASER_PG_DSN="dbname=postgres user=postgres host=localhost port=5432"

# Enable test mode (for debugging)
export GPUPGPARSER_TEST_MODE=1
```

## Python Development Commands

### Main Entry Points
```bash
# Run the main parser
python cu_pg_parquet.py

# Run with specific table and options
python -m gpupaser.main --table your_table --limit 10000 --parquet output.parquet

# Run simple test
python simple_test.py
```

### Testing Commands
```bash
# Run all tests
pytest tests/

# Run specific test category
pytest tests/e2e/                    # End-to-end tests
pytest tests/integration/             # Integration tests

# Run with specific markers
pytest tests/ -m "not gpu"           # Skip GPU tests
pytest tests/ -m "not slow"          # Skip slow tests

# Run specific test file
pytest tests/e2e/test_rust_extraction.py

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run test runner script
python tests/run_all_tests.py
```

### Code Quality Commands
```bash
# Format code with Black
black . --line-length=100

# Sort imports
isort . --profile black --line-length=100

# Run flake8 linter
flake8 . --max-line-length=100 --extend-ignore=E203,W503

# Run pre-commit hooks
pre-commit run --all-files

# Install pre-commit hooks
pre-commit install
```

## Rust Development Commands

### Building Rust Extensions
```bash
# Development build (Python extension)
cd rust
maturin develop

# Release build (Python extension)
cd rust
maturin develop --release

# Build standalone Rust binaries
cd rust_pg_binary_extractor
cargo build --release

# Run Rust tests
cargo test
cargo test --release
```

### Rust Environment Variables
```bash
export RUST_LOG=info
export RUST_PARALLEL_CONNECTIONS=16
```

## Benchmarking Commands
```bash
# Run compression benchmarks
python benchmarks/compression_experiment.py

# Run GPU parallel benchmarks
python benchmarks/gpu_parallel_benchmark.py

# Run GPU pipeline benchmarks
python benchmarks/gpu_pipeline_benchmark.py
```

## Git Commands (Common Workflow)
```bash
# Check status
git status

# Stage changes
git add .

# Commit with message
git commit -m "feat: your feature description"

# Push to remote
git push origin your-branch
```

## System Commands (Linux)
```bash
# List files
ls -la

# Navigate directories
cd /path/to/directory

# Search for files
find . -name "*.py"

# Search in files
grep -r "search_term" .

# Monitor GPU usage
nvidia-smi

# Check running processes
ps aux | grep python

# Kill process
pkill -f "python script_name.py"
```

## Quick Debugging
```bash
# Run with debug output
export GPUPGPARSER_DEBUG=1
python cu_pg_parquet.py

# Run with Python debugger
python -m pdb cu_pg_parquet.py

# Check CUDA availability
python -c "import cupy; print(cupy.cuda.runtime.getDeviceCount())"
```
