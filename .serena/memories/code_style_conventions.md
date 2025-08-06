# Code Style and Conventions for GPU PostgreSQL Parser

## Python Code Style

### General Guidelines
- **Line Length**: Maximum 100 characters
- **Indentation**: 4 spaces (no tabs)
- **String Quotes**: Use double quotes for strings, triple quotes for docstrings
- **Imports**: Grouped and sorted using isort (standard library, third-party, local)

### Formatting Tools
- **Black**: Automatic code formatting with line-length=100
- **isort**: Import sorting with profile=black
- **flake8**: Linting with max-line-length=100, ignoring E203,W503

### Naming Conventions
- **Classes**: PascalCase (e.g., `DirectProcessor`, `PostgresToCudfConverter`)
- **Functions/Methods**: snake_case (e.g., `convert_postgres_to_parquet_format`)
- **Constants**: UPPER_SNAKE_CASE (e.g., `PG_OID_TO_ARROW`)
- **Private Methods**: Leading underscore (e.g., `_internal_method`)

### Type Hints
- Use type hints for function parameters and return values
- Import types from `typing` module
- Example:
```python
from typing import Optional, List, Dict, Tuple

def process_data(
    input_data: bytes,
    columns: List[ColumnMeta],
    options: Optional[Dict[str, Any]] = None
) -> Tuple[cudf.DataFrame, int]:
```

### Docstrings
- Use Google-style or NumPy-style docstrings
- Include description, Args, Returns, and Raises sections
- Japanese comments are acceptable for complex logic
- Example:
```python
def function_name(param1: str, param2: int) -> bool:
    """
    Brief description of the function.

    Args:
        param1: Description of param1
        param2: Description of param2

    Returns:
        Description of return value

    Raises:
        ValueError: When invalid input is provided
    """
```

### File Organization
- Module-level docstring at the top (can be in Japanese)
- Imports grouped: standard library, third-party, local
- Constants after imports
- Classes and functions follow
- `__all__` export list at the end if needed

## Rust Code Style

### General Guidelines
- Follow Rust standard formatting (rustfmt)
- Use `cargo fmt` before committing
- Use `cargo clippy` for additional linting

### Naming Conventions
- **Structs/Enums**: PascalCase
- **Functions/Methods**: snake_case
- **Constants**: UPPER_SNAKE_CASE
- **Modules**: snake_case

## CUDA Kernel Conventions

### Numba CUDA
- Kernel functions decorated with `@cuda.jit`
- Use explicit block and thread indexing
- Handle boundary conditions carefully
- Example:
```python
@cuda.jit
def parse_kernel(input_data, output_array, n_rows):
    tid = cuda.grid(1)
    if tid < n_rows:
        # Process row
```

### Memory Management
- Use RMM (Rapids Memory Manager) when available
- Explicitly free GPU memory when needed
- Monitor GPU memory usage during development

## Testing Conventions

### Test Structure
- Test files prefixed with `test_` or suffixed with `_test.py`
- Test classes start with `Test`
- Test functions start with `test_`
- Use pytest fixtures for setup/teardown

### Test Markers
- `@pytest.mark.gpu`: Tests requiring GPU
- `@pytest.mark.integration`: Integration tests
- `@pytest.mark.slow`: Long-running tests
- `@pytest.mark.datatypes`: Data type specific tests
- `@pytest.mark.e2e`: End-to-end tests

### Assertions
- Use descriptive assertion messages
- Test both success and failure cases
- Include edge cases and boundary conditions

## Documentation
- Keep documentation up-to-date with code changes
- Document complex algorithms and GPU kernels thoroughly
- Include examples in docstrings where helpful
- Japanese documentation is acceptable for internal notes

## Version Control
- Commit messages follow conventional commits format:
  - `feat:` for new features
  - `fix:` for bug fixes
  - `docs:` for documentation
  - `test:` for test changes
  - `refactor:` for code refactoring
- Keep commits atomic and focused
- Write clear, descriptive commit messages
