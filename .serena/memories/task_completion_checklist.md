# Task Completion Checklist

When completing any development task in the GPU PostgreSQL Parser project, ensure you follow these steps:

## Before Marking Task as Complete

### 1. Code Quality Checks
- [ ] Run Black formatter: `black . --line-length=100`
- [ ] Sort imports with isort: `isort . --profile black --line-length=100`
- [ ] Run flake8 linter: `flake8 . --max-line-length=100 --extend-ignore=E203,W503`
- [ ] Run pre-commit hooks: `pre-commit run --all-files`

### 2. Testing
- [ ] Run relevant tests for your changes:
  - For new features: Write tests first (TDD approach)
  - For bug fixes: Add regression tests
  - Run tests: `pytest tests/` or specific test files
- [ ] Ensure all existing tests still pass
- [ ] Check test coverage if significant changes made

### 3. Documentation
- [ ] Update docstrings for new/modified functions
- [ ] Update relevant documentation in `docs/` if needed
- [ ] Add inline comments for complex logic

### 4. Rust-Specific (if applicable)
- [ ] Run `cargo fmt` for formatting
- [ ] Run `cargo clippy` for linting
- [ ] Run `cargo test` for Rust tests
- [ ] Rebuild extension if needed: `maturin develop --release`

### 5. GPU/CUDA Checks (if applicable)
- [ ] Test with different GPU memory configurations
- [ ] Verify no memory leaks (monitor with `nvidia-smi`)
- [ ] Test with both small and large datasets

### 6. Performance Considerations
- [ ] Run benchmarks if performance-critical code changed
- [ ] Compare performance before/after changes
- [ ] Document any performance impacts

### 7. Final Verification
- [ ] Code follows project conventions (see code_style_conventions.md)
- [ ] No hardcoded paths or credentials
- [ ] No debug print statements left in production code
- [ ] Error handling is appropriate and informative

### 8. Git Workflow
- [ ] Stage only relevant changes: `git add <files>`
- [ ] Write clear commit message following convention:
  - `feat:` for features
  - `fix:` for fixes
  - `test:` for tests
  - `docs:` for documentation
- [ ] Ensure commit is atomic (one logical change)

## Environment Variables to Check
```bash
# Should be set appropriately
GPUPASER_PG_DSN="dbname=postgres user=postgres host=localhost port=5432"

# Should be unset for production
unset GPUPGPARSER_TEST_MODE
unset GPUPGPARSER_DEBUG
```

## Quick Validation Commands
```bash
# Full validation suite
black . --line-length=100 --check
isort . --profile black --line-length=100 --check
flake8 . --max-line-length=100 --extend-ignore=E203,W503
pytest tests/ -x  # Stop on first failure

# If all pass, you're ready to commit!
```

## Common Issues to Watch For
- Memory leaks in GPU code
- Incorrect type mappings between PostgreSQL and Arrow
- Thread safety in parallel processing
- Proper cleanup of database connections
- Handling of NULL values and edge cases

## Notes
- Always test with both small and large datasets
- Consider multi-GPU scenarios if applicable
- Verify compatibility with different PostgreSQL versions
- Test with various data types, especially edge cases
