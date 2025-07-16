# Deprecated Modules

This directory contains deprecated modules that are no longer used in the main codebase but are kept for reference.

## metadata.py
- **Deprecated**: This module was used to fetch PostgreSQL table metadata using psycopg2/3
- **Replaced by**: Rust implementation for metadata fetching
- **Reason**: To eliminate psycopg2 dependency and improve performance
- **Date**: 2025-07-16