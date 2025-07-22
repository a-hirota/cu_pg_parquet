#!/usr/bin/env python3
"""
PostgreSQL Test Database Setup Script

This script creates a test database and configures it for GPU PostgreSQL Parser testing.
"""

import logging
import os
import sys

import psycopg2
from psycopg2 import sql

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def get_connection_params():
    """Get PostgreSQL connection parameters from environment or defaults."""
    return {
        "host": os.environ.get("GPUPASER_PG_HOST", "localhost"),
        "port": os.environ.get("GPUPASER_PG_PORT", "5432"),
        "user": os.environ.get("GPUPASER_PG_USER", "postgres"),
        "password": os.environ.get("GPUPASER_PG_PASSWORD", ""),
        "dbname": "postgres",  # Connect to default database first
    }


def create_test_database(conn_params, test_db_name="gpupgparser_test"):
    """Create test database if it doesn't exist."""
    try:
        # Connect to default database
        conn = psycopg2.connect(**conn_params)
        conn.autocommit = True
        cur = conn.cursor()

        # Check if test database exists
        cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (test_db_name,))

        if cur.fetchone():
            logger.info(f"Test database '{test_db_name}' already exists")
            # Drop and recreate for clean state
            cur.execute(sql.SQL("DROP DATABASE IF EXISTS {}").format(sql.Identifier(test_db_name)))
            logger.info(f"Dropped existing test database '{test_db_name}'")

        # Create test database
        cur.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(test_db_name)))
        logger.info(f"Created test database '{test_db_name}'")

        cur.close()
        conn.close()

        return test_db_name

    except Exception as e:
        logger.error(f"Failed to create test database: {e}")
        raise


def setup_test_schema(test_db_name):
    """Set up test schema and extensions."""
    conn_params = get_connection_params()
    conn_params["dbname"] = test_db_name

    try:
        conn = psycopg2.connect(**conn_params)
        conn.autocommit = True
        cur = conn.cursor()

        # Create schema for tests
        cur.execute("CREATE SCHEMA IF NOT EXISTS test_schema")
        logger.info("Created test schema")

        # Create any necessary extensions
        # cur.execute("CREATE EXTENSION IF NOT EXISTS ...")

        cur.close()
        conn.close()

    except Exception as e:
        logger.error(f"Failed to setup test schema: {e}")
        raise


def verify_setup(test_db_name):
    """Verify the test database setup."""
    conn_params = get_connection_params()
    conn_params["dbname"] = test_db_name

    try:
        conn = psycopg2.connect(**conn_params)
        cur = conn.cursor()

        # Test connection and permissions
        cur.execute("SELECT current_database(), current_user, version()")
        db_info = cur.fetchone()
        logger.info(f"Connected to: {db_info[0]} as {db_info[1]}")
        logger.info(f"PostgreSQL version: {db_info[2].split(',')[0]}")

        # Check COPY permissions
        cur.execute(
            """
            SELECT has_database_privilege(current_user, current_database(), 'CREATE')
               AND has_database_privilege(current_user, current_database(), 'CONNECT')
        """
        )
        has_perms = cur.fetchone()[0]

        if not has_perms:
            logger.warning("User may not have sufficient permissions for COPY operations")
        else:
            logger.info("User has required permissions")

        cur.close()
        conn.close()

        return True

    except Exception as e:
        logger.error(f"Failed to verify setup: {e}")
        return False


def main():
    """Main setup function."""
    logger.info("Starting PostgreSQL test database setup...")

    try:
        # Get connection parameters
        conn_params = get_connection_params()
        logger.info(f"Connecting to PostgreSQL at {conn_params['host']}:{conn_params['port']}")

        # Create test database
        test_db_name = os.environ.get("GPUPASER_TEST_DB", "gpupgparser_test")
        create_test_database(conn_params, test_db_name)

        # Setup schema
        setup_test_schema(test_db_name)

        # Verify setup
        if verify_setup(test_db_name):
            logger.info("Test database setup completed successfully!")
            logger.info(f"Test database name: {test_db_name}")

            # Write connection info for tests
            config_path = os.path.join(os.path.dirname(__file__), "..", "test_config.py")
            with open(config_path, "w") as f:
                f.write(
                    f"""# Auto-generated test configuration
TEST_DB_NAME = '{test_db_name}'
TEST_DB_HOST = '{conn_params['host']}'
TEST_DB_PORT = '{conn_params['port']}'
TEST_DB_USER = '{conn_params['user']}'
TEST_DB_PASSWORD = '{conn_params.get('password', '')}'

def get_test_dsn():
    return f"host={{TEST_DB_HOST}} port={{TEST_DB_PORT}} dbname={{TEST_DB_NAME}} user={{TEST_DB_USER}}"
"""
                )
            logger.info(f"Test configuration written to {config_path}")

            return 0
        else:
            logger.error("Test database setup verification failed")
            return 1

    except Exception as e:
        logger.error(f"Setup failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
