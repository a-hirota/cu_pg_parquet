# Auto-generated test configuration
TEST_DB_NAME = "gpupgparser_test"
TEST_DB_HOST = "localhost"
TEST_DB_PORT = "5432"
TEST_DB_USER = "postgres"
TEST_DB_PASSWORD = ""


def get_test_dsn():
    return f"host={TEST_DB_HOST} port={TEST_DB_PORT} dbname={TEST_DB_NAME} user={TEST_DB_USER}"
