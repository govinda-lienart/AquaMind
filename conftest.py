import os
import pytest
import subprocess
from dotenv import load_dotenv
from scripts.logger import setup_logging

load_dotenv(".env.test", override=True)
TEST_DB = os.getenv("DB_NAME")

setup_logging()

def _kill_connections_to_test_db():
    """Kill any sleeping connections to TEST_DB so DROP DATABASE never blocks."""
    result = subprocess.run(
        ["mysql", "-u", "root", "-paquamind", "--protocol=TCP",
         "--batch", "--skip-column-names", "-e",
         f"SELECT CONCAT('KILL ', id, ';') FROM information_schema.processlist WHERE db = '{TEST_DB}' AND command != 'Daemon'"],
        capture_output=True, text=True
    )
    kill_stmts = result.stdout.strip()
    if kill_stmts:
        subprocess.run(["mysql", "-u", "root", "-paquamind", "--protocol=TCP", "-e", kill_stmts])

@pytest.fixture(autouse=True)
def reset_test_db():
    _kill_connections_to_test_db()
    subprocess.run(["mysql", "-u", "root", "-paquamind", "--protocol=TCP", "-e",
                    f"DROP DATABASE IF EXISTS {TEST_DB}; CREATE DATABASE {TEST_DB};"])
    subprocess.run(["mysql", "-u", "root", "-paquamind", "--protocol=TCP", TEST_DB],
                   stdin=open("fixtures/aquamind_schema.sql"))
    subprocess.run(["mysql", "-u", "root", "-paquamind", "--protocol=TCP", TEST_DB],
                   stdin=open("fixtures/fixtures.sql"))


@pytest.fixture
def db_conn():
    from scripts.db import get_connection
    conn = get_connection()
    yield conn
    conn.close()
