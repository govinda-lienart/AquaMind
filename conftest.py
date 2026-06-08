import logging
import pytest
import subprocess

logging.basicConfig(level=logging.DEBUG, format='%(levelname)s | %(funcName)s | %(message)s')

def _kill_connections_to_test_db():
    """Kill any sleeping connections to aquamind_test so DROP DATABASE never blocks."""
    result = subprocess.run(
        ["mysql", "-u", "root", "-paquamind", "--protocol=TCP",
         "--batch", "--skip-column-names", "-e",
         "SELECT CONCAT('KILL ', id, ';') FROM information_schema.processlist WHERE db = 'aquamind_test' AND command != 'Daemon'"],
        capture_output=True, text=True
    )
    kill_stmts = result.stdout.strip()
    if kill_stmts:
        subprocess.run(["mysql", "-u", "root", "-paquamind", "--protocol=TCP", "-e", kill_stmts])

@pytest.fixture(autouse=True)
def reset_test_db():
    _kill_connections_to_test_db()
    subprocess.run(["mysql", "-u", "root", "-paquamind", "--protocol=TCP", "-e",
                    "DROP DATABASE IF EXISTS aquamind_test; CREATE DATABASE aquamind_test;"])
    subprocess.run(["mysql", "-u", "root", "-paquamind", "--protocol=TCP", "aquamind_test"],
                   stdin=open("fixtures/aquamind_schema.sql"))
    subprocess.run(["mysql", "-u", "root", "-paquamind", "--protocol=TCP", "aquamind_test"],
                   stdin=open("fixtures/fixtures.sql"))



 