import pytest
import subprocess

# autouse=True means pytest runs this automatically before every test — no need to call it manually
@pytest.fixture(autouse=True)
def reset_test_db():
    # wipe aquamind_test and recreate it empty
    subprocess.run(["mysql", "-u", "root", "-paquamind", "--protocol=TCP", "-e",
                    "DROP DATABASE IF EXISTS aquamind_test; CREATE DATABASE aquamind_test;"])
    # load table structure (CREATE TABLE statements) into the fresh database
    subprocess.run(["mysql", "-u", "root", "-paquamind", "--protocol=TCP", "aquamind_test"],
                   stdin=open("fixtures/aquamind_schema.sql"))
    # load fake test rows (one video, one frame) into the tables
    subprocess.run(["mysql", "-u", "root", "-paquamind", "--protocol=TCP", "aquamind_test"],
                   stdin=open("fixtures/fixtures.sql"))
