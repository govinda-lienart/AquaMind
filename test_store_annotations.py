import subprocess

def test_annotations_inserted(db_conn):
    subprocess.run(["python", "scripts/store_annotations.py"])

    cursor = db_conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM annotations")
    assert cursor.fetchone()[0] == 2

    cursor.execute("SELECT COUNT(*) FROM keypoints")
    assert cursor.fetchone()[0] == 1
