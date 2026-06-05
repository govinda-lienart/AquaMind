import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))
import subprocess
from db import get_connection

def test_annotations_inserted():
    # run the script against aquamind_test
    subprocess.run(["python", "scripts/store_annotations.py"])

    with get_connection() as conn:
        cursor = conn.cursor()

        # verify 2 annotation rows were inserted (one bbox-only, one bbox+keypoint)
        cursor.execute("SELECT COUNT(*) FROM annotations")
        assert cursor.fetchone()[0] == 2

        # verify 1 keypoint row was inserted (only danio_rerio with 8 values gets one)
        cursor.execute("SELECT COUNT(*) FROM keypoints")
        assert cursor.fetchone()[0] == 1
