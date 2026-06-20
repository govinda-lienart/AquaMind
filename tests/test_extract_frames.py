import re
from scripts.extract_frames import build_path_storage_frames, ensure_unique_constraint, frames_already_extracted, main


def test_main(db_conn):
    main(db_conn, video_path='videos/IMG_0350.MOV', frames_dir='frames')

def test_ensure_unique_constraint(db_conn):
    cursor = db_conn.cursor()
    ensure_unique_constraint(cursor)
    ensure_unique_constraint(cursor)

def test_frames_already_extracted(db_conn):
    cursor = db_conn.cursor()
    assert frames_already_extracted(cursor, 9999) == False
    assert frames_already_extracted(cursor, 19)   == True

def test_build_path_storage_frames():
    result = build_path_storage_frames('videos/IMG_0350.MOV', 'frames')
    assert re.match(r'frames/frames_IMG_0350_\d{8}_\d{4}$', result), f"Unexpected path format: {result}"
