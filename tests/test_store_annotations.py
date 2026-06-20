from scripts.store_annotations import parse_frame_number, parse_annotation_line, main


def test_parse_frame_number():
    assert parse_frame_number('e6d83681-frame_360.txt') == 360
    assert parse_frame_number('abc12345-frame_0.txt')   == 0

def test_parse_annotation_line():
    bbox = parse_annotation_line(['0', '0.5', '0.4', '0.1', '0.08'])
    assert bbox['class_id'] == 0
    assert bbox['x_center'] == 0.5
    assert parse_annotation_line(['0', '0.5']) is None

def test_main(db_conn):
    main(db_conn, labels_path='fixtures/labels', frames_folder='frames/frames_IMG_0350_20260101_2000',
         video_name='IMG_0350', frame_source='regular', notes='test import')
    cursor = db_conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM annotations WHERE annotation_set_id = 2")
    assert cursor.fetchone()[0] == 1
