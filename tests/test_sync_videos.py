import openpyxl
from scripts.sync_videos import load_video_rows


def test_load_video_rows(tmp_path):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(['file_path', 'session_type', 'obstacles', 'fish_count', 'notes',
               'species', 'morph', 'tank_width_cm', 'tank_height_cm', 'tank_depth_cm', 'filmed_at'])
    ws.append(['videos/IMG_0350.MOV', 'behaviour', 1, 5, 'test',
               'danio_rerio', 'golden', 35.0, 21.0, 23.0, None])
    xlsx = str(tmp_path / 'test.xlsx')
    wb.save(xlsx)

    rows = load_video_rows(xlsx)
    assert len(rows) == 1
    assert rows[0]['file_path']    == 'videos/IMG_0350.MOV'
    assert rows[0]['fish_count']   == 5
    assert rows[0]['session_type'] == 'behaviour'
