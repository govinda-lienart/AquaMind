import yaml
from scripts.extract_crossing_frames import parse_crossing_frames, write_sidecar


def test_parse_crossing_frames(tmp_path):
    log = tmp_path / 'tracker.log'
    log.write_text(
        "Overlap detected: Fish 1 ↔ Fish 2 [frame 100] IoU=0.50\n"
        "Overlap detected: Fish 1 ↔ Fish 2 [frame 101] IoU=0.55\n"
        "Overlap detected: Fish 1 ↔ Fish 2 [frame 200] IoU=0.30\n"
        "Overlap detected: Fish 1 ↔ Fish 2 [frame 300] IoU=0.60\n"
    )
    frames = parse_crossing_frames(str(log), iou_threshold=0.4, dedup_window=5)
    assert 100 in frames
    assert 200 not in frames  # IoU=0.30 below threshold
    assert 300 in frames
    assert 101 not in frames  # deduped — within 5 frames of 100

def test_write_sidecar(tmp_path):
    p = {
        'video_path':    'videos/IMG_0350.MOV',
        'log_path':      'logs/tracker.log',
        'iou_threshold': 0.4,
        'dedup_window':  5,
    }
    write_sidecar(str(tmp_path), p, frames_extracted=12)
    sidecar_path = tmp_path / 'extraction_params.yaml'
    assert sidecar_path.exists()
    data = yaml.safe_load(sidecar_path.read_text())
    assert data['frame_source']     == 'crossing_event'
    assert data['iou_threshold']    == 0.4
    assert data['frames_extracted'] == 12
