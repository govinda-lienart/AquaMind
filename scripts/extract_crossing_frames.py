import os
import re
from datetime import datetime

import cv2
import yaml

from scripts.db import get_connection, register_frames


CONFIG_PATH = 'config.yaml'


def load_config():
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    c = cfg['extract_crossing_frames']
    video_name = os.path.splitext(os.path.basename(c['video_path']))[0]
    timestamp  = datetime.now().strftime('%Y%m%d_%Hh%M')
    return {
        'log_path':      c['log_path'],
        'video_path':    c['video_path'],
        'output_dir':    f"{c['output_dir']}_{video_name}_{timestamp}",
        'iou_threshold': c.get('iou_threshold', 0.4),
        'dedup_window':  c.get('dedup_window', 5),
    }


def parse_crossing_frames(log_path, iou_threshold, dedup_window):
    """Return sorted list of unique frame numbers where bbox IoU exceeds threshold."""
    pattern = re.compile(r'Overlap detected.*\[frame (\d+)\] IoU=([\d.]+)')
    frame_iou = {}
    with open(log_path) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                frame  = int(m.group(1))
                iou    = float(m.group(2))
                frame_iou[frame] = max(frame_iou.get(frame, 0), iou)
    unique     = sorted(frame_iou)
    candidates = sorted(f for f, v in frame_iou.items() if v > iou_threshold)
    deduped, last = [], -999
    for f in candidates:
        if f - last >= dedup_window:
            deduped.append(f)
            last = f
    print(f"  Found {len(unique)} overlap frames (IoU > 0)")
    print(f"  Of which {len(candidates)} have IoU > {iou_threshold} → {len(deduped)} after dedup (1 per {dedup_window}-frame window)")
    return deduped


def extract_frames(video_path, output_dir, crossing_frames):
    """Extract exactly one frame per crossing event."""
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)

    saved = 0
    for frame_num in crossing_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if ret:
            path = os.path.join(output_dir, f"frame_{frame_num:06d}.jpg")
            cv2.imwrite(path, frame)
            saved += 1

    cap.release()
    print(f"  Saved {saved} frames to {output_dir}")
    return saved


def write_sidecar(output_dir, p, frames_extracted):
    """Write extraction metadata alongside the frames so store_annotations.py can read it."""
    sidecar = {
        'frame_source':     'crossing_event',
        'video_path':       p['video_path'],
        'tracker_log_path': p['log_path'],
        'iou_threshold':    p['iou_threshold'],
        'dedup_window':     p['dedup_window'],
        'frames_extracted': frames_extracted,
        'extracted_at':     datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }
    path = os.path.join(output_dir, 'extraction_params.yaml')
    with open(path, 'w') as f:
        yaml.dump(sidecar, f, default_flow_style=False, sort_keys=False)
    print(f"  Sidecar written → {path}")


def main():
    p = load_config()
    print("=" * 50)
    print(f"  Log:    {p['log_path']}")
    print(f"  Video:  {p['video_path']}")
    print(f"  Output: {p['output_dir']}")
    print("=" * 50)

    crossing_frames = parse_crossing_frames(p['log_path'], p['iou_threshold'], p['dedup_window'])
    if not crossing_frames:
        print("  No crossings found in log. Run the tracker first.")
        return

    print(f"\n  Will extract {len(crossing_frames)} frames (1 per overlap event).")
    input("  Press Enter to proceed, Ctrl+C to cancel... ")

    saved = extract_frames(p['video_path'], p['output_dir'], crossing_frames)
    write_sidecar(p['output_dir'], p, saved)

    conn = get_connection()
    n = register_frames(conn, p['output_dir'], p['video_path'])
    conn.close()
    print(f"  Registered {n} frames in MySQL")
    print("\nDone. Import the output folder into LabelStudio.")


if __name__ == '__main__':
    main()


# ── TESTS ─────────────────────────────────────────────────────────────────────
#  pytest scripts/extract_crossing_frames.py -v -s

def test_parse_crossing_frames(tmp_path):
    print("\n*****************************************************")
    print("\n--- testing: parse_crossing_frames ---")
    log = tmp_path / "tracker.log"
    log.write_text(
        "Overlap detected: Fish 1 ↔ Fish 2 [frame 100] IoU=0.50\n"
        "Overlap detected: Fish 1 ↔ Fish 2 [frame 101] IoU=0.55\n"
        "Overlap detected: Fish 1 ↔ Fish 2 [frame 200] IoU=0.30\n"  # below threshold
        "Overlap detected: Fish 1 ↔ Fish 2 [frame 300] IoU=0.60\n"
    )
    frames = parse_crossing_frames(str(log), iou_threshold=0.4, dedup_window=5)
    assert 100 in frames
    assert 200 not in frames  # IoU=0.30 below threshold
    assert 300 in frames
    assert 101 not in frames  # deduped — within 5 frames of 100


def test_write_sidecar(tmp_path):
    print("\n*****************************************************")
    print("\n--- testing: write_sidecar ---")
    p = {
        'video_path':    'videos/IMG_0350.MOV',
        'log_path':      'logs/tracker.log',
        'iou_threshold': 0.4,
        'dedup_window':  5,
    }
    write_sidecar(str(tmp_path), p, frames_extracted=12)
    sidecar_path = tmp_path / 'extraction_params.yaml'
    assert sidecar_path.exists()
    import yaml
    data = yaml.safe_load(sidecar_path.read_text())
    assert data['frame_source']     == 'crossing_event'
    assert data['iou_threshold']    == 0.4
    assert data['frames_extracted'] == 12
