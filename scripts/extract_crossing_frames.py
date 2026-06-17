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
        'log_path':   c['log_path'],
        'video_path': c['video_path'],
        'output_dir': f"{c['output_dir']}_{video_name}_{timestamp}",
    }


def parse_crossing_frames(log_path):
    """Return sorted list of unique frame numbers where bbox IoU > 0."""
    pattern = re.compile(r'Overlap detected.*\[frame (\d+)\] IoU=([\d.]+)')
    frame_iou = {}
    with open(log_path) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                frame  = int(m.group(1))
                iou    = float(m.group(2))
                frame_iou[frame] = max(frame_iou.get(frame, 0), iou)
    unique   = sorted(frame_iou)
    candidates = sorted(f for f, v in frame_iou.items() if v > 0.4)
    deduped, last = [], -999
    for f in candidates:
        if f - last >= 5:
            deduped.append(f)
            last = f
    print(f"  Found {len(unique)} overlap frames (IoU > 0)")
    print(f"  Of which {len(candidates)} have IoU > 0.4→ {len(deduped)} after dedup (1 per 5-frame window)")
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


def main():
    p = load_config()
    print("=" * 50)
    print(f"  Log:    {p['log_path']}")
    print(f"  Video:  {p['video_path']}")
    print(f"  Output: {p['output_dir']}")
    print("=" * 50)

    crossing_frames = parse_crossing_frames(p['log_path'])
    if not crossing_frames:
        print("  No crossings found in log. Run the tracker first.")
        return

    print(f"\n  Will extract {len(crossing_frames)} frames (1 per overlap event).")
    input("  Press Enter to proceed, Ctrl+C to cancel... ")

    extract_frames(p['video_path'], p['output_dir'], crossing_frames)

    conn = get_connection()
    n = register_frames(conn, p['output_dir'], p['video_path'])
    conn.close()
    print(f"  Registered {n} frames in MySQL")
    print("\nDone. Import the output folder into LabelStudio.")


if __name__ == '__main__':
    main()
