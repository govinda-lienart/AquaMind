"""
extract_ghost_frames.py — hard-negative mining by GHOSTING (an ACTIVE-LEARNING frame selector).

Third frame-selection strategy in AquaMind:
  1. extract_frames.py          — regular sampling (every Nth frame)         [coverage]
  2. extract_crossing_frames.py — IoU/crossing events (two boxes overlap)    [interactions]
  3. extract_ghost_frames.py    — GHOSTING (tracker lost a fish = YOLO FAILED) [the detector's own mistakes]

This one reads a tracker run's LOG, finds the frames where a fish went occlusion_lost -> occlusion_recovery
(i.e. YOLO produced no box for it), and cuts exactly those frames from the RAW video. Labelling those +
retraining YOLO closes an active-learning loop: the model picks its OWN hardest cases (plant/filter/overlap)
for the next round -> less ghosting -> fewer swaps -> cleaner fingerprints.

Reads config `extract_ghost_frames`. Output: frames/ghost_frames_<video>_<stamp>/ + extraction_params.yaml.
Usage:  python -m scripts.extract_ghost_frames
"""
import os
import json
import logging
from datetime import datetime

import cv2
import yaml
import numpy as np

from scripts.logger import setup_logging
from scripts.console import banner
from scripts.db import get_connection, register_frames   # register frames in MySQL so store_annotations can link labels (like extract_crossing_frames)

logger = logging.getLogger(__name__)


def ghosting_ranges(log_path):
    """Parse occlusion_lost -> occlusion_recovery pairs into contiguous frame RANGES where some fish had no box."""
    lost, ivals = {}, []
    for line in open(log_path):
        if '"event"' not in line:
            continue
        try:
            e = json.loads(line[line.index('{'):])
        except Exception:
            continue
        if e.get('event') == 'occlusion_lost':
            lost[e['fish_ids']] = e['frame']                     # fish dropped out here
        elif e.get('event') == 'occlusion_recovery' and e.get('fish_ids') in lost:
            ivals.append((lost.pop(e['fish_ids']), e['frame']))  # ...reappeared here -> [lost, recovered] is a ghost gap
    frames = set()
    for s, en in ivals:
        frames.update(range(s, en + 1))
    if not frames:
        return [], 0
    fs = sorted(frames); ranges, a, b = [], fs[0], fs[0]
    for f in fs[1:]:
        if f == b + 1:
            b = f
        else:
            ranges.append((a, b)); a = b = f
    ranges.append((a, b))
    return ranges, len(ivals)


def pick_frames(ranges, per_burst):
    """Take `per_burst` evenly-spaced frames from each ghost burst (0 = every ghosting frame). Cuts redundancy."""
    out = set()
    for s, e in ranges:
        if per_burst <= 0 or (e - s + 1) <= per_burst:
            out.update(range(s, e + 1))
        else:
            out.update(int(round(x)) for x in np.linspace(s, e, per_burst))
    return sorted(out)


def main():
    with open('config.yaml') as f:
        c = yaml.safe_load(f)['extract_ghost_frames']
    log_path, video_path = c['log_path'], c['video_path']
    per_burst = int(c.get('per_burst', 3))

    ranges, n_ival = ghosting_ranges(log_path)
    targets = pick_frames(ranges, per_burst)
    logger.info(f"{n_ival} ghosting intervals -> {len(ranges)} bursts -> {len(targets)} frames selected (per_burst={per_burst})")
    if not targets:
        logger.info("no ghosting found in the log — nothing to extract"); return

    base = os.path.splitext(os.path.basename(video_path))[0]
    out_dir = os.path.join(c.get('output_dir', 'frames'), f"ghost_frames_{base}_{datetime.now().strftime('%Y%m%d_%H%M')}")
    os.makedirs(out_dir, exist_ok=True)

    # sequential read (frame numbers in the log = raw-video frame indices when the tracker ran from start_seconds 0)
    cap = cv2.VideoCapture(video_path)
    target_set, hi, saved, fn = set(targets), max(targets), 0, 0
    while fn <= hi:
        ok, frame = cap.read()
        if not ok:
            break
        if fn in target_set:
            cv2.imwrite(os.path.join(out_dir, f"frame_{fn:06d}.jpg"), frame); saved += 1   # 6-digit -> matches get_frame_id
        fn += 1
    cap.release()

    # register in MySQL (INSERT IGNORE) so store_annotations can resolve frame_id — same step as extract_crossing_frames
    n_reg = None
    try:
        conn = get_connection()
        n_reg = register_frames(conn, out_dir, video_path)
        conn.commit(); conn.close()
    except Exception as ex:
        logger.warning(f"MySQL registration SKIPPED ({ex}). Is {video_path} in the videos table? Register it, then re-run.")

    with open(os.path.join(out_dir, 'extraction_params.yaml'), 'w') as f:
        yaml.dump({'frame_source': 'ghosting_event',
                   'strategy': 'ghosting (tracker lost the fish = YOLO failure) — active-learning hard mining',
                   'source_log': log_path, 'video': video_path, 'per_burst': per_burst,
                   'ghosting_intervals': n_ival, 'bursts': len(ranges), 'frames_saved': saved}, f)

    banner("✅ GHOST FRAMES SAVED")
    logger.info(f"  frames saved : {saved}   (from {len(ranges)} ghosting bursts / {n_ival} lost-fish intervals)")
    logger.info(f"  registered   : {n_reg if n_reg is not None else 'SKIPPED'} rows in MySQL frames table")
    logger.info(f"  location     : {os.path.abspath(out_dir)}")
    logger.info(f"  sidecar      : {os.path.join(out_dir, 'extraction_params.yaml')}")
    logger.info(f"  NEXT         : set  upload_labelstudio.frames_dir: {out_dir}   then  make upload-labelstudio")


if __name__ == '__main__':
    setup_logging()
    main()
