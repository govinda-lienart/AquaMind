"""
tracker_basic.py — a deliberately minimal fish tracker (EXPERIMENT).
Run:  python -m scripts.tracker_basic
Output: <output_video_path with '_basic' before .mp4>
"""

import os
import cv2
import yaml
import numpy as np
from ultralytics import YOLO
from scipy.optimize import linear_sum_assignment

CONFIG_PATH   = 'config.yaml'
REACQUIRE_TAU = 15   # ghost search radius grows by one max_distance every 15 missing frames


# ── a bare-bones track: just where the fish was last seen ──────────────────────
class Track:
    def __init__(self, tid, x, y, bbox):
        self.id      = tid          # None while tentative (during calibration)
        self.x       = x
        self.y       = y
        self.bbox    = bbox
        self.hits    = 1            # how many frames it has been matched
        self.missing = 0            # consecutive frames with no detection (ghost)

    def update(self, x, y, bbox):
        self.x, self.y, self.bbox = x, y, bbox
        self.hits   += 1
        self.missing = 0

    def mark_missing(self):
        self.missing += 1


# ── the whole tracker in one function: match tracks → detections ───────────────
def associate(tracks, dets, det_pos, max_distance):
    """
    Two-pass matching that protects identities from theft and teleport-swaps.

    Pass 1 — HEALTHY tracks (matched last frame) grab their nearest detection
             within a tight radius. They get first claim, so a ghost can never
             steal a detection away from a correctly-tracked fish.

    Pass 2 — GHOST tracks (currently missing) may reclaim a LEFTOVER detection,
             with a radius that widens the longer they've been missing BUT is
             CAPPED (3x max_distance). A fish can't cross the tank in one frame,
             so a tag can never teleport onto a far, unrelated fish — it waits as
             a ghost until its own fish reappears nearby.

    Unmatched tracks are marked missing. Returns matched detection indices.
    """
    matched_dets, matched_tracks = set(), set()
    if not tracks or not len(dets):
        for t in tracks:
            t.mark_missing()
        return matched_dets

    def _match(track_idx, gate_fn):
        avail = [c for c in range(len(dets)) if c not in matched_dets]
        if not track_idx or not avail:
            return
        anchors = np.array([[tracks[i].x, tracks[i].y] for i in track_idx])
        pos     = det_pos[avail]
        cost    = np.linalg.norm(anchors[:, None] - pos[None, :], axis=2)
        for r, c in zip(*linear_sum_assignment(cost)):
            ti = track_idx[r]
            if cost[r, c] > gate_fn(tracks[ti]):
                continue
            x, y, bbox = dets[avail[c]]
            tracks[ti].update(x, y, bbox)
            matched_dets.add(avail[c])
            matched_tracks.add(ti)

    healthy = [i for i, t in enumerate(tracks) if t.missing == 0]
    ghosts  = [i for i, t in enumerate(tracks) if t.missing > 0]

    _match(healthy, lambda t: max_distance)                      # tight, first claim
    _match(ghosts,  lambda t: min(max_distance * (1 + t.missing / REACQUIRE_TAU),
                                  max_distance * 3))              # widening but CAPPED

    for i, t in enumerate(tracks):
        if i not in matched_tracks:
            t.mark_missing()
    return matched_dets


# ── drawing ────────────────────────────────────────────────────────────────────
def id_color(tid):
    palette = [(255, 80, 80), (80, 180, 255), (80, 255, 80), (0, 220, 255),
               (255, 0, 255), (255, 180, 0), (180, 100, 255)]
    return palette[(tid - 1) % len(palette)] if tid else (0, 255, 255)


def draw_dashed(frame, p1, p2, color, dash=6):
    x1, y1 = p1; x2, y2 = p2
    for x in range(x1, x2, dash * 2):
        cv2.line(frame, (x, y1), (min(x + dash, x2), y1), color, 1)
        cv2.line(frame, (x, y2), (min(x + dash, x2), y2), color, 1)
    for y in range(y1, y2, dash * 2):
        cv2.line(frame, (x1, y), (x1, min(y + dash, y2)), color, 1)
        cv2.line(frame, (x2, y), (x2, min(y + dash, y2)), color, 1)


def draw_frame(frame, tracks, locked, frame_count):
    for t in tracks:
        x1, y1, x2, y2 = [int(v) for v in t.bbox]
        color = id_color(t.id)
        if locked and t.missing > 0:
            draw_dashed(frame, (x1, y1), (x2, y2), color)
            label = f"Fish {t.id} (lost)"
        else:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"Fish {t.id}" if t.id else ""
        if label:
            cv2.putText(frame, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.putText(frame, f"Frame: {frame_count}", (10, frame.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


# ── main ───────────────────────────────────────────────────────────────────────
def main():
    c = yaml.safe_load(open(CONFIG_PATH))['fish_tracker']
    model        = YOLO(c['model_path'])
    num_fish     = c['num_fish']
    max_distance = c['max_distance']

    cap = cv2.VideoCapture(c['input_video_path'])
    fps = cap.get(cv2.CAP_PROP_FPS)
    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    start = c.get('start_seconds') or 0
    end   = c.get('end_seconds')
    if start:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(start * fps))
    max_frames  = int((end - start) * fps) if end else None
    calib_frames = int(c['calibration_secs'] * fps)

    # build a per-run output FOLDER (holds the video now; a log/config can join it later)
    clean    = c['output_video_path'].rstrip('/,. ')          # tolerate trailing junk
    run_name = os.path.splitext(os.path.basename(clean))[0]
    base_dir = os.path.dirname(clean) or 'output_fish_tracker'
    run_dir  = os.path.join(base_dir, f'{run_name}_basic')
    os.makedirs(run_dir, exist_ok=True)                       # create it if missing
    out_path = os.path.join(run_dir, f'{run_name}_basic.mp4')
    print(f"  run folder: {run_dir}")
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))

    tracks, next_id, locked, frame_count = [], 1, False, 0

    while True:
        ret, frame = cap.read()
        if not ret or (max_frames and frame_count >= max_frames):
            break

        # detect fish (class 0 only — reflections are class 1, ignored)
        res = model(frame, verbose=False)[0]
        dets, det_pos = [], []
        for i, cls in enumerate(res.boxes.cls.cpu().numpy()):
            if int(cls) != 0:
                continue
            x1, y1, x2, y2 = res.boxes.xyxy[i].cpu().numpy().tolist()
            dets.append(((x1 + x2) / 2, (y1 + y2) / 2, [x1, y1, x2, y2]))
            det_pos.append(((x1 + x2) / 2, (y1 + y2) / 2))
        det_pos = np.array(det_pos) if det_pos else np.empty((0, 2))

        if not locked:
            # CALIBRATION: match, then spawn a new track for any leftover detection
            matched = associate(tracks, dets, det_pos, max_distance)
            for i, (cx, cy, bbox) in enumerate(dets):
                if i not in matched:
                    tracks.append(Track(None, cx, cy, bbox))

            # Lock only once we actually have all num_fish tracks — never commit to
            # fewer, so every ID is real and can be followed for the whole video.
            # (Tracks persist once created, so all fish accumulate over the window.)
            past_window   = frame_count >= calib_frames
            hard_fallback = frame_count >= calib_frames * 5
            if (past_window and len(tracks) >= num_fish) or (hard_fallback and tracks):
                tracks.sort(key=lambda t: t.hits, reverse=True)
                tracks = tracks[:num_fish]
                for t in tracks:
                    t.id, t.missing = next_id, 0
                    next_id += 1
                locked = True
                print(f"  locked — {len(tracks)}/{num_fish} fish at frame {frame_count}")
        else:
            # LOCKED: exactly num_fish tracks, never added or removed. Healthy fish
            # keep their tags (first claim); a genuinely occluded fish's tag waits
            # as a ghost near its last position and only re-acquires a NEARBY
            # detection — never teleports across the tank onto another fish.
            associate(tracks, dets, det_pos, max_distance)

        draw_frame(frame, tracks, locked, frame_count)
        out.write(frame)

        if frame_count % 60 == 0:
            print(f"  frame {frame_count} | tracks={len(tracks)} | locked={locked}")
        frame_count += 1

    cap.release()
    out.release()
    print(f"\nDone. Saved to {out_path}")


if __name__ == '__main__':
    main()
