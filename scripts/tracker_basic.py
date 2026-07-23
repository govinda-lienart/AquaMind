"""
tracker_basic.py — a deliberately minimal fish tracker (EXPERIMENT).
Run:  python -m scripts.tracker_basic
Output: <output_video_path with '_basic' before .mp4>
"""

import os
import json
import logging
import subprocess
from datetime import datetime

import cv2
import yaml
import numpy as np
from ultralytics import YOLO
from scipy.optimize import linear_sum_assignment

from scripts.db import get_connection, get_video_id, register_track

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

CONFIG_PATH   = 'config.yaml'
REACQUIRE_TAU = 15    # ghost search radius grows by one max_distance every 15 missing frames
VEL_SMOOTH    = 0.5   # EMA weight for the velocity estimate (0=frozen, 1=raw last step)


# ── a track that remembers where the fish was AND where it's heading ────────────
class Track:
    def __init__(self, tid, x, y, bbox, confidence):
        self.id         = tid       # None while tentative (during calibration)
        self.x          = x
        self.y          = y
        self.vx         = 0.0       # per-frame velocity — the constant-velocity motion model
        self.vy         = 0.0
        self.bbox       = bbox
        self.confidence = confidence  # YOLO detection confidence (stored in tracks, like fish_tracker)
        self.hits       = 1         # how many frames it has been matched
        self.missing    = 0         # consecutive frames with no detection (ghost)


    



    @property
    def pred(self):
        """Predicted next-frame position under constant velocity — the anchor we match on.
        Using this instead of the last position is what stops two crossing fish swapping IDs:
        the fish overshoot past each other, so 'nearest last position' picks the WRONG detection,
        but 'nearest predicted position' picks the right one."""
        return (self.x + self.vx, self.y + self.vy)

    def update(self, x, y, bbox, confidence):
        frames  = self.missing + 1                    # frames elapsed since the last real detection
        inst_vx = (x - self.x) / frames               # per-frame velocity across that gap
        inst_vy = (y - self.y) / frames
        self.vx = VEL_SMOOTH * inst_vx + (1 - VEL_SMOOTH) * self.vx   # EMA — smooth out detection jitter
        self.vy = VEL_SMOOTH * inst_vy + (1 - VEL_SMOOTH) * self.vy
        self.x, self.y, self.bbox = x, y, bbox
        self.confidence = confidence
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
        anchors = np.array([tracks[i].pred for i in track_idx])   # predicted position, not last seen
        pos     = det_pos[avail]
        cost    = np.linalg.norm(anchors[:, None] - pos[None, :], axis=2)
        for r, c in zip(*linear_sum_assignment(cost)):
            ti = track_idx[r]
            if cost[r, c] > gate_fn(tracks[ti]):
                continue
            x, y, bbox, conf = dets[avail[c]]
            tracks[ti].update(x, y, bbox, conf)
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


def draw_calibration_badge(frame, frame_count):
    """Semi-transparent amber 'CALIBRATING' badge, top-left, with an animated ellipsis."""
    dots = "." * (1 + (frame_count // 15) % 3)          # ".", "..", "..." — shows it's live
    text = f"CALIBRATING{dots}"
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
    (tw, th), _ = cv2.getTextSize("CALIBRATING...", font, scale, thick)  # size on longest form so box doesn't jitter
    x, y, pad = 15, 15, 10
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + tw + 2 * pad, y + th + 2 * pad), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)  # translucent dark backing
    cv2.putText(frame, text, (x + pad, y + th + pad - 2), font, scale, (0, 215, 255), thick, cv2.LINE_AA)


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
    if not locked:
        draw_calibration_badge(frame, frame_count)


# ── logging + run summary (same style as fish_tracker.py) ──────────────────────
def setup_run_logging(log_path):
    """One logger, two handlers: file (in the run folder) + console — bare messages, like fish_tracker."""
    formatter       = logging.Formatter('%(message)s')
    file_handler    = logging.FileHandler(log_path)
    console_handler = logging.StreamHandler()
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


def print_run_config(input_video_path, model_path, output_video_path, start_seconds, end_seconds,
                     num_fish, calibration_secs, max_distance):
    logger.info("=" * 50)
    logger.info(f"  Video:          {input_video_path}")
    logger.info(f"  Model:          {model_path}")
    logger.info(f"  Output:         {output_video_path}")
    logger.info(f"  Seconds:        {start_seconds} → {end_seconds}")
    logger.info(f"  Fish:           {num_fish}")
    logger.info(f"  Calibration:    {calibration_secs} seconds")
    logger.info(f"  max_distance:   {max_distance} px")
    logger.info(f"  reacquire_tau:  {REACQUIRE_TAU} frames")
    logger.info("=" * 50)
    input("Press Enter to start...")


# ── main ───────────────────────────────────────────────────────────────────────
def main():
    c = yaml.safe_load(open(CONFIG_PATH))['basic_tracker']
    input_video_path  = c['input_video_path']
    model_path        = c['model_path']
    num_fish          = c['num_fish']
    max_distance      = c['max_distance']
    calibration_secs  = c['calibration_secs']
    start = c.get('start_seconds') or 0
    end   = c.get('end_seconds')

    # build a per-run output FOLDER (holds the video, log, and config sidecar)
    clean    = c['output_video_path'].rstrip('/,. ')          # tolerate trailing junk
    base     = os.path.splitext(os.path.basename(clean))[0]
    stamp    = datetime.now().strftime('%Y_%m_%d_%H%M')       # auto timestamp → a fresh folder every run
    run_name = f'{base}_basic_{stamp}'                        # e.g. 'tracker_IMG_1839_basic_2026_07_23_1642'
    base_dir = os.path.dirname(clean) or 'output_fish_tracker'
    run_dir  = os.path.join(base_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)                       # create it if missing
    out_path = os.path.join(run_dir, f'{run_name}.mp4')
    log_path = os.path.join(run_dir, f'{run_name}.log')

    # freeze this run's exact config (+ the git commit it ran at) next to the outputs,
    # so a later ground-truth/crop review knows which model & params produced this folder
    c['git_commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()
    with open(os.path.join(run_dir, f'{run_name}_config.yaml'), 'w') as f:
        yaml.dump(c, f)

    # logging + parameter summary FIRST, then block on Enter before any heavy work
    setup_run_logging(log_path)
    logger.info(f"  run folder: {run_dir}")
    print_run_config(input_video_path, model_path, out_path, start, end,
                     num_fish, calibration_secs, max_distance)

    model = YOLO(model_path)

    cap = cv2.VideoCapture(input_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if start:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(start * fps))
    max_frames  = int((end - start) * fps) if end else None
    # total frames we'll actually process — the progress denominator (whole video if no end_seconds)
    total_frames = max_frames if max_frames else int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - int(start * fps)
    calib_frames = int(calibration_secs * fps)

    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))

    # ── MySQL: clear this video's old tracks, then write fresh (identical to fish_tracker) ──
    conn     = get_connection()
    cursor   = conn.cursor()
    video_id = get_video_id(cursor, input_video_path)
    cursor.execute("DELETE FROM tracks WHERE video_id = %s", (video_id,))  # working storage: wipe prior run
    conn.commit()

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
            conf = float(res.boxes.conf[i].cpu().numpy())     # YOLO confidence for this detection
            dets.append(((x1 + x2) / 2, (y1 + y2) / 2, [x1, y1, x2, y2], conf))
            det_pos.append(((x1 + x2) / 2, (y1 + y2) / 2))
        det_pos = np.array(det_pos) if det_pos else np.empty((0, 2))

        if not locked:
            # CALIBRATION: match, then spawn a new track for any leftover detection
            matched = associate(tracks, dets, det_pos, max_distance)
            for i, (cx, cy, bbox, conf) in enumerate(dets):
                if i not in matched:
                    tracks.append(Track(None, cx, cy, bbox, conf))

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
                logger.info(f"  locked — {len(tracks)}/{num_fish} fish at frame {frame_count}")
        else:
            # LOCKED: exactly num_fish tracks, never added or removed. Healthy fish
            # keep their tags (first claim); a genuinely occluded fish's tag waits
            # as a ghost near its last position and only re-acquires a NEARBY
            # detection — never teleports across the tank onto another fish.
            prev_missing = {t.id: t.missing for t in tracks}   # snapshot before matching
            associate(tracks, dets, det_pos, max_distance)
            for t in tracks:                                   # log lost/recovered transitions
                was, now = prev_missing[t.id], t.missing
                if was == 0 and now > 0:                       # first frame a fish drops out
                    logger.info(json.dumps({"event": "occlusion_lost", "fish_ids": str(t.id),
                                            "frame": frame_count}))
                elif was > 0 and now == 0:                     # fish reappears — was = frames it was missing
                    logger.info(json.dumps({"event": "occlusion_recovery", "fish_ids": str(t.id),
                                            "decision": "recovered", "frame": frame_count,
                                            "missing_frames": was}))

        
        # write this frame's identified tracks to MySQL (only tracks that already have an id;
        # x/y are the centroid, occluded = ghost, confidence from the last matched detection)
        timestamp = start + frame_count / fps
        for t in tracks:
            if t.id is None:                         # tentative tracks (pre-lock) aren't persisted
                continue
            register_track(cursor, video_id, t.id, frame_count, timestamp, t.x, t.y, t.confidence, t.missing > 0)

            # cropping out the fish bbox to save it on disk and use later for fish RE-ID Machine learning
            if t.missing > 0: 
                continue # here if for example the frame is a ghost...skip no need to grab it for RE_ID as we dont want to crop a ghost
            x1, y1, x2, y2 = t.bbox # here we unpack the bbox attribnute of track objects...which gives the format xyxy (different fdormat than xywh) but great cause this gives us the edges of the image to crop
            crop = frame[int(y1):int(y2), int(x1):int(x2)] # so here we crop in the frame array pixels - numpy - so we select the row first (vertical) range y1 to y2 for height of the box, and then we select horizantal x1 to x2 
            
            # creating folders storing the crops
            crop_folder_path = os.path.join(run_dir, 'crops', f'fish_{t.id}') # outputs e.g. 'output_fish_tracker/run_01/crops/fish_5'
            os.makedirs(crop_folder_path, exist_ok=True) # this one outputs nothing but creates the folder 'output_fish_tracker/run_01/crops/fish_5' on disk (does nothing if it already exists)
            
            # creating the actual crop file
            crop_name = f"frame_{frame_count}_fish_{t.id}"  # outputs e.g. 'frame_1001_fish_5'
            filename = f"{crop_folder_path}/{crop_name}.jpg"  # outputs e.g. 'output_fish_tracker/run_01/crops/fish_5/frame_1001_fish_5.jpg'
            cv2.imwrite(filename, crop) # outputs True/False; writes the .jpg image to that path on disk

        # drawing the frame    
        draw_frame(frame, tracks, locked, frame_count)
        out.write(frame)

        cv2.imshow('Fish Tracker (basic)', frame)   # live preview
        key = cv2.waitKey(1) & 0xFF                  # refresh window; read any key press
        if key == ord('q') or key == 27:            # q or ESC (27) quits early
            logger.info("quit requested - stopping early")
            break

        if frame_count % 30 == 0:                    # commit periodically — same cadence as fish_tracker
            conn.commit()
        if frame_count % 60 == 0:
            logger.info(f"  frame {frame_count}/{total_frames} | tracks={len(tracks)} | locked={locked}")
        frame_count += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()                          # close preview window
    conn.commit()                                    # final flush
    cursor.close()
    conn.close()
    logger.info(f"\nDone. Saved to {out_path}")


if __name__ == '__main__':
    main()
