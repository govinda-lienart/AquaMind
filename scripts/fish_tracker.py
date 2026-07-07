# ── IMPORTS ───────────────────────────────────────────────────────────────────

import os
import sys
import warnings
import subprocess

import cv2
import numpy as np
import yaml
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO

from scripts.db import get_connection, get_video_id, register_track

warnings.filterwarnings('ignore')
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# ── CONSTANTS ─────────────────────────────────────────────────────────────────

CONFIG_PATH = 'config.yaml'
DEVICE      = 'mps'
HISTORY_LEN    = 20   # observed positions kept per track for crossing/velocity checks
CROSS_DISTANCE = 80   # px — centroid distance below which two tracks are considered crossing

# fixed per-fish identity colors (BGR) — color = identity, line style = tracking confidence
ID_COLORS = [
    (255, 0, 0),     # blue
    (0, 165, 255),   # orange
    (255, 0, 255),   # magenta
    (0, 255, 255),   # yellow
    (255, 255, 0),   # cyan
    (0, 128, 255),   # amber
    (147, 20, 255),  # pink
    (0, 255, 128),   # spring green
]


# ── HELPERS ───────────────────────────────────────────────────────────────────

class Tee:
    """Mirror stdout to a log file simultaneously."""
    def __init__(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._file   = open(path, 'w')
        self._stdout = sys.stdout
        sys.stdout   = self

    def write(self, data):
        self._stdout.write(data)
        self._file.write(data)

    def flush(self):
        self._stdout.flush()
        self._file.flush()

    def close(self):
        sys.stdout = self._stdout
        self._file.close()



def load_config():
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    return cfg['fish_tracker']


def print_run_config(input_video_path, model_path, output_video_path, start_seconds, end_seconds,
                      num_fish, calibration_secs, confirm_hits, max_distance, max_missing, show_trail):
    print("=" * 50)
    print(f"  Video:          {input_video_path}")
    print(f"  Model:          {model_path}")
    print(f"  Output:         {output_video_path}")
    print(f"  Seconds:        {start_seconds} → {end_seconds}")
    print(f"  Fish:           {num_fish}")
    print(f"  Calibration:    {calibration_secs} seconds")
    print(f"  confirm_hits:   {confirm_hits} frames")
    print(f"  max_distance:   {max_distance} px")
    print(f"  max_missing:    {max_missing} frames")
    print(f"  show_trail:     {show_trail}")
    print("=" * 50)
    input("Press Enter to start...")


def open_video_io(input_path, output_path, start_seconds, end_seconds):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cap         = cv2.VideoCapture(input_path)
    width       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps         = round(cap.get(cv2.CAP_PROP_FPS))
    start_frame = start_seconds * fps
    end_frame   = int(end_seconds * fps) if end_seconds is not None else int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_frames  = end_frame - start_frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    return cap, out, fps, max_frames


def bbox_iou(a, b):
    """IoU between two [x1,y1,x2,y2] boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def draw_dashed_rect(frame, pt1, pt2, color, thickness=1, dash_len=6):
    """Dashed rectangle — used to mark a track as 'ghost' (coasting on prediction, not detected)."""
    x1, y1 = pt1
    x2, y2 = pt2
    for y in (y1, y2):
        x = x1
        while x < x2:
            cv2.line(frame, (x, y), (min(x + dash_len, x2), y), color, thickness)
            x += dash_len * 2
    for x in (x1, x2):
        y = y1
        while y < y2:
            cv2.line(frame, (x, y), (x, min(y + dash_len, y2)), color, thickness)
            y += dash_len * 2


def id_color(tid):
    return ID_COLORS[(tid - 1) % len(ID_COLORS)]


def draw_frame(frame, confirmed_tracks, tentative_boxes, in_calibration, trail=None, frame_count = None, show_frame_number = False):
    
    if show_frame_number: # burns frame number on the frame
        cv2.putText(frame,f"Frame: {frame_count}", (20, frame.shape[0] - 20), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255), 2) # cv2puttext is open text function, Frame: frame_count is the text, Font_hershy 0.7 is font + size, 20 frameshape is the position of the text, (255, 255, 255) is the color, 2 is thikcness  )
    
    if in_calibration:
        for x1, y1, x2, y2 in tentative_boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 215, 255), 1)
        cv2.putText(frame, "CALIBRATION", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 165, 255), 3)

    if trail:
        for tid, _, _, _, _, _, _ in confirmed_tracks:
            pts = trail.get(tid, [])
            if len(pts) < 2:
                continue
            color = id_color(tid)
            for (x1, y1), (x2, y2) in zip(pts, pts[1:]):
                cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

    for tid, x1, y1, x2, y2, missing, _ in confirmed_tracks:
        color = id_color(tid)
        if missing > 0:
            label = f"Fish {tid} (lost)"
            draw_dashed_rect(frame, (x1, y1), (x2, y2), color, thickness=1)
        else:
            label = f"Fish {tid}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


# ── KALMAN TRACK ──────────────────────────────────────────────────────────────

class KalmanTrack:
    # State transition: position moves by velocity each frame
    F = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], dtype=float)
    # Observation: we only measure x, y (not velocity)
    H = np.array([[1,0,0,0],[0,1,0,0]], dtype=float)
    Q = np.eye(4) * 2    # process noise — how much we trust the motion model
    R = np.eye(2) * 10   # measurement noise — how much we trust the detector

    def __init__(self, x, y, bbox, confidence):
        self.hits           = 1
        self.missing_frames = 0
        self.state          = np.array([x, y, 0., 0.])  # [x, y, vx, vy]
        self.P              = np.eye(4) * 100            # initial uncertainty
        self.bbox           = bbox
        self.confidence     = confidence

    def predict(self):
        self.state = self.F @ self.state
        self.P     = self.F @ self.P @ self.F.T + self.Q

    def update(self, x, y, bbox, confidence):
        z     = np.array([x, y])
        y_res = z - self.H @ self.state
        S     = self.H @ self.P @ self.H.T + self.R
        K     = self.P @ self.H.T @ np.linalg.inv(S)
        self.state          = self.state + K @ y_res
        self.P              = (np.eye(4) - K @ self.H) @ self.P
        self.bbox           = bbox
        self.confidence     = confidence
        self.hits          += 1
        self.missing_frames = 0

    def mark_missing(self):
        self.missing_frames += 1
        self.state[2] *= 0.8  # decay velocity when fish is not seen
        self.state[3] *= 0.8

    @property
    def predicted_centre(self):
        return self.state[:2]


# ── ZEBRAFISH TRACKER ─────────────────────────────────────────────────────────

class ZebrafishTracker:
    """
    Two-phase tracker for a known number of zebrafish.

    During the calibration window tracks are tentative — no IDs assigned.
    After calibration_secs, any track with enough hits is confirmed with a
    permanent ID. Once num_fish IDs exist, the pool locks and no new IDs
    are created, blocking reflections and false positives.

    After calibration, crossing detection monitors centroid distances between
    confirmed tracks. When two fish get close, their pre-crossing trajectory
    direction is snapshotted from position history. When they separate, the
    post-crossing direction is compared against the snapshot — if both fish
    reversed relative to their pre-crossing heading, their IDs are swapped back.
    """

    def __init__(self, num_fish, max_distance, confirm_hits, max_missing,
                 show_trail=False, trail_length=150):
        self.num_fish        = num_fish
        self.max_distance    = max_distance
        self.confirm_hits    = confirm_hits
        self.max_missing     = max_missing
        self.show_trail      = show_trail
        self.trail_length    = trail_length
        self.confirmed            = {}    # tid → KalmanTrack
        self.tentative            = []    # list of KalmanTrack (no ID yet)
        self.next_id              = 1
        self.pool_locked          = False
        self.history              = {}    # tid → list of (cx, cy) bbox centres, capped short — crossing/velocity checks only
        self.trail                = {}    # tid → list of (cx, cy), longer window — visualisation only
        self.crossing_pairs       = set() # frozenset({tid_a, tid_b}) currently crossing
        self.crossing_had_overlap = {}    # frozenset → True if bboxes overlapped during this crossing
        self.pre_cross_pos        = {}    # tid → (cx, cy) position snapshotted at crossing start
        self.pre_cross_vel        = {}    # tid → (vx, vy) velocity snapshotted at crossing start

    def _update_trail(self, tid, bbox):
        if not self.show_trail:
            return
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        pts = self.trail.setdefault(tid, [])
        pts.append((cx, cy))
        if self.trail_length is not None and len(pts) > self.trail_length:
            self.trail[tid] = pts[-self.trail_length:]

    def _update_history(self, tid, bbox):
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        if tid not in self.history:
            self.history[tid] = []
        pts = self.history[tid]
        if not pts or np.linalg.norm([cx - pts[-1][0], cy - pts[-1][1]]) < self.max_distance:
            pts.append((cx, cy))
        if len(pts) > HISTORY_LEN:
            self.history[tid] = pts[-HISTORY_LEN:]

    def _velocity_from_history(self, tid, n=6):
        """Direction of travel from the last n positions in history."""
        pts = self.history.get(tid, [])
        if len(pts) < 2:
            return np.array([0., 0.])
        recent = pts[-min(n, len(pts)):]
        return np.array([recent[-1][0] - recent[0][0], recent[-1][1] - recent[0][1]])

    def update(self, detections, frame_count=0):
        """
        detections  : list of (cx, cy, bbox) — one per fish detection this frame
        frame_count : current frame index, used for crossing logs
        returns     : list of (tid, x1, y1, x2, y2, missing_frames) for all confirmed tracks
        """
        self._frame_count = frame_count
        # ── step 1: predict all tracks forward one frame ──────────────────────
        for t in self.confirmed.values():
            t.predict()
        for t in self.tentative:
            t.predict()

        if not detections:
            for t in self.confirmed.values():
                t.mark_missing()
            for t in self.tentative:
                t.mark_missing()
            self._prune_tentative()
            return self._output()

        det_positions = np.array([(x, y) for x, y, _, _ in detections]) # this line gets rid of the 2 last elements in (cx, cy, bbox, conf) usung _,  _ and keeps purely the centroid data for furtther use in the hungarian algoritm
        matched_dets  = set()

        # ── step 2: match confirmed tracks to detections ──────────────────────
        if self.confirmed:
            conf_ids = list(self.confirmed.keys())
            anchors  = np.array([self.confirmed[tid].predicted_centre for tid in conf_ids])
            cost     = np.linalg.norm(anchors[:, np.newaxis] - det_positions[np.newaxis, :], axis=2)
            r_idx, c_idx = linear_sum_assignment(cost)

            matched_conf = set()
            for ri, ci in zip(r_idx, c_idx):
                if cost[ri, ci] > self.max_distance:
                    continue
                tid          = conf_ids[ri]
                x, y, bbox, conf   = detections[ci]
                prev_missing = self.confirmed[tid].missing_frames
                self.confirmed[tid].update(x, y, bbox, conf)
                self._update_history(tid, bbox)
                self._update_trail(tid, bbox)
                if prev_missing > 0:
                    print(f"  Fish {tid} recovered after {prev_missing} missing frames")
                matched_conf.add(tid)
                matched_dets.add(ci)

            for tid in conf_ids:
                if tid not in matched_conf:
                    self.confirmed[tid].mark_missing()

        # ── step 3: match tentative tracks to remaining detections ────────────
        remaining = [i for i in range(len(detections)) if i not in matched_dets]

        if self.tentative and remaining:
            tent_centres  = np.array([t.predicted_centre for t in self.tentative])
            rem_positions = det_positions[remaining]
            cost          = np.linalg.norm(tent_centres[:, np.newaxis] - rem_positions[np.newaxis, :], axis=2)
            r_idx, c_idx  = linear_sum_assignment(cost)

            matched_tent = set()
            for ri, ci in zip(r_idx, c_idx):
                if cost[ri, ci] > self.max_distance:
                    continue
                x, y, bbox, conf = detections[remaining[ci]]
                self.tentative[ri].update(x, y, bbox, conf)
                matched_tent.add(ri)
                matched_dets.add(remaining[ci])

            for ti, t in enumerate(self.tentative):
                if ti not in matched_tent:
                    t.mark_missing()
        else:
            for t in self.tentative:
                t.mark_missing()

        # ── step 4: create new tentative tracks for unmatched detections ──────
        if not self.pool_locked:
            for i in range(len(detections)):
                if i not in matched_dets:
                    x, y, bbox, conf = detections[i]
                    self.tentative.append(KalmanTrack(x, y, bbox, conf))

        # ── step 5: promote tentative → confirmed when hits threshold reached ─
        if not self.pool_locked:
            still_tentative = []
            for t in self.tentative:
                if t.hits >= self.confirm_hits and self.next_id <= self.num_fish:
                    self.confirmed[self.next_id] = t
                    print(f"  Fish {self.next_id} confirmed after {t.hits} frames")
                    self.next_id += 1
                else:
                    still_tentative.append(t)
            self.tentative = still_tentative

        # ── step 6: lock pool once all fish are confirmed ─────────────────────
        if self.next_id > self.num_fish:
            self.pool_locked = True
            self.tentative   = []

        if self.pool_locked:
            self._check_crossings()

        self._prune_tentative()
        return self._output()

    def _maybe_swap(self, tid_a, tid_b):
        """Swap IDs only when BOTH position AND velocity agree that an exchange occurred.

        Position check: does swapping give better continuity with pre-crossing positions?
        Velocity check: did each fish reverse direction relative to its pre-crossing heading?
        Both must agree — one signal alone can be misleading.
        """
        if tid_a not in self.confirmed or tid_b not in self.confirmed:
            return
        if tid_a not in self.pre_cross_pos or tid_b not in self.pre_cross_pos:
            return

        pre_a  = np.array(self.pre_cross_pos[tid_a])
        pre_b  = np.array(self.pre_cross_pos[tid_b])
        post_a = np.array(self.confirmed[tid_a].predicted_centre)
        post_b = np.array(self.confirmed[tid_b].predicted_centre)

        no_swap_cost = np.linalg.norm(post_a - pre_a) + np.linalg.norm(post_b - pre_b)
        swap_cost    = np.linalg.norm(post_a - pre_b) + np.linalg.norm(post_b - pre_a)
        position_says_swap = swap_cost < no_swap_cost

        pre_va = np.array(self.pre_cross_vel.get(tid_a, (0., 0.)))
        pre_vb = np.array(self.pre_cross_vel.get(tid_b, (0., 0.)))
        cur_va = self._velocity_from_history(tid_a)
        cur_vb = self._velocity_from_history(tid_b)

        a_was_moving = np.linalg.norm(pre_va) > 1.0 and np.linalg.norm(cur_va) > 1.0
        b_was_moving = np.linalg.norm(pre_vb) > 1.0 and np.linalg.norm(cur_vb) > 1.0

        if a_was_moving and b_was_moving:
            velocity_says_swap = np.dot(cur_va, pre_va) < 0 and np.dot(cur_vb, pre_vb) < 0
        else:
            velocity_says_swap = None  # one fish too slow — velocity signal unreliable

        if velocity_says_swap is None:
            should_swap = position_says_swap
            reason = f"pos={'swap' if position_says_swap else 'keep'}, vel=N/A (fish too slow)"
        else:
            should_swap = position_says_swap and velocity_says_swap
            reason = f"pos={'swap' if position_says_swap else 'keep'}, vel={'swap' if velocity_says_swap else 'keep'}"

        if should_swap:
            self.confirmed[tid_a], self.confirmed[tid_b] = self.confirmed[tid_b], self.confirmed[tid_a]
            self.history[tid_a],   self.history[tid_b]   = self.history.get(tid_b, []), self.history.get(tid_a, [])
            self.trail[tid_a],     self.trail[tid_b]     = self.trail.get(tid_b, []),   self.trail.get(tid_a, [])
            print(f"  Fish {tid_a} ↔ Fish {tid_b}: IDs swapped ({reason}) [frame {self._frame_count}]")
        else:
            print(f"  Fish {tid_a} ↔ Fish {tid_b}: no swap ({reason}) [frame {self._frame_count}]")

    def _check_crossings(self):
        """Snapshot trajectory direction when two tracks start crossing; correct IDs when they separate."""
        tids = list(self.confirmed.keys())
        current_pairs = set()

        for i in range(len(tids)):
            for j in range(i + 1, len(tids)):
                tid_a, tid_b = tids[i], tids[j]
                ca   = self.confirmed[tid_a].predicted_centre
                cb   = self.confirmed[tid_b].predicted_centre
                dist = np.linalg.norm(ca - cb)
                pair = frozenset({tid_a, tid_b})

                iou = bbox_iou(self.confirmed[tid_a].bbox, self.confirmed[tid_b].bbox)

                if dist < CROSS_DISTANCE:
                    current_pairs.add(pair)
                    if iou > 0:
                        self.crossing_had_overlap[pair] = True
                        print(f"  Overlap detected: Fish {tid_a} ↔ Fish {tid_b} [frame {self._frame_count}] IoU={iou:.2f}")
                    if pair not in self.crossing_pairs:
                        for tid in (tid_a, tid_b):
                            if tid not in self.pre_cross_pos:
                                self.pre_cross_pos[tid] = tuple(self.confirmed[tid].predicted_centre)
                                vel = self._velocity_from_history(tid)
                                if np.linalg.norm(vel) < 0.5:
                                    vel = self.confirmed[tid].state[2:4]  # fallback to Kalman
                                self.pre_cross_vel[tid] = tuple(vel)
                        print(f"  Crossing started: Fish {tid_a} ↔ Fish {tid_b} [frame {self._frame_count}]")

        for pair in self.crossing_pairs - current_pairs:
            tid_a, tid_b = tuple(pair)
            if self.crossing_had_overlap.get(pair, False):
                self._maybe_swap(tid_a, tid_b)
            else:
                print(f"  Fish {tid_a} ↔ Fish {tid_b}: proximity only, no overlap — swap skipped [frame {self._frame_count}]")
            self.pre_cross_pos.pop(tid_a, None)
            self.pre_cross_pos.pop(tid_b, None)
            self.pre_cross_vel.pop(tid_a, None)
            self.pre_cross_vel.pop(tid_b, None)
            self.crossing_had_overlap.pop(pair, None)

        self.crossing_pairs = current_pairs

    def _prune_tentative(self):
        self.tentative = [t for t in self.tentative if t.missing_frames <= self.max_missing]

    def _output(self):
        return [(tid, *[int(v) for v in t.bbox], t.missing_frames, t.confidence) for tid, t in self.confirmed.items()]

    def tentative_boxes(self):
        return [[int(v) for v in t.bbox] for t in self.tentative]


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():

    # LOAD CONFIGURATION 
    p = load_config()
    input_video_path  = p['input_video_path']
    model_path        = p['model_path']
    output_video_path = p['output_video_path']
    start_seconds     = p['start_seconds']
    end_seconds       = p['end_seconds']
    num_fish          = p['num_fish']
    calibration_secs  = p['calibration_secs']
    max_distance      = p['max_distance']
    confirm_hits      = p['confirm_hits']
    max_missing       = p['max_missing']
    show_trail        = p['show_trail']
    trail_length      = p['trail_length']
    show_frame_number = p['show_frame_number']

    # ----------------------------------------------
    # PARSING, NAMING AND STORING SIDECAR AND LOG
    # ----------------------------------------------

    # Build a dedicated subfolder for this run, named after the video's base filename
    run_name = os.path.splitext(os.path.basename(output_video_path))[0] # run_name = "stage5_tracker_IMG_2349_as_3r_4r_5r_8c_2026_07_06_1033"
    run_dir = os.path.join(os.path.dirname(output_video_path), run_name) # run_dir = "output_video_zebratracker/stage5_tracker_IMG_2349_as_3r_4r_5r_8c_2026_07_06_1033"
    os.makedirs(run_dir, exist_ok=True)  # creates that folder on disk (no error if it already exists)

    # Redirect the video into that subfolder
    output_video_path = os.path.join(run_dir, run_name + '.mp4')     # output_video_path = "output_video_zebratracker/stage5_tracker_IMG_2349_as_3r_4r_5r_8c_2026_07_06_1033/stage5_tracker_IMG_2349_as_3r_4r_5r_8c_2026_07_06_1033.mp4"

    # Save this run's exact config next to the video-sidecar, in the same subfolder
    config_sidecar_path = os.path.join(run_dir, run_name + '_config.yaml')     # config_sidecar_path = "output_video_zebratracker/stage5_tracker_IMG_2349_as_3r_4r_5r_8c_2026_07_06_1033/stage5_tracker_IMG_2349_as_3r_4r_5r_8c_2026_07_06_1033_config.yaml"
    p['git_commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip() # # strip gets rid of \n # decode back to normal from raw bites # t5akes last git commit to get referencing code when the trackert was run

    with open(config_sidecar_path, 'w') as f:
        yaml.dump(p, f)

    log_name = os.path.splitext(os.path.basename(output_video_path))[0] + '.log'     # log_name = "stage5_tracker_IMG_2349_as_3r_4r_5r_8c_2026_07_06_1033.log"
    tee = Tee(os.path.join(run_dir, log_name))     # writes to "output_video_zebratracker/stage5_tracker_IMG_2349_as_3r_4r_5r_8c_2026_07_06_1033/stage5_tracker_IMG_2349_as_3r_4r_5r_8c_2026_07_06_1033.log"

    print_run_config(input_video_path, model_path, output_video_path, start_seconds, end_seconds,
                      num_fish, calibration_secs, confirm_hits, max_distance, max_missing, show_trail)

    # ----------------------------------------------
    # CONNECT WITH MYSQL AND DELETE CURRENT TRACKING DATA
    # ----------------------------------------------
    
    conn = get_connection()
    cursor = conn.cursor()

    video_id = get_video_id(cursor, input_video_path)
    logger.info(f"{video_id} found!")

    cursor.execute("DELETE FROM tracks WHERE video_id = %s", (video_id,)) # delete all tracking info of a particular video to start from scratch
    conn.commit()

    # ----------------------------------------------

    model   = YOLO(model_path) # access model
    model.to(DEVICE)
    tracker = ZebrafishTracker(
        num_fish     = num_fish,
        max_distance = max_distance,
        confirm_hits = confirm_hits,
        max_missing  = max_missing,
        show_trail   = show_trail,
        trail_length = trail_length,
    )

    cap, out, fps, max_frames = open_video_io(
        input_video_path, output_video_path,
        start_seconds,    end_seconds,
    )
    calibration_frames = calibration_secs * fps

    frame_count = 0
    while True: # 
        ret, frame = cap.read() # reading next frame on top of stack # frame ( the image ) a NumPy array (height × width × 3 color channels) / ret (true/false) false if no frame
        if not ret or frame_count >= max_frames: # if no more frame to read stop
            break

        results = model(frame, verbose=False, iou=0.5) # estalish fish bboxes / verbose false to avoid stats on every frame / iou controls non-maximum suppression(NMS) - raw output..not just one clean box per fis...everal...- nms looks at heavily overlapping boxes...kill the low confidence one
        boxes   = results[0].boxes  # this is YOLO's full report for a particular frame - every box - if 3 fish and 4 reflections found - 7 entries.
                                    # results is  a plain python list - contains an instance of a class also called Results which needs to be extracted using [0]
                                    # boxes contains several arrays including .cls, .xyxy, .conf
                                    # boxes.xyxy is a 2D matrix consisting of x1,y1,x2,y2 
                                    # boxes.cls is a 1D array - plain list of numbers, one per detection # [0, 0, 1, 0] so means  (using  class IDs: 0 = danio_rerio, 1 = reflection).
                                    # boxes.conf -  1D array, one confidence number per detection
                                    
        # detection loop 
        detections = []
        for i, cls_id in enumerate(boxes.cls.cpu().numpy()): 
            # cpu copies array from GPU to CPU so python/numpy can work with it
            # numpy() converts pytorch tensor into NumpyArray - easier to loop example array([0., 0., 1., 0.])
            # enumarate - wraps it as pair(index, value) ex ((0, 0.0)(1, 0.0)(2, 1.0)....
 
            if int(cls_id) != 0:
                continue
            bbox        = boxes.xyxy[i].cpu().numpy().tolist() # tensor([120.3, 340.1, 160.8, 380.5]) ->array > [120.3, 340.1, 160.8, 380.5]
            x1, y1, x2, y2 = bbox
            cx, cy      = (x1 + x2) / 2, (y1 + y2) / 2
            conf = float(boxes.conf[i].cpu().numpy()) # no need for tolist...its just one number...convert to float
            detections.append((cx, cy, bbox, conf))

        in_calibration = frame_count < calibration_frames
        tracked        = tracker.update(detections, frame_count=frame_count) # list of tuples (tid, x1, y1, x2, y2, missing, confidence)

        for tid, x1, y1, x2, y2, missing, confidence in tracked:
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            timestamp = start_seconds + frame_count / fps
            occluded = missing > 0
            register_track(cursor, video_id, tid, frame_count, timestamp, cx, cy, confidence, occluded)

        draw_frame(frame, tracked, tracker.tentative_boxes(), in_calibration, tracker.trail, frame_count, show_frame_number)
        out.write(frame)
        
        # -------------------------------------------------------------
        # LIVE PREVIEW OF TRACKER
        # -------------------------------------------------------------
        cv2.imshow('Fish Tracker', frame) # opens a window titled Fish tracker 
        key = cv2.waitKey(1) & 0xFF # refreshes what is on the screen (every millisecond) and 0xFF checks if any key was pressed like a q or ESC for escape 
        if key == ord('q') or key == 27: # 27 refers to ESC
            print("quit requested - stopping early")            
            break
        
        # -------------------------------------------------------------

        if frame_count == calibration_frames and not tracker.pool_locked:
            tracker.pool_locked = True
            tracker.tentative   = []
            print(f"  Calibration closed — {len(tracker.confirmed)} fish confirmed")

        # -------------------------------------------------------------
        # COMMIT TO MYSQL
        # -------------------------------------------------------------
        if frame_count % 30 == 0:
            current_second = start_seconds + frame_count // fps
            print(f"  Frame {frame_count}/{max_frames} | second {current_second}")
            conn.commit() # commits every 30 frames to mysql - to now slow down the process but also to not lose all if crash

        frame_count += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows() # if pressed q or ESC - closes preview window 
    conn.commit() # ensuring full commit
    cursor.close()
    conn.close()

    print(f"\nDone. Saved to {output_video_path}")
    tee.close()

# ── ENTRY POINT ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    main()