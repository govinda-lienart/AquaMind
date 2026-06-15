# ── IMPORTS ───────────────────────────────────────────────────────────────────

import os
import warnings

import cv2
import numpy as np
import yaml
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO

warnings.filterwarnings('ignore')


# ── CONSTANTS ─────────────────────────────────────────────────────────────────

CONFIG_PATH = 'config.yaml'
DEVICE      = 'mps'
HISTORY_LEN    = 20   # observed positions kept per track for polyfit
PROJ_AHEAD     = 60   # frames to project arrow forward for visualisation (~1 sec at 60fps)
CROSS_DISTANCE = 80   # px — centroid distance below which two tracks are considered crossing


# ── HELPERS ───────────────────────────────────────────────────────────────────

def load_config():
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    t = cfg['track_zebrafish']
    return {
        'input_video_path': t['input_video_path'],
        'model_path':       t['model_path'],
        'output_video_path':t['output_video_path'],
        'start_seconds':    t['start_seconds'],
        'end_seconds':      t['end_seconds'],
        'num_fish':         t['num_fish'],
        'calibration_secs': t['calibration_secs'],
        'max_distance':     t['max_distance'],
        'confirm_hits':     t['confirm_hits'],
        'max_missing':      t['max_missing'],
    }


def print_run_config(p):
    print("=" * 50)
    print(f"  Video:          {p['input_video_path']}")
    print(f"  Model:          {p['model_path']}")
    print(f"  Output:         {p['output_video_path']}")
    print(f"  Seconds:        {p['start_seconds']} → {p['end_seconds']}")
    print(f"  Fish:           {p['num_fish']}")
    print(f"  Calibration:    {p['calibration_secs']} seconds")
    print(f"  confirm_hits:   {p['confirm_hits']} frames")
    print(f"  max_distance:   {p['max_distance']} px")
    print(f"  max_missing:    {p['max_missing']} frames")
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


def draw_frame(frame, confirmed_tracks, tentative_boxes, in_calibration, history=None):
    if in_calibration:
        for x1, y1, x2, y2 in tentative_boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 215, 255), 1)
        cv2.putText(frame, "CALIBRATION", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 165, 255), 3)

    for tid, x1, y1, x2, y2 in confirmed_tracks:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"Fish {tid}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    if history:
        for tid, x1, y1, x2, y2 in confirmed_tracks:
            pts = history.get(tid, [])
            if len(pts) < 2:
                continue
            t_vals = np.arange(len(pts))
            px     = np.polyfit(t_vals, [p[0] for p in pts], 1)
            py     = np.polyfit(t_vals, [p[1] for p in pts], 1)
            t_proj = len(pts) - 1 + PROJ_AHEAD
            x_proj = int(np.polyval(px, t_proj))
            y_proj = int(np.polyval(py, t_proj))
            cx_now, cy_now = int(pts[-1][0]), int(pts[-1][1])
            cv2.arrowedLine(frame, (cx_now, cy_now), (x_proj, y_proj), (0, 255, 255), 2, tipLength=0.3)


# ── KALMAN TRACK ──────────────────────────────────────────────────────────────

class KalmanTrack:
    # State transition: position moves by velocity each frame
    F = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], dtype=float)
    # Observation: we only measure x, y (not velocity)
    H = np.array([[1,0,0,0],[0,1,0,0]], dtype=float)
    Q = np.eye(4) * 2    # process noise — how much we trust the motion model
    R = np.eye(2) * 10   # measurement noise — how much we trust the detector

    def __init__(self, x, y, bbox):
        self.hits           = 1
        self.missing_frames = 0
        self.state          = np.array([x, y, 0., 0.])  # [x, y, vx, vy]
        self.P              = np.eye(4) * 100            # initial uncertainty
        self.bbox           = bbox

    def predict(self):
        self.state = self.F @ self.state
        self.P     = self.F @ self.P @ self.F.T + self.Q

    def update(self, x, y, bbox):
        z     = np.array([x, y])
        y_res = z - self.H @ self.state
        S     = self.H @ self.P @ self.H.T + self.R
        K     = self.P @ self.H.T @ np.linalg.inv(S)
        self.state          = self.state + K @ y_res
        self.P              = (np.eye(4) - K @ self.H) @ self.P
        self.bbox           = bbox
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
    confirmed tracks. When two fish overlap, their pre-crossing Kalman velocities
    are snapshotted. When they separate, if both velocities reversed direction
    relative to the snapshot, the IDs are swapped back.
    """

    def __init__(self, num_fish, max_distance, confirm_hits, max_missing):
        self.num_fish        = num_fish
        self.max_distance    = max_distance
        self.confirm_hits    = confirm_hits
        self.max_missing     = max_missing
        self.confirmed       = {}    # tid → KalmanTrack
        self.tentative       = []    # list of KalmanTrack (no ID yet)
        self.next_id         = 1
        self.pool_locked     = False
        self.history         = {}    # tid → list of (cx, cy) bbox centres
        self.crossing_pairs  = set() # frozenset({tid_a, tid_b}) currently crossing
        self.pre_cross_vel   = {}    # tid → (vx, vy) snapshotted at crossing start

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

    def update(self, detections):
        """
        detections : list of (cx, cy, bbox) — one per fish detection this frame
        returns    : list of (tid, x1, y1, x2, y2) for all confirmed tracks
        """
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

        det_positions = np.array([(x, y) for x, y, _ in detections])
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
                x, y, bbox   = detections[ci]
                prev_missing = self.confirmed[tid].missing_frames
                self.confirmed[tid].update(x, y, bbox)
                self._update_history(tid, bbox)
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
                x, y, bbox = detections[remaining[ci]]
                self.tentative[ri].update(x, y, bbox)
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
                    x, y, bbox = detections[i]
                    self.tentative.append(KalmanTrack(x, y, bbox))

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
        """Swap IDs if both fish reversed direction relative to their pre-crossing heading."""
        if tid_a not in self.confirmed or tid_b not in self.confirmed:
            return
        if tid_a not in self.pre_cross_vel or tid_b not in self.pre_cross_vel:
            return
        pre_va = np.array(self.pre_cross_vel[tid_a])
        pre_vb = np.array(self.pre_cross_vel[tid_b])
        if np.linalg.norm(pre_va) < 0.5 or np.linalg.norm(pre_vb) < 0.5:
            return  # too slow at crossing start — direction unreliable
        cur_va = self.confirmed[tid_a].state[2:4]
        cur_vb = self.confirmed[tid_b].state[2:4]
        if np.dot(cur_va, pre_va) < 0 and np.dot(cur_vb, pre_vb) < 0:
            self.confirmed[tid_a], self.confirmed[tid_b] = self.confirmed[tid_b], self.confirmed[tid_a]
            self.history[tid_a],   self.history[tid_b]   = self.history.get(tid_b, []), self.history.get(tid_a, [])
            print(f"  Fish {tid_a} ↔ Fish {tid_b}: IDs swapped back after crossing")
        else:
            print(f"  Fish {tid_a} ↔ Fish {tid_b}: crossing resolved, no swap needed")

    def _check_crossings(self):
        """Snapshot velocities when two tracks start crossing; correct IDs when they separate."""
        tids = list(self.confirmed.keys())
        current_pairs = set()

        for i in range(len(tids)):
            for j in range(i + 1, len(tids)):
                tid_a, tid_b = tids[i], tids[j]
                ca   = self.confirmed[tid_a].predicted_centre
                cb   = self.confirmed[tid_b].predicted_centre
                dist = np.linalg.norm(ca - cb)
                pair = frozenset({tid_a, tid_b})

                if dist < CROSS_DISTANCE:
                    current_pairs.add(pair)
                    if pair not in self.crossing_pairs:
                        for tid in (tid_a, tid_b):
                            if tid not in self.pre_cross_vel:
                                vx, vy = self.confirmed[tid].state[2], self.confirmed[tid].state[3]
                                self.pre_cross_vel[tid] = (vx, vy)
                        print(f"  Crossing started: Fish {tid_a} ↔ Fish {tid_b}")

        for pair in self.crossing_pairs - current_pairs:
            tid_a, tid_b = tuple(pair)
            self._maybe_swap(tid_a, tid_b)
            self.pre_cross_vel.pop(tid_a, None)
            self.pre_cross_vel.pop(tid_b, None)

        self.crossing_pairs = current_pairs

    def _prune_tentative(self):
        self.tentative = [t for t in self.tentative if t.missing_frames <= self.max_missing]

    def _output(self):
        return [(tid, *[int(v) for v in t.bbox]) for tid, t in self.confirmed.items()]

    def tentative_boxes(self):
        return [[int(v) for v in t.bbox] for t in self.tentative]


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    p = load_config()
    print_run_config(p)

    model   = YOLO(p['model_path'])
    model.to(DEVICE)
    tracker = ZebrafishTracker(
        num_fish     = p['num_fish'],
        max_distance = p['max_distance'],
        confirm_hits = p['confirm_hits'],
        max_missing  = p['max_missing'],
    )

    cap, out, fps, max_frames = open_video_io(
        p['input_video_path'], p['output_video_path'],
        p['start_seconds'],    p['end_seconds'],
    )
    calibration_frames = p['calibration_secs'] * fps

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret or frame_count >= max_frames:
            break

        results = model(frame, verbose=False, iou=0.5)
        boxes   = results[0].boxes

        detections = []
        for i, cls_id in enumerate(boxes.cls.cpu().numpy()):
            if int(cls_id) != 0:
                continue
            bbox        = boxes.xyxy[i].cpu().numpy().tolist()
            x1, y1, x2, y2 = bbox
            cx, cy      = (x1 + x2) / 2, (y1 + y2) / 2
            detections.append((cx, cy, bbox))

        in_calibration = frame_count < calibration_frames
        tracked        = tracker.update(detections)
        draw_frame(frame, tracked, tracker.tentative_boxes(), in_calibration, tracker.history)
        out.write(frame)

        if frame_count == calibration_frames and not tracker.pool_locked:
            tracker.pool_locked = True
            tracker.tentative   = []
            print(f"  Calibration closed — {len(tracker.confirmed)} fish confirmed")

        if frame_count % 30 == 0:
            current_second = p['start_seconds'] + frame_count // fps
            print(f"  Frame {frame_count}/{max_frames} | second {current_second}")

        frame_count += 1

    cap.release()
    out.release()
    print(f"\nDone. Saved to {p['output_video_path']}")


# ── ENTRY POINT ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    main()
