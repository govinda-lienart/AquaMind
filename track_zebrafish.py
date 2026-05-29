import os
import warnings
import yaml
warnings.filterwarnings('ignore')
import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO
from scipy.optimize import linear_sum_assignment


# ── Kalman track ──────────────────────────────────────────────────────────────

class KalmanTrack:
    F = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], dtype=float)
    H = np.array([[1,0,0,0],[0,1,0,0]], dtype=float)
    Q = np.eye(4) * 2
    R = np.eye(2) * 10

    def __init__(self, bbox):
        self.hits           = 1
        self.missing_frames = 0
        cx, cy              = self._centre(bbox)
        self.state          = np.array([cx, cy, 0., 0.])
        self.P              = np.eye(4) * 100
        self.bbox           = bbox

    @staticmethod
    def _centre(bbox):
        x1, y1, x2, y2 = bbox
        return (x1 + x2) / 2, (y1 + y2) / 2

    def predict(self):
        self.state = self.F @ self.state
        self.P     = self.F @ self.P @ self.F.T + self.Q

    def update(self, bbox):
        cx, cy = self._centre(bbox)
        z      = np.array([cx, cy])
        y      = z - self.H @ self.state
        S      = self.H @ self.P @ self.H.T + self.R
        K      = self.P @ self.H.T @ np.linalg.inv(S)
        self.state          = self.state + K @ y
        self.P              = (np.eye(4) - K @ self.H) @ self.P
        self.bbox           = bbox
        self.hits          += 1
        self.missing_frames = 0

    def mark_missing(self):
        self.missing_frames += 1

    @property
    def predicted_centre(self):
        return self.state[:2]


# ── Zebrafish tracker ─────────────────────────────────────────────────────────

class ZebrafishTracker:
    """
    Two-phase tracker for a fixed known population of zebrafish.

    Phase 1 — Tentative: a new detection is watched silently for `confirm_hits`
    consecutive frames. No ID is assigned, nothing is drawn on screen.

    Phase 2 — Confirmed: after `confirm_hits` frames the track is promoted and
    receives a permanent ID. Confirmed tracks are never deleted — their Kalman
    ghost keeps extrapolating position during occlusion so the fish is matched
    back to its original ID when it reappears.

    Once `num_fish` confirmed IDs exist the pool locks and no new IDs are
    ever created, preventing reflections and false positives from stealing slots.
    """

    def __init__(self, num_fish=5, max_distance=150, confirm_hits=15, max_tentative_missing=5):
        self.num_fish              = num_fish
        self.max_distance          = max_distance
        self.confirm_hits          = confirm_hits          # frames needed to get a permanent ID
        self.max_tentative_missing = max_tentative_missing # frames before dropping a tentative track

        self.confirmed  = {}   # {track_id: KalmanTrack}  — permanent fish
        self.tentative  = []   # [KalmanTrack]             — candidates on probation
        self.next_id    = 1
        self.pool_locked = False

    def update(self, bboxes):
        # 1. Predict positions for all tracks
        for t in self.confirmed.values():
            t.predict()
        for t in self.tentative:
            t.predict()

        if len(bboxes) == 0:
            for t in self.confirmed.values():
                t.mark_missing()
            for t in self.tentative:
                t.mark_missing()
            self._prune_tentative()
            return self._output()

        det_centres = np.array([[(b[0]+b[2])/2, (b[1]+b[3])/2] for b in bboxes])
        matched_dets = set()

        # 2. Match detections to CONFIRMED tracks first (priority)
        if self.confirmed:
            conf_ids     = list(self.confirmed.keys())
            conf_centres = np.array([self.confirmed[tid].predicted_centre for tid in conf_ids])
            cost         = np.linalg.norm(conf_centres[:, np.newaxis] - det_centres[np.newaxis, :], axis=2)
            r_idx, c_idx = linear_sum_assignment(cost)

            matched_conf = set()
            for ri, ci in zip(r_idx, c_idx):
                if cost[ri, ci] > self.max_distance:
                    continue
                self.confirmed[conf_ids[ri]].update(bboxes[ci])
                matched_conf.add(conf_ids[ri])
                matched_dets.add(ci)

            for tid in conf_ids:
                if tid not in matched_conf:
                    self.confirmed[tid].mark_missing()

        # 3. Match remaining detections to TENTATIVE tracks
        remaining_dets = [i for i in range(len(bboxes)) if i not in matched_dets]

        if self.tentative and remaining_dets:
            tent_centres  = np.array([t.predicted_centre for t in self.tentative])
            rem_centres   = det_centres[remaining_dets]
            cost          = np.linalg.norm(tent_centres[:, np.newaxis] - rem_centres[np.newaxis, :], axis=2)
            r_idx, c_idx  = linear_sum_assignment(cost)

            matched_tent = set()
            for ri, ci in zip(r_idx, c_idx):
                if cost[ri, ci] > self.max_distance:
                    continue
                self.tentative[ri].update(bboxes[remaining_dets[ci]])
                matched_tent.add(ri)
                matched_dets.add(remaining_dets[ci])

            for ti, t in enumerate(self.tentative):
                if ti not in matched_tent:
                    t.mark_missing()
        else:
            for t in self.tentative:
                t.mark_missing()

        # 4. Unmatched detections → new tentative tracks (if pool not locked)
        if not self.pool_locked:
            for i in range(len(bboxes)):
                if i not in matched_dets:
                    self.tentative.append(KalmanTrack(bboxes[i]))

        # 5. Promote tentative → confirmed if they have enough hits
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

        if self.next_id > self.num_fish:
            self.pool_locked = True
            self.tentative   = []  # no more candidates needed

        # 6. Remove tentative tracks that disappeared quickly (false positives)
        self._prune_tentative()

        return self._output()

    def _prune_tentative(self):
        self.tentative = [t for t in self.tentative if t.missing_frames <= self.max_tentative_missing]

    def _output(self):
        return [
            (tid, *[int(v) for v in t.bbox])
            for tid, t in self.confirmed.items()
            if t.hits >= 1
        ]


# ── Configuration ─────────────────────────────────────────────────────────────

with open('config.yaml') as f:
    cfg = yaml.safe_load(f)

input_video_path  = cfg['track_zebrafish']['input_video_path']
model_path        = cfg['track_zebrafish']['model_path']
output_video_path = cfg['track_zebrafish']['output_video_path']
start_seconds     = cfg['track_zebrafish']['start_seconds']
end_seconds       = cfg['track_zebrafish']['end_seconds']
num_fish          = cfg['track_zebrafish']['num_fish']

# ── Load model ────────────────────────────────────────────────────────────────

model   = YOLO(model_path)
tracker = ZebrafishTracker(num_fish=num_fish, max_distance=150, confirm_hits=15, max_tentative_missing=5)

# ── Open video ────────────────────────────────────────────────────────────────

os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
cap        = cv2.VideoCapture(input_video_path)
width      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps        = round(cap.get(cv2.CAP_PROP_FPS))
start_frame = start_seconds * fps
max_frames  = ((end_seconds - start_seconds) * fps) if end_seconds is not None else int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - start_frame
cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
out        = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

# ── Main loop ─────────────────────────────────────────────────────────────────

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret or frame_count >= max_frames:
        break

    results    = model(frame, verbose=False)
    detections = sv.Detections.from_ultralytics(results[0])
    detections = detections[detections.class_id == 0]  # danio_rerio only

    bboxes  = detections.xyxy.tolist() if len(detections) > 0 else []
    tracked = tracker.update(bboxes)

    for track_id, x1, y1, x2, y2 in tracked:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"Fish {track_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    out.write(frame)
    frame_count += 1
    if frame_count % 30 == 0:
        current_second = start_seconds + frame_count // fps
        print(f"Frame {frame_count}/{max_frames}  |  second {current_second}")

cap.release()
out.release()
print(f"Done. Saved to {output_video_path}")
