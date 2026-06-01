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
        self.bbox            = bbox
        self.origin          = (cx, cy)  # first detected position
        self.last_seen_bbox  = bbox      # last real detection (never updated during occlusion)
        self.is_lost         = False     # True when ghost box contains no detection
        self.lost_frames     = 0
        x1, y1, x2, y2      = bbox
        self.typical_w       = float(x2 - x1)
        self.typical_h       = float(y2 - y1)

    def displacement(self):
        cx, cy = self.state[:2]
        return np.sqrt((cx - self.origin[0])**2 + (cy - self.origin[1])**2)

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
        x1, y1, x2, y2      = bbox
        self.typical_w       = 0.05 * (x2 - x1) + 0.95 * self.typical_w
        self.typical_h       = 0.05 * (y2 - y1) + 0.95 * self.typical_h
        self.bbox            = bbox
        self.last_seen_bbox  = bbox  # update only on real detections
        self.hits           += 1
        self.missing_frames  = 0
        self.is_lost         = False
        self.lost_frames     = 0

    def mark_missing(self):
        self.missing_frames += 1
        self.state[2] *= 0.8  # damp vx
        self.state[3] *= 0.8  # damp vy

    def clip_to_typical(self, bbox, size_ratio=1.8):
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        if w * h > self.typical_w * self.typical_h * size_ratio:
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            hw, hh = self.typical_w / 2, self.typical_h / 2
            return [cx - hw, cy - hh, cx + hw, cy + hh]
        return bbox

    @staticmethod
    def iou(bbox_a, bbox_b):
        ax1, ay1, ax2, ay2 = bbox_a
        bx1, by1, bx2, by2 = bbox_b
        ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if inter == 0:
            return 0.0
        return inter / ((ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter)

    @property
    def predicted_bbox(self):
        cx, cy = self.state[:2]
        x1, y1, x2, y2 = self.last_seen_bbox
        hw = (x2 - x1) / 2
        hh = (y2 - y1) / 2
        return [cx - hw, cy - hh, cx + hw, cy + hh]

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

    def __init__(self, num_fish=5, max_distance=150, max_frame_step=60, confirm_hits=60, max_tentative_missing=5, min_displacement=30):
        self.num_fish              = num_fish
        self.max_distance          = max_distance
        self.max_frame_step        = max_frame_step
        self.confirm_hits          = confirm_hits          # frames needed to get a permanent ID
        self.max_tentative_missing = max_tentative_missing # frames before dropping a tentative track

        self.min_displacement = min_displacement
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
            conf_centres = np.array([
                KalmanTrack._centre(self.confirmed[tid].last_seen_bbox)
                if self.confirmed[tid].missing_frames > 30
                else self.confirmed[tid].predicted_centre
                for tid in conf_ids
            ])
            dist_cost = np.linalg.norm(conf_centres[:, np.newaxis] - det_centres[np.newaxis, :], axis=2)

            # IoU between each predicted box and each detection — better at crossings
            iou_mat = np.zeros((len(conf_ids), len(bboxes)))
            for i, tid in enumerate(conf_ids):
                for j, b in enumerate(bboxes):
                    iou_mat[i, j] = KalmanTrack.iou(self.confirmed[tid].predicted_bbox, b)

            # Where boxes overlap use IoU cost, otherwise fall back to distance
            cost     = np.where(iou_mat > 0, (1 - iou_mat) * self.max_distance, dist_cost)
            r_idx, c_idx = linear_sum_assignment(cost)

            matched_conf = set()
            for ri, ci in zip(r_idx, c_idx):
                if cost[ri, ci] > self.max_distance:
                    continue
                tid  = conf_ids[ri]
                bbox = bboxes[ci]
                # Reject if the fish would jump more than max_frame_step in one frame
                pcx, pcy   = self.confirmed[tid].predicted_centre
                dcx, dcy   = KalmanTrack._centre(bbox)
                frame_dist = np.linalg.norm([dcx - pcx, dcy - pcy])
                if frame_dist > self.max_frame_step:
                    continue
                # Clip oversized detection when another confirmed track is nearby
                other_centres = [self.confirmed[o].predicted_centre for o in conf_ids if o != tid]
                det_cx, det_cy = KalmanTrack._centre(bbox)
                if any(np.linalg.norm([det_cx - ox, det_cy - oy]) < self.max_distance
                       for ox, oy in other_centres):
                    bbox = self.confirmed[tid].clip_to_typical(bbox)
                self.confirmed[tid].update(bbox)
                matched_conf.add(tid)
                matched_dets.add(ci)

            for tid in conf_ids:
                if tid not in matched_conf:
                    self.confirmed[tid].mark_missing()

            # 2b. Classify unmatched confirmed as lost or occluded
            for tid in conf_ids:
                if tid not in matched_conf:
                    t = self.confirmed[tid]
                    max_iou = max((KalmanTrack.iou(t.predicted_bbox, b) for b in bboxes), default=0.0)
                    if max_iou < 0.1:
                        t.is_lost    = True
                        t.lost_frames += 1
                    else:
                        t.is_lost = False  # something is in the predicted area — occluded, not lost

            # 2c. Recovery pass: re-match lost tracks against unclaimed detections
            lost_ids       = [tid for tid in conf_ids if self.confirmed[tid].is_lost]
            unmatched_now  = [i for i in range(len(bboxes)) if i not in matched_dets]

            if lost_ids and unmatched_now:
                lost_centres = np.array([
                    KalmanTrack._centre(self.confirmed[tid].last_seen_bbox) for tid in lost_ids
                ])
                rem_centres = det_centres[unmatched_now]
                cost        = np.linalg.norm(lost_centres[:, np.newaxis] - rem_centres[np.newaxis, :], axis=2)
                r_idx, c_idx = linear_sum_assignment(cost)
                for ri, ci in zip(r_idx, c_idx):
                    if cost[ri, ci] > self.max_distance * 3:
                        continue
                    tid = lost_ids[ri]
                    print(f"  Fish {tid} recovered after {self.confirmed[tid].lost_frames} lost frames")
                    self.confirmed[tid].update(bboxes[unmatched_now[ci]])
                    matched_dets.add(unmatched_now[ci])

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
                if t.hits >= self.confirm_hits and t.displacement() >= self.min_displacement and self.next_id <= self.num_fish:
                    self.confirmed[self.next_id] = t
                    print(f"  Fish {self.next_id} confirmed after {t.hits} frames (displacement={t.displacement():.1f}px)")
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

    def tentative_boxes(self):
        return [[int(v) for v in t.bbox] for t in self.tentative]

    def lost_ids(self):
        return {tid for tid, t in self.confirmed.items() if t.is_lost}


# ── Configuration ─────────────────────────────────────────────────────────────

with open('config.yaml') as f:
    cfg = yaml.safe_load(f)

input_video_path    = cfg['track_zebrafish']['input_video_path']
model_path          = cfg['track_zebrafish']['model_path']
output_video_path   = cfg['track_zebrafish']['output_video_path']
start_seconds       = cfg['track_zebrafish']['start_seconds']
end_seconds         = cfg['track_zebrafish']['end_seconds']
num_fish            = cfg['track_zebrafish']['num_fish']
max_distance        = cfg['track_zebrafish']['max_distance']
max_frame_step      = cfg['track_zebrafish']['max_frame_step']
confirm_hits        = cfg['track_zebrafish']['confirm_hits']
min_displacement    = cfg['track_zebrafish']['min_displacement']
max_tentative_missing = cfg['track_zebrafish']['max_tentative_missing']

# ── Print configuration ───────────────────────────────────────────────────────

print("=" * 50)
print(f"  Video:            {input_video_path}")
print(f"  Model:            {model_path}")
print(f"  Output:           {output_video_path}")
print(f"  Seconds:          {start_seconds} → {end_seconds}")
print(f"  Fish:             {num_fish}")
print(f"  confirm_hits:     {confirm_hits}")
print(f"  min_displacement: {min_displacement}px")
print(f"  max_distance:     {max_distance}px")
print(f"  max_frame_step:   {max_frame_step}px")
print(f"  max_tent_missing: {max_tentative_missing}")
print("=" * 50)
input("Press Enter to confirm and start...")

# ── Load model ────────────────────────────────────────────────────────────────

model   = YOLO(model_path)
model.to('mps')
tracker = ZebrafishTracker(num_fish=num_fish, max_distance=max_distance, max_frame_step=max_frame_step, confirm_hits=confirm_hits, max_tentative_missing=max_tentative_missing, min_displacement=min_displacement)

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

calibration_frames = fps * 5  # 5 second calibration window

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

    in_calibration = frame_count < calibration_frames

    # draw tentative boxes in yellow during calibration
    if in_calibration:
        for x1, y1, x2, y2 in tracker.tentative_boxes():
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 215, 255), 1)

    # draw confirmed boxes — green if tracked, orange if lost and searching
    lost = tracker.lost_ids()
    for track_id, x1, y1, x2, y2 in tracked:
        colour = (0, 140, 255) if track_id in lost else (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
        label  = f"Fish {track_id} [?]" if track_id in lost else f"Fish {track_id}"
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)

    if in_calibration:
        cv2.putText(frame, "CALIBRATION", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 165, 255), 3)

    out.write(frame)
    frame_count += 1
    if frame_count == calibration_frames and not tracker.pool_locked:
        tracker.pool_locked = True
        tracker.tentative   = []
        print(f"  Calibration window closed — {len(tracker.confirmed)} fish confirmed")

    if frame_count % 30 == 0:
        current_second = start_seconds + frame_count // fps
        print(f"Frame {frame_count}/{max_frames}  |  second {current_second}")

cap.release()
out.release()
print(f"Done. Saved to {output_video_path}")
