"""
tracker.py — the fish tracker (SORT-style: velocity prediction + OC-SORT direction term).
Run:  python -m scripts.tracker
Output: a per-run folder <base>_<timestamp>/ holding the video, log, config sidecar, crops, and tracks.parquet
"""

import os
import json
import logging
import subprocess
from datetime import datetime

import cv2
import yaml
import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.optimize import linear_sum_assignment
from scripts.reid_features import load_backbone, transform   # shared DINOv2 embedder (appearance only)
from scripts.model_registry import load_yolo                 # path OR models:/...@champion → native YOLO

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

CONFIG_PATH   = 'config.yaml'
REACQUIRE_TAU = 15    # ghost search radius grows by one max_distance every 15 missing frames
VEL_SMOOTH    = 0.5   # EMA weight for the velocity estimate (0=frozen, 1=raw last step)
MIN_SPEED     = 3.0   # px/frame — below this a ghost's heading is noise, so DON'T coast it (stay at last position)
MAX_COAST     = 8     # frames — coast a ghost AT MOST this far (covers a crossing merge); past this it FREEZES and the
                      # WIDENING gate re-acquires. Frame-cap is what keeps a long occlusion from drifting into a permanent ghost.


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
        self.appearance = None      # running (unit-length) DINOv2 fingerprint — this fish's look, when appearance is ON
        self.match_geom = None      # debug: geometry distance (px) to the matched detection this frame
        self.match_app  = None      # debug: appearance cosine-distance to memory for the matched detection
        self.match_flipped = False  # debug: True if appearance CHANGED the match (geometry alone would pick a different fish)
        self.prev_appearance = None # snapshot of the memory BEFORE this frame (for the audit)
        self.exit_feat = None       # the (clean, separated) detection fingerprint this fish matched this frame
        self.audit_partner = None   # post-crossing audit: id of the fish this one's exit looks MORE like (suspected swap) — for overlay
        self.cross_memory = None    # clean fingerprint snapshotted when the fish ENTERED a crossing (its pre-crossing look)
        self.crossing = False       # currently close to (crossing) another fish -> audit its look once it separates again
        self.audit_hold = 0         # frames-remaining to keep the SWAP highlight up in the status panel

    @property
    def pred(self):
        """Predicted next-frame position under constant velocity — the anchor we match on.
        Using this instead of the last position is what stops two crossing fish swapping IDs:
        the fish overshoot past each other, so 'nearest last position' picks the WRONG detection,
        but 'nearest predicted position' picks the right one."""
        return (self.x + self.vx, self.y + self.vy)

    def predicted(self, cap):
        """COAST a ghost forward along its trajectory (your straight-line-momentum idea) so a fish lost
        mid-crossing is searched for where it WOULD be, not where it was lost. Triple-guarded so it can't
        destabilise: (1) near-stationary tracks don't coast (heading = noise); (2) capped in FRAMES
        (MAX_COAST) so a long occlusion doesn't drift off; (3) capped in DISTANCE so a bad velocity can't
        fling it across the tank. Healthy track (missing=0) -> just the 1-frame pred."""
        speed = (self.vx ** 2 + self.vy ** 2) ** 0.5
        if speed < MIN_SPEED:
            return (self.x, self.y)
        frames = min(self.missing + 1, MAX_COAST)
        dx, dy = self.vx * frames, self.vy * frames
        dist = (dx ** 2 + dy ** 2) ** 0.5
        if dist > cap:
            dx, dy = dx * cap / dist, dy * cap / dist
        return (self.x + dx, self.y + dy)

    def update(self, x, y, bbox, confidence, feat=None, freeze_vel=False):
        frames  = self.missing + 1                    # frames elapsed since the last real detection
        if not freeze_vel:                            # freeze through a MERGE: the blob centroid is BOTH fish, so
            inst_vx = (x - self.x) / frames           # recomputing velocity from it BENDS the trajectory -> swap at split.
            inst_vy = (y - self.y) / frames           # keeping the clean pre-merge velocity lets both coast + re-sort correctly.
            self.vx = VEL_SMOOTH * inst_vx + (1 - VEL_SMOOTH) * self.vx   # EMA — smooth out detection jitter
            self.vy = VEL_SMOOTH * inst_vy + (1 - VEL_SMOOTH) * self.vy
        self.x, self.y, self.bbox = x, y, bbox
        self.confidence = confidence
        self.hits   += 1
        self.missing = 0
        if feat is not None:                          # EMA-update the appearance memory, keep it unit-length
            self.appearance = feat if self.appearance is None else 0.9 * self.appearance + 0.1 * feat
            self.appearance = self.appearance / (np.linalg.norm(self.appearance) + 1e-8)

    def mark_missing(self):
        self.missing += 1


def box_gap(bbox_a, bbox_b):
    """Shortest distance between two axis-aligned xyxy boxes' EDGES (0 if touching/overlapping) —
    unlike center-to-center distance, this doesn't fire early just because two fish have big boxes."""
    ax1, ay1, ax2, ay2 = bbox_a
    bx1, by1, bx2, by2 = bbox_b
    dx = max(bx1 - ax2, ax1 - bx2, 0)   # horizontal gap; 0 if they overlap in x
    dy = max(by1 - ay2, ay1 - by2, 0)   # vertical gap; 0 if they overlap in y
    return np.hypot(dx, dy)


# ── the whole tracker in one function: match tracks → detections ───────────────
def associate(tracks, dets, det_pos, max_distance, det_feats=None, app_weight=0.0, app_min_sep=150, app_gate=None, app_max_gap=None, merge_fix=False, debug=False):
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

    APPEARANCE GATE (app_gate) — a VETO, not just a tie-breaker: a track REFUSES a
    detection whose appearance cosine-distance to its memory exceeds app_gate, even
    if geometry is fine. That track stays unmatched (ghosts) so the RIGHT track can
    claim the detection — this is what stops a track locking onto a wrong-but-near
    fish after a merge (issue #2). Only applied to tracks that HAVE an appearance memory.

    Unmatched tracks are marked missing. Returns matched detection indices.
    """
    matched_dets, matched_tracks = set(), set()
    for t in tracks:                                          # reset per-frame debug values + snapshot clean pre-frame memory for the audit
        t.match_geom = t.match_app = None; t.match_flipped = False
        t.prev_appearance = t.appearance; t.exit_feat = None   # audit_partner persists via audit_hold (managed in main)
    if not tracks or not len(dets):
        for t in tracks:
            t.mark_missing()
        return matched_dets

    # which detections are WELL-SEPARATED from all others? only those are safe to update appearance from
    # (a crop near another fish is overlap-contaminated — updating memory with it poisons the fingerprint)
    if len(det_pos) > 1:
        dd = np.linalg.norm(det_pos[:, None] - det_pos[None, :], axis=2)
        np.fill_diagonal(dd, np.inf)                          # ignore self-distance
        det_separated = dd.min(axis=1) >= app_min_sep         # True = nearest other fish is far enough
    else:
        det_separated = np.ones(len(det_pos), dtype=bool)     # a lone detection can't be contaminated

    # MERGE detected = fewer boxes than confirmed fish. During a merge, NEITHER involved fish should
    # snap onto the blob (its centroid is both fish) — both coast on clean momentum and re-sort at the split.
    n_confirmed = sum(1 for t in tracks if t.id is not None)
    is_merge = merge_fix and 0 < len(dets) < n_confirmed

    def _match(track_idx, gate_fn):
        avail = [c for c in range(len(dets)) if c not in matched_dets]
        if not track_idx or not avail:
            return
        # merge_fix: COAST ghosts along their momentum (frame-capped) so a fish lost mid-crossing is searched
        # for where it WOULD be; else the plain 1-frame prediction. Healthy tracks coast 1 frame either way.
        anchors  = np.array([(tracks[i].predicted(max_distance * 3) if merge_fix else tracks[i].pred)
                             for i in track_idx])
        pos      = det_pos[avail]
        pos_cost = np.linalg.norm(anchors[:, None] - pos[None, :], axis=2)  # geometry (pixels) — used for the GATE
        assign_cost  = pos_cost.copy()                             # geometry (+ appearance) — used for the ASSIGNMENT
        app_dist_mat = np.full_like(pos_cost, np.nan)              # per-pair cosine distance (NaN = track has no memory -> don't gate it)
        if det_feats is not None and app_weight > 0:
            for r, ti in enumerate(track_idx):
                if tracks[ti].appearance is None:                  # no memory yet -> geometry only for this track
                    continue
                if app_max_gap is not None and tracks[ti].missing > app_max_gap:   # memory too STALE (hidden > ~hold-time) -> don't trust appearance, geometry only
                    continue
                for k, c in enumerate(avail):
                    app_dist = 1.0 - float(np.dot(tracks[ti].appearance, det_feats[c]))   # cosine distance (feats are unit-length)
                    app_dist_mat[r, k] = app_dist
                    assign_cost[r, k] += app_weight * app_dist
        # debug: what would GEOMETRY ALONE have chosen? (to flag when appearance FLIPPED the match)
        geo_choice = {}
        if debug:
            for gr, gc in zip(*linear_sum_assignment(pos_cost)):
                geo_choice[track_idx[gr]] = avail[gc]
        for r, c in zip(*linear_sum_assignment(assign_cost)):
            ti = track_idx[r]
            if pos_cost[r, c] > gate_fn(tracks[ti]):               # GEOMETRY gate — a tag can never teleport onto a far fish
                continue
            if app_gate is not None and not np.isnan(app_dist_mat[r, c]) and app_dist_mat[r, c] > app_gate:
                continue                                           # APPEARANCE gate (veto): "that's not me" -> stay unmatched (ghost), let the right track claim it
            x, y, bbox, conf = dets[avail[c]]
            sep = det_separated[avail[c]]
            if is_merge:                                           # MERGE: if 2+ fish predict onto this ONE box, it's a blob (both fish) —
                dx0, dy0 = det_pos[avail[c]]                       # don't snap either onto it; let them coast + re-sort at the split
                n_near = sum(1 for tt in tracks if tt.id is not None
                             and np.hypot(dx0 - tt.pred[0], dy0 - tt.pred[1]) < max_distance)
                if n_near > 1:
                    continue
            if debug:                                              # record WHAT DROVE this match, for the overlay
                tracks[ti].match_geom = float(pos_cost[r, c])
                a = app_dist_mat[r, c]
                tracks[ti].match_app = None if np.isnan(a) else float(a)
                tracks[ti].match_flipped = (geo_choice.get(ti) != avail[c])   # appearance changed the pick
            # only feed appearance into memory when this detection is WELL-SEPARATED (else keep the clean pre-crossing memory)
            feat = det_feats[avail[c]] if (det_feats is not None and sep) else None
            if feat is not None:                                   # clean exit fingerprint — kept for the post-crossing swap audit
                tracks[ti].exit_feat = feat
            # merge_fix: if this detection is a BLOB (not separated), freeze velocity so the merge doesn't bend the trajectory
            tracks[ti].update(x, y, bbox, conf, feat, freeze_vel=(merge_fix and not sep))
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


def draw_status_panel(frame, tracks, W):
    """Stable top-right list of all fish: 'Fish N: OK', or an orange 'Fish N: SWAP? -> M' while a
    suspected swap is highlighted (held ~2s by audit_hold)."""
    conf = sorted([t for t in tracks if t.id is not None], key=lambda t: t.id)
    if not conf:
        return
    pad, rowh = 12, 26
    w, h = 230, rowh * (len(conf) + 1) + pad
    x0, y0 = W - w - 12, 55
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + w, y0 + h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)
    cv2.putText(frame, "FISH STATUS", (x0 + pad, y0 + rowh - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    for i, t in enumerate(conf):
        y = y0 + rowh * (i + 2) - 6
        if t.audit_hold > 0:                                   # flagged a suspected swap (held ~2s)
            txt, col = f"Fish {t.id}: SWAP? -> {t.audit_partner}", (0, 140, 255)
        elif t.crossing:                                       # currently crossing another fish -> identity unverifiable
            txt, col = f"Fish {t.id}: crossing...", (0, 220, 255)
        else:                                                  # alone + identity confirmed
            txt, col = f"Fish {t.id}: safe", (80, 220, 80)
        cv2.putText(frame, txt, (x0 + pad, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, col, 2)


def draw_frame(frame, tracks, locked, frame_count, debug=False, audit=False):
    for t in tracks:
        x1, y1, x2, y2 = [int(v) for v in t.bbox]
        color = id_color(t.id)
        flagged  = audit and t.audit_hold > 0                  # suspected-swap fish -> thick orange
        crossing = audit and t.crossing and not flagged        # currently crossing -> yellow
        if locked and t.missing > 0:
            draw_dashed(frame, (x1, y1), (x2, y2), color)
            label = f"Fish {t.id} (lost)"
        else:
            box_col = (0, 140, 255) if flagged else (0, 220, 255) if crossing else color
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_col, 4 if flagged else 2)
            label = f"Fish {t.id}" if t.id else ""
        if label:
            cv2.putText(frame, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        # debug: WHY this match — g=geometry px, a=appearance cosine-dist; 'APP' (red) = appearance FLIPPED the pick
        if debug and t.match_geom is not None:
            a_txt = f" a{t.match_app:.2f}" if t.match_app is not None else ""
            cv2.putText(frame, f"g{t.match_geom:.0f}{a_txt}", (x1, y2 + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            if t.match_flipped:
                cv2.putText(frame, "APP", (x1, y2 + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(frame, f"Frame: {frame_count}", (10, frame.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    if locked and audit:
        draw_status_panel(frame, tracks, frame.shape[1])
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


# ── appearance: one DINOv2 fingerprint per detection (only when appearance is ON) ──
def embed_detections(frame, dets, backbone, device, head=None):
    """L2-normalized appearance vector per detection: BGR crop -> RGB -> transform -> backbone [-> head].
       With head=None it's the RAW DINOv2 fingerprint; with a trained head it's the DISCRIMINATIVE
       (identity-focused) projection — much better at telling your fish apart than raw cosine."""
    if not dets:
        return None
    crops = []
    for _cx, _cy, bbox, _conf in dets:
        x1, y1, x2, y2 = [max(0, int(v)) for v in bbox]
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:                                   # guard a degenerate box
            crop = frame[0:2, 0:2]
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)          # cv2 is BGR; DINOv2 wants RGB
        crops.append(transform(Image.fromarray(rgb)))
    batch = torch.stack(crops).to(device)
    with torch.no_grad():
        feats = backbone(batch)
        if head is not None:
            feats = head(feats)                              # project into the trained head's discriminative space
    feats = torch.nn.functional.normalize(feats, dim=1)      # unit-length -> cosine similarity == dot product
    return feats.cpu().numpy()


# ── main ───────────────────────────────────────────────────────────────────────
def main():
    c = yaml.safe_load(open(CONFIG_PATH))['tracker']
    input_video_path  = c['input_video_path']
    model_path        = c['model_path']
    num_fish          = c['num_fish']
    max_distance      = c['max_distance']
    calibration_secs  = c['calibration_secs']
    start = c.get('start_seconds') or 0
    end   = c.get('end_seconds')
    use_appearance = bool(c.get('appearance', False))                       # fuse DINOv2 appearance into matching?
    app_weight     = float(c.get('appearance_weight', 100)) if use_appearance else 0.0
    app_min_sep    = float(c.get('appearance_min_sep', 150))                 # only update appearance memory when a fish is this far from all others (px)
    app_gate       = c.get('appearance_gate')                               # VETO: refuse a match whose cosine-dist to memory exceeds this (None = off)
    app_gate       = float(app_gate) if (use_appearance and app_gate is not None) else None
    app_max_gap_secs = float(c.get('appearance_max_gap_secs', 2.5))          # trust appearance only within this many seconds (measured ~2s hold-time); staler ghost memory -> geometry only
    merge_fix        = bool(c.get('merge_fix', False))                       # coast ghosts on frozen pre-merge velocity + freeze velocity through a blob -> hold IDs through a detector MERGE crossing
    app_debug        = bool(c.get('appearance_debug', False)) and use_appearance   # draw g/a per fish + flag when appearance FLIPPED a match
    app_audit        = bool(c.get('appearance_audit', False)) and use_appearance    # post-crossing swap AUDIT: after a fish exits a crossing, does its look match its OWN pre-crossing memory or another fish's?
    audit_margin     = float(c.get('appearance_audit_margin', 0.03))                # flag if the exit is this-much-more similar to ANOTHER fish's memory than its own (relative check)

    # build a per-run output FOLDER (holds the video, log, and config sidecar)
    clean    = c['output_video_path'].rstrip('/,. ')          # tolerate trailing junk
    base     = os.path.splitext(os.path.basename(clean))[0]
    stamp    = datetime.now().strftime('%Y_%m_%d_%H%M')       # auto timestamp → a fresh folder every run
    run_name = f'{base}_{stamp}'                              # e.g. 'tracker_IMG_1839_2026_07_23_1642'
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

    model = load_yolo(model_path)

    # appearance embedder — loaded ONLY when appearance is ON. If a trained head is given, detections
    # are projected through backbone -> head (discriminative); otherwise raw DINOv2 (weak cosine).
    backbone = app_device = app_head = None
    if use_appearance:
        app_device = "mps" if torch.backends.mps.is_available() else "cpu"
        head_path  = c.get('appearance_head')
        if head_path:
            hd = torch.load(head_path, map_location=app_device)
            app_backbone_name = hd["backbone"]                      # match the backbone the head was trained on
            if "out_dim" in hd:                                     # CONTRASTIVE projection head (MLP) — discriminative, short-range identity metric
                app_head = torch.nn.Sequential(
                    torch.nn.Linear(hd["in_dim"], hd["hidden_dim"]), torch.nn.ReLU(),
                    torch.nn.Linear(hd["hidden_dim"], hd["out_dim"]))
                app_head.load_state_dict({k.replace("net.", ""): v for k, v in hd["head_state"].items()})
            else:                                                  # train_reid CLASSIFIER head (single Linear)
                app_head = torch.nn.Linear(hd["feat_dim"], hd["n_classes"])
                app_head.load_state_dict(hd["head_state"])
            app_head.eval().to(app_device)
        else:
            app_backbone_name = c.get('appearance_backbone', 'dinov2_vits14')
        backbone = load_backbone(app_backbone_name, device=app_device)
        logger.info(f"  appearance ON — {app_backbone_name}{' + trained head' if app_head else ' (raw)'} on {app_device}, weight={app_weight}")

    cap = cv2.VideoCapture(input_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    app_max_gap = int(app_max_gap_secs * fps) if use_appearance else None    # hold-time as frames (fps known now)
    if use_appearance:
        logger.info(f"  appearance time-gate: trust memory up to {app_max_gap} frames (~{app_max_gap_secs}s)")

    if start:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(start * fps))
    max_frames  = int((end - start) * fps) if end else None
    # total frames we'll actually process — the progress denominator (whole video if no end_seconds)
    total_frames = max_frames if max_frames else int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - int(start * fps)
    calib_frames = int(calibration_secs * fps)

    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))

    # ── collect this run's tracks in memory, write to parquet at the end (no MySQL) ──
    track_rows = []   # one dict per (frame, fish); saved as tracks.parquet in this run's output folder

    tracks, next_id, locked, frame_count = [], 1, False, 0
    app_flip_count = 0                                        # how many matches appearance CHANGED vs geometry alone (debug)
    app_audit_count = 0                                       # how many post-crossing swaps the appearance audit FLAGGED

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

        # appearance fingerprints for THIS frame's detections (None when appearance is OFF)
        det_feats = embed_detections(frame, dets, backbone, app_device, app_head) if use_appearance else None

        if not locked:
            # CALIBRATION: match, then spawn a new track for any leftover detection
            matched = associate(tracks, dets, det_pos, max_distance, det_feats, app_weight, app_min_sep, app_gate, app_max_gap, merge_fix, app_debug)
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
            associate(tracks, dets, det_pos, max_distance, det_feats, app_weight, app_min_sep, app_gate, app_max_gap, merge_fix, app_debug)
            for t in tracks:                                   # log lost/recovered transitions
                was, now = prev_missing[t.id], t.missing
                if was == 0 and now > 0:                       # first frame a fish drops out
                    logger.info(json.dumps({"event": "occlusion_lost", "fish_ids": str(t.id),
                                            "frame": frame_count}))
                elif was > 0 and now == 0:                     # fish reappears — was = frames it was missing
                    logger.info(json.dumps({"event": "occlusion_recovery", "fish_ids": str(t.id),
                                            "decision": "recovered", "frame": frame_count,
                                            "missing_frames": was}))
                if app_debug and t.match_flipped:              # appearance CHANGED this match — the direct "does appearance matter" signal
                    app_flip_count += 1
                    logger.info(json.dumps({"event": "appearance_flip", "fish_id": t.id, "frame": frame_count,
                                            "g": round(t.match_geom, 1), "a": round(t.match_app, 3) if t.match_app else None}))

            if app_audit:                                      # PROXIMITY-TRIGGERED SWAP AUDIT — fires whenever two boxes cross, checks when they separate
                def mem(t):
                    return t.cross_memory if t.cross_memory is not None else t.prev_appearance
                conf = [t for t in tracks if t.id is not None]
                for A in conf:
                    if A.audit_hold > 0:                       # count down the 2-second SWAP highlight
                        A.audit_hold -= 1
                        if A.audit_hold == 0:
                            A.audit_partner = None
                    near = any(box_gap(A.bbox, B.bbox) <= 0 for B in conf if B.id != A.id)   # boxes actually touching/overlapping, not just "nearby"
                    if near:
                        if not A.crossing:                     # ENTERING a crossing -> snapshot the clean pre-crossing look
                            A.crossing = True; A.cross_memory = A.prev_appearance
                    elif A.crossing:                           # just SEPARATED -> audit once it has a clean look
                        if A.exit_feat is None or mem(A) is None:
                            continue                           # not a clean look yet -> stay armed
                        A.crossing = False
                        self_d = 1.0 - float(np.dot(A.exit_feat, mem(A)))
                        others = [(1.0 - float(np.dot(A.exit_feat, mem(B))), B)
                                  for B in conf if B.id != A.id and mem(B) is not None]
                        if not others:
                            continue
                        cross_d, B = min(others, key=lambda x: x[0])
                        if self_d - cross_d > audit_margin:    # exit looks MORE like B than itself -> suspected swap
                            A.audit_partner = B.id; A.audit_hold = int(2 * fps); app_audit_count += 1
                            logger.info(json.dumps({"event": "appearance_swap_flag", "frame": frame_count,
                                                    "fish_id": A.id, "suspected_swap_with": B.id,
                                                    "self_dist": round(self_d, 3), "cross_dist": round(cross_d, 3),
                                                    "margin": round(self_d - cross_d, 3)}))

        
        # collect this frame's identified tracks for the parquet (only tracks that already have an id;
        # x/y are the centroid, occluded = ghost, confidence from the last matched detection)
        timestamp = start + frame_count / fps
        for t in tracks:
            if t.id is None:                         # tentative tracks (pre-lock) aren't persisted
                continue
            track_rows.append({                      # one row per identified track this frame (parquet schema)
                "frame_number": frame_count, "timestamp": timestamp, "fish_id": t.id,
                "x": t.x, "y": t.y, "confidence": t.confidence, "occluded": int(t.missing > 0),
            })

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
        draw_frame(frame, tracks, locked, frame_count, app_debug, app_audit)
        out.write(frame)

        cv2.imshow('Fish Tracker', frame)   # live preview
        key = cv2.waitKey(1) & 0xFF                  # refresh window; read any key press
        if key == ord('q') or key == 27:            # q or ESC (27) quits early
            logger.info("quit requested - stopping early")
            break

        if frame_count % 60 == 0:
            logger.info(f"  frame {frame_count}/{total_frames} | tracks={len(tracks)} | locked={locked}")
        frame_count += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()                          # close preview window

    # write this run's tracks to parquet (schema identical to the Stage 7 parquets — no MySQL)
    tracks_df = pd.DataFrame(track_rows, columns=["frame_number", "timestamp", "fish_id", "x", "y", "confidence", "occluded"])
    tracks_path = os.path.join(run_dir, "tracks.parquet")
    tracks_df.to_parquet(tracks_path)
    logger.info(f"saved {len(tracks_df)} track rows -> {tracks_path}")
    if app_debug:                                            # the direct answer to "does appearance still matter?"
        logger.info(f"APPEARANCE flipped the match in {app_flip_count} track-frames "
                    f"(0 = appearance never changed a decision -> geometry alone suffices; you could set appearance:false)")
    if app_audit:                                            # the auditor's yield — measure precision (real swaps / flagged) before trusting it to CORRECT
        logger.info(f"APPEARANCE AUDIT flagged {app_audit_count} post-crossing SUSPECTED SWAPS "
                    f"(check these frames against the video; if precision is high, promote to auto-correct)")
    logger.info(f"\nDone. Saved to {out_path}")


if __name__ == '__main__':
    main()
