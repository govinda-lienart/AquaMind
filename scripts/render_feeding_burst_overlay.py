"""
render_feeding_burst_overlay.py — live per-fish burst overlay on the tracked video.

usage:  python -m scripts.render_feeding_burst_overlay
"""

# ── imports ──────────────────────────────────────────────────────────────────


# ── config ──────────────────────────────────────────────────────────────────


# ── STEP 1 — load tracks + resolve the video path ────────────────────────────


# ── STEP 2 — per-fish speed + burst detection ────────────────────────────────
# same thresholds as feeding_activity.py: SMOOTH_WIN, BURST_MIN_CMS, BURST_PROM, BURST_GAP_S
# helper: bursts_up_to(fish_id, frame_number) -> running count
# helper: is_flickering(fish_id, frame_number) -> True if a burst just peaked


# ── STEP 3 — render the overlay video ────────────────────────────────────────
# open the tracked video, loop frames, draw counter + marker per fish, write output
