"""
tracker_diagnostics.py — run startup logging + post-run diagnostic plots for tracker.py.
Reporting only: nothing here feeds back into tracking decisions.

Uses the SAME logger object as tracker.py ('scripts.tracker', not __name__) so that setup_run_logging's
file+console handlers — wherever they get attached from — apply to every log call across both files.
"""

import os
import json
import logging

import matplotlib
matplotlib.use('Agg')                                         # headless — no popup window
import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger('scripts.tracker')


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
                     num_fish, calibration_secs, max_distance, reacquire_tau,
                     merge_fix=False, use_appearance=False, appearance_weight=None,
                     appearance_head=None, appearance_backbone=None, appearance_max_gap_secs=None,
                     save_crops_in_appearance_mode=False):
    logger.info("=" * 50)
    logger.info(f"  Video:          {input_video_path}")
    logger.info(f"  Model:          {model_path}")
    logger.info(f"  Output:         {output_video_path}")
    logger.info(f"  Seconds:        {start_seconds} → {end_seconds}")
    logger.info(f"  Fish:           {num_fish}")
    logger.info(f"  Calibration:    {calibration_secs} seconds")
    logger.info(f"  max_distance:   {max_distance} px")
    logger.info(f"  reacquire_tau:  {reacquire_tau} frames")
    logger.info(f"  merge_fix:      {merge_fix}")
    logger.info(f"  appearance:     {use_appearance}")
    if use_appearance:
        logger.info(f"    weight:       {appearance_weight}")
        logger.info(f"    backbone:     {appearance_backbone}")
        logger.info(f"    head:         {appearance_head or '(none — raw DINOv2)'}")
        logger.info(f"    max_gap_secs: {appearance_max_gap_secs}")
        logger.info(f"    save_crops:   {save_crops_in_appearance_mode}")
    logger.info("=" * 50)
    input("Press Enter to start...")


# ── appearance diagnostic plots (only populated when appearance_debug/appearance_audit are on) ──
def plot_appearance_timeline(log_path, run_dir, num_fish):
    """Reads THIS run's own log file and plots WHEN appearance actually mattered: a tick each time it
    tie-broke a live match (appearance_flip), a star each time the post-crossing audit flagged a
    suspected swap (appearance_swap_flag), with occlusion windows shaded for context (fish naturally
    cluster near crossings — this is the visual proof, not just the summary count)."""
    flips, swaps, lost_at, occlusions = [], [], {}, {}
    with open(log_path) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue                                       # plain progress/info lines aren't JSON — skip them
            ev = rec.get('event')
            if ev == 'appearance_flip':
                flips.append((rec['frame'], rec['fish_id']))
            elif ev == 'appearance_swap_flag':
                swaps.append((rec['frame'], rec['fish_id']))
            elif ev == 'occlusion_lost':
                lost_at[rec['fish_ids']] = rec['frame']
            elif ev == 'occlusion_recovery':
                fid = rec['fish_ids']
                if fid in lost_at:
                    occlusions.setdefault(int(fid), []).append((lost_at.pop(fid), rec['frame']))

    if not flips and not swaps:
        logger.info("appearance timeline: no flip/swap events to plot (appearance off, or never mattered this run)")
        return

    fig, ax = plt.subplots(figsize=(12, 0.6 * num_fish + 1.5))
    for fid in range(1, num_fish + 1):
        ax.axhline(fid, color='#eeeeee', linewidth=6, zorder=0)                   # lane background
        for lo, hi in occlusions.get(fid, []):
            ax.axvspan(lo, hi, ymin=(fid - 0.4) / (num_fish + 1), ymax=(fid + 0.4) / (num_fish + 1),
                       color='orange', alpha=0.25, zorder=1)                      # shaded = fish was occluded/crossing here

    if flips:
        fx, fy = zip(*flips)
        ax.scatter(fx, fy, marker='|', s=200, color='#1f77b4', label='appearance flip (tie-broke a match)', zorder=2)
    if swaps:
        sx, sy = zip(*swaps)
        ax.scatter(sx, sy, marker='*', s=250, color='red', label='swap flagged by audit', zorder=3)

    ax.set_yticks(range(1, num_fish + 1))
    ax.set_yticklabels([f'Fish {i}' for i in range(1, num_fish + 1)])
    ax.set_xlabel('Frame')
    ax.set_title('When did appearance matter? (orange = occluded/crossing)')
    ax.legend(loc='upper right', fontsize=8)
    ax.invert_yaxis()
    plt.tight_layout()

    out_path = os.path.join(run_dir, 'appearance_timeline.png')
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    logger.info(f"appearance timeline saved -> {out_path}")


def plot_force_diagram(debug_rows, run_dir, num_fish, max_distance):
    """The two FORCES behind every match, continuously, not just the moments one of them won:
    geometry cost (px to the matched detection) vs appearance's WEIGHTED contribution (app_weight *
    cosine distance, same px units — directly comparable). A flip only happens where the appearance
    line is large enough, relative to geometry, to change which detection wins the argmin — this is
    that threshold made visible, instead of just the yes/no outcome."""
    if not debug_rows:
        logger.info("force diagram: no debug rows to plot (appearance_debug was off, or nothing ever matched)")
        return
    df = pd.DataFrame(debug_rows)

    fig, axes = plt.subplots(num_fish, 1, figsize=(13, 2.1 * num_fish), sharex=True)
    if num_fish == 1:
        axes = [axes]
    for i, fid in enumerate(range(1, num_fish + 1)):
        ax = axes[i]
        sub = df[df.fish_id == fid].sort_values('frame')
        if sub.empty:
            ax.set_ylabel(f'Fish {fid}')
            continue
        ax.plot(sub.frame, sub.geom, color='#1f77b4', linewidth=1, label='geometry cost (px)')
        ax.plot(sub.frame, sub.app_weighted, color='#ff7f0e', linewidth=1, label='appearance contribution (weighted, px)')
        ax.axhline(max_distance, color='grey', linestyle='--', linewidth=0.8, label='gate (max_distance)')
        flips = sub[sub.flipped]
        if not flips.empty:                                    # mark exactly where appearance's line actually WON the argmin
            ax.scatter(flips.frame, flips.geom, color='red', s=18, zorder=3, label='flip (appearance won)')
        ax.set_ylabel(f'Fish {fid}', rotation=0, labelpad=30, va='center')
        if i == 0:
            ax.legend(loc='upper right', fontsize=7, ncol=4)
    axes[-1].set_xlabel('Frame')
    fig.suptitle('The two forces behind every match: geometry vs appearance (weighted, same px units)')
    plt.tight_layout()

    out_path = os.path.join(run_dir, 'force_diagram.png')
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    logger.info(f"force diagram saved -> {out_path}")


def plot_margin_diagram(margin_rows, run_dir, num_fish):
    """Does appearance act as a CONTINUOUS force, not just a flip-or-not decision? For every match, this
    plots how much the winning detection beat the runner-up by — geometry alone (blue) vs the full combined
    cost (orange) — even on frames where the winner never changes. If orange consistently sits ABOVE blue,
    appearance is quietly widening the safety margin around an already-correct pick; a flip (red dot) is
    just the extreme case where that widening pulls so hard a DIFFERENT candidate becomes the winner."""
    if not margin_rows:
        logger.info("margin diagram: no margin rows to plot (appearance_debug was off, or no frame ever had 2+ candidates)")
        return
    df = pd.DataFrame(margin_rows)

    fig, axes = plt.subplots(num_fish, 1, figsize=(13, 2.1 * num_fish), sharex=True)
    if num_fish == 1:
        axes = [axes]
    for i, fid in enumerate(range(1, num_fish + 1)):
        ax = axes[i]
        sub = df[df.fish_id == fid].sort_values('frame')
        if sub.empty:
            ax.set_ylabel(f'Fish {fid}')
            continue
        ax.plot(sub.frame, sub.margin_geom, color='#1f77b4', linewidth=1, label='margin, geometry alone (px)')
        ax.plot(sub.frame, sub.margin_combined, color='#ff7f0e', linewidth=1, label='margin, geometry + appearance (px)')
        flips = sub[sub.flipped]
        if not flips.empty:
            ax.scatter(flips.frame, flips.margin_combined, color='red', s=18, zorder=3, label='flip (the extreme case)')
        ax.set_ylabel(f'Fish {fid}', rotation=0, labelpad=30, va='center')
        if i == 0:
            ax.legend(loc='upper right', fontsize=7, ncol=4)
    axes[-1].set_xlabel('Frame')
    fig.suptitle('Is appearance a continuous force? Winner-vs-runner-up margin, geometry alone vs combined')
    plt.tight_layout()

    out_path = os.path.join(run_dir, 'margin_diagram.png')
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    logger.info(f"margin diagram saved -> {out_path}")
