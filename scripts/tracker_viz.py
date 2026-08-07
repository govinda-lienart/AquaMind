"""
tracker_viz.py — overlay rendering for tracker.py: boxes, labels, status panel, swap-audit arrows.
Pure drawing on top of already-decided Track state — no tracking logic lives here.
"""

import cv2


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
            txt, col = f"Fish {t.id}: crossing...", (255, 255, 255)   # white — (0,220,255) collided with Fish 4's own ID color
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
            box_col = (0, 140, 255) if flagged else (255, 255, 255) if crossing else color   # white — (0,220,255) collided with Fish 4's own ID color
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
    if audit:                                                   # second pass: for each flagged fish, highlight the
        by_id = {t.id: t for t in tracks if t.id is not None}   # SPECIFIC partner it's suspected of swapping with —
        for t in tracks:                                        # not just the flagged fish's own box, since "who" is the point
            if not (t.audit_hold > 0 and t.audit_partner is not None):
                continue
            partner = by_id.get(t.audit_partner)
            if partner is None or partner.missing > 0:          # partner not visible this frame -> nothing to point at
                continue
            px1, py1, px2, py2 = [int(v) for v in partner.bbox]
            draw_dashed(frame, (px1, py1), (px2, py2), (0, 140, 255), dash=8)   # dashed orange = "implicated, not itself flagged"
            ax1, ay1, ax2, ay2 = [int(v) for v in t.bbox]
            c1 = ((ax1 + ax2) // 2, (ay1 + ay2) // 2)
            c2 = ((px1 + px2) // 2, (py1 + py2) // 2)
            cv2.arrowedLine(frame, c1, c2, (0, 140, 255), 2, tipLength=0.08)     # points FROM the flagged fish TO the suspected partner
            mid = ((c1[0] + c2[0]) // 2, (c1[1] + c2[1]) // 2)
            cv2.putText(frame, "SWAP?", (mid[0] - 30, mid[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 140, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Frame: {frame_count}", (10, frame.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    if locked and audit:
        draw_status_panel(frame, tracks, frame.shape[1])
    if not locked:
        draw_calibration_badge(frame, frame_count)
