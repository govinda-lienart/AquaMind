"""
Reads video metadata from an Excel sheet and registers videos in MySQL.

Input  : XLSX_PATH spreadsheet with one video per row
Needs  : MySQL running, videos table created
Output : rows inserted into videos table (skips already-registered entries)
"""

# ── IMPORTS ───────────────────────────────────────────────────────────────────

import logging

import openpyxl

from scripts.db import register_video


# ── CONSTANTS ─────────────────────────────────────────────────────────────────

logger = logging.getLogger(__name__)


# ── CONSTANTS ─────────────────────────────────────────────────────────────────

XLSX_PATH = 'video_metadata_2026_5_30.xlsx'


# ── HELPERS ───────────────────────────────────────────────────────────────────

def load_video_rows(xlsx_path):
    wb      = openpyxl.load_workbook(xlsx_path)
    ws      = wb.active
    headers = [cell.value for cell in ws[1]]
    return [dict(zip(headers, row)) for row in ws.iter_rows(min_row=2, values_only=True)]


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():

    rows    = load_video_rows(XLSX_PATH)
    added   = 0
    skipped = 0

    for data in rows:
        video_id = register_video(
            file_path      = data['file_path'],
            session_type   = data['session_type'],
            obstacles      = bool(data['obstacles']),
            fish_count     = data['fish_count'],
            notes          = data['notes'],
            species        = data['species'],
            morph          = data['morph'],
            tank_width_cm  = data['tank_width_cm'],
            tank_height_cm = data['tank_height_cm'],
            tank_depth_cm  = data['tank_depth_cm'],
            filmed_at      = data.get('filmed_at'),
        )

        if video_id:
            logger.info(f"added   {data['file_path']} → video_id={video_id}")
            added += 1
        else:
            logger.info(f"skipped {data['file_path']} (already registered)")
            skipped += 1

    logger.info(f"done — {added} added, {skipped} skipped")


# ── ENTRY POINT ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    from scripts.logger import setup_logging
    setup_logging()
    main()
