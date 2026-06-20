"""
Backs up all LabelStudio projects as JSON to a dated folder.

Needs  : LABEL_STUDIO_URL and LABEL_STUDIO_API_KEY in .env
Output : labelstudio_backup/YYYY-MM-DD_HHMMSS/project_title.json
"""

# ── IMPORTS ───────────────────────────────────────────────────────────────────

import logging
import os
from datetime import datetime
from pathlib import Path

import requests
from dotenv import load_dotenv

BACKUP_DIR = "labelstudio_backup"

logger = logging.getLogger(__name__)

load_dotenv(Path(__file__).parent.parent / ".env")

LS_URL   = os.getenv("LABEL_STUDIO_URL")
LS_TOKEN = os.getenv("LABEL_STUDIO_API_KEY")

if not LS_TOKEN:
    raise RuntimeError("LABEL_STUDIO_API_KEY not found — check your .env file")


# ── AUTH ──────────────────────────────────────────────────────────────────────

def get_access_token():
    resp = requests.post(f"{LS_URL}/api/token/refresh", json={"refresh": LS_TOKEN})
    resp.raise_for_status()
    return resp.json()["access"]


HEADERS = {"Authorization": f"Bearer {get_access_token()}"}


# ── HELPERS ───────────────────────────────────────────────────────────────────

def get_all_projects():
    resp = requests.get(f"{LS_URL}/api/projects", headers=HEADERS)
    resp.raise_for_status()
    return [(p['id'], p['title']) for p in resp.json()['results']]


def export_project_json(project_id, title, backup_dir):
    resp = requests.get(
        f"{LS_URL}/api/projects/{project_id}/export?exportType=JSON",
        headers=HEADERS,
    )
    resp.raise_for_status()
    safe_title = title.replace(" ", "_")
    out_path   = os.path.join(backup_dir, f"{safe_title}.json")
    with open(out_path, "wb") as f:
        f.write(resp.content)
    logger.info(f"backed up '{title}' → {out_path}")
    return out_path


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    ts         = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    backup_dir = os.path.join(BACKUP_DIR, ts)
    os.makedirs(backup_dir, exist_ok=True)

    projects = get_all_projects()

    print("\n" + "─" * 50)
    print("  LABELSTUDIO BACKUP")
    print("─" * 50)
    print(f"  Destination : {backup_dir}")
    print(f"  Projects    : {len(projects)}")

    for project_id, title in projects:
        path = export_project_json(project_id, title, backup_dir)
        size = os.path.getsize(path) / 1024
        print(f"  ✓ {title:<40} {size:.0f} KB")

    print("─" * 50)
    print(f"  Done. {len(projects)} project(s) backed up.")
    print("─" * 50)


# ── ENTRY POINT ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    from scripts.logger import setup_logging
    setup_logging()
    main()
