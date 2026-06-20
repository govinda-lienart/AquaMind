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


# ── AUTH ──────────────────────────────────────────────────────────────────────

def get_access_token(ls_url, ls_token):
    resp = requests.post(f"{ls_url}/api/token/refresh", json={"refresh": ls_token})
    resp.raise_for_status()
    return resp.json()["access"]


# ── HELPERS ───────────────────────────────────────────────────────────────────

def get_all_projects(ls_url, headers):
    resp = requests.get(f"{ls_url}/api/projects", headers=headers)
    resp.raise_for_status()
    return [(p['id'], p['title']) for p in resp.json()['results']]


def export_project_json(ls_url, project_id, title, backup_dir, headers):
    resp = requests.get(
        f"{ls_url}/api/projects/{project_id}/export?exportType=JSON",
        headers=headers,
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
    load_dotenv(Path(__file__).parent.parent / ".env")

    ls_url   = os.getenv("LABEL_STUDIO_URL")
    ls_token = os.getenv("LABEL_STUDIO_API_KEY")

    if not ls_token:
        raise RuntimeError("LABEL_STUDIO_API_KEY not found — check your .env file")

    headers = {"Authorization": f"Bearer {get_access_token(ls_url, ls_token)}"}

    ts         = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    backup_dir = os.path.join(BACKUP_DIR, ts)
    os.makedirs(backup_dir, exist_ok=True)

    projects = get_all_projects(ls_url, headers)

    print("\n" + "─" * 50)
    print("  LABELSTUDIO BACKUP")
    print("─" * 50)
    print(f"  Destination : {backup_dir}")
    print(f"  Projects    : {len(projects)}")

    for project_id, title in projects:
        path = export_project_json(ls_url, project_id, title, backup_dir, headers)
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
