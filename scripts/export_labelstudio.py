"""
Exports YOLO annotations from a LabelStudio project, or searches for a frame by name.

Needs  : LABEL_STUDIO_URL and LABEL_STUDIO_API_KEY in .env
Output : YOLO .txt files in OUTPUT_DIR/
"""

# ── IMPORTS ───────────────────────────────────────────────────────────────────

import logging
import os
import zipfile
from datetime import datetime #
from pathlib import Path

import requests
import yaml
from dotenv import load_dotenv

CONFIG_PATH = 'config.yaml'

with open(CONFIG_PATH) as f:
    _cfg = yaml.safe_load(f)['export_labelstudio']

PROJECT_NAME = _cfg['project_name']
EXPORT_TYPE  = _cfg['export_type']
MIN_TASK_ID  = _cfg.get('min_task_id') or None
MAX_TASK_ID  = _cfg.get('max_task_id') or None
MODE         = _cfg.get('mode', 'export')
SEARCH_TERM  = _cfg.get('search_term', '')

_video_name = PROJECT_NAME.split("_", 1)[1]
_ts         = datetime.now().strftime("%d%m%Y_%Hh%M")
OUTPUT_DIR  = f"labelstudio_export/labelstudio_{EXPORT_TYPE}_{_video_name}_{_ts}"

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

def get_project_id(name):
    resp = requests.get(f"{LS_URL}/api/projects", headers=HEADERS)
    resp.raise_for_status()
    for p in resp.json()['results']:
        if p['title'] == name:
            logger.info(f"project '{name}' → id={p['id']}")
            return p['id']
    raise ValueError(f"project '{name}' not found in Label Studio")


def fetch_all_tasks(project_id):
    url  = f"{LS_URL}/api/tasks?project={project_id}&page_size=1000"
    resp = requests.get(url, headers=HEADERS)
    resp.raise_for_status()
    return resp.json()['tasks']


def get_labeled_task_ids(project_id, min_id=None, max_id=None):
    tasks = fetch_all_tasks(project_id)
    ids   = [t['id'] for t in tasks if t['is_labeled']]
    if min_id:
        ids = [i for i in ids if i >= min_id]
    if max_id:
        ids = [i for i in ids if i <= max_id]
    logger.info(f"found {len(ids)} labeled tasks (min_id={min_id}, max_id={max_id})")
    return ids


def search_task(project_id, term):
    tasks = fetch_all_tasks(project_id)
    results = [t for t in tasks if term in t['data'].get('image', '')]
    if not results:
        print(f"no tasks found matching '{term}'")
        return
    for t in results:
        print(f"task id: {t['id']}  |  {t['data']['image']}  |  labeled: {t['is_labeled']}")


def export_yolo(project_id, task_ids, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    params = "exportType=YOLO&" + "&".join([f"ids[]={i}" for i in task_ids])
    url    = f"{LS_URL}/api/projects/{project_id}/export?{params}"
    logger.info(f"exporting {len(task_ids)} tasks as YOLO → {output_dir}")
    resp = requests.get(url, headers=HEADERS)
    resp.raise_for_status()
    zip_path = os.path.join(output_dir, "export.zip")
    with open(zip_path, "wb") as f:
        f.write(resp.content)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(output_dir)
    os.remove(zip_path)

    labels_dir       = os.path.join(output_dir, "labels")
    txt_files        = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]
    total_annotations = sum(
        1 for txt in txt_files
        for line in open(os.path.join(labels_dir, txt)).readlines()
        if len(line.split()) == 5
    )

    print("\n" + "─" * 50)
    print("  EXPORT SUMMARY")
    print("─" * 50)
    print(f"  Frames exported      : {len(txt_files)}")
    print(f"  Annotations          : {total_annotations}")
    print(f"  Output               : {output_dir}")
    print("─" * 50)


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    project_id = get_project_id(PROJECT_NAME)
    if MODE == "search":
        search_task(project_id, SEARCH_TERM)
    elif MODE == "export":
        task_ids = get_labeled_task_ids(project_id, MIN_TASK_ID, MAX_TASK_ID)
        export_yolo(project_id, task_ids, OUTPUT_DIR)
    else:
        raise ValueError(f"unknown MODE '{MODE}' — use 'export' or 'search'")


# ── ENTRY POINT ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    from scripts.logger import setup_logging
    setup_logging()
    main()
