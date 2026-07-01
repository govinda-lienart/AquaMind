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
import requests
import yaml
from dotenv import load_dotenv

CONFIG_PATH = 'config.yaml'

with open(CONFIG_PATH) as f:
    _cfg = yaml.safe_load(f)['download_labelstudio']

PROJECT_NAME   = _cfg['project_name']
MIN_TASK_ID    = _cfg.get('min_task_id') or None
MAX_TASK_ID    = _cfg.get('max_task_id') or None
MODE           = _cfg.get('mode', 'export')
SEARCH_TERM    = _cfg.get('search_term', '')
EXPORT_LABELS  = _cfg.get('export_labels', True)
EXPORT_IMAGES  = _cfg.get('export_images', False)

_video_name = PROJECT_NAME.split("_", 1)[1]
_ts         = datetime.now().strftime("%d%m%Y_%Hh%M")
OUTPUT_DIR  = f"labelstudio_download/labelstudio_{_video_name}_{_ts}"

logger = logging.getLogger(__name__)

load_dotenv()

LS_URL   = os.getenv("LABEL_STUDIO_URL")
LS_TOKEN = os.getenv("LABEL_STUDIO_API_KEY")

if not LS_TOKEN:
    raise RuntimeError("LABEL_STUDIO_API_KEY not found — check your .env file")


# ── AUTH ──────────────────────────────────────────────────────────────────────

def get_access_token() -> str:
    """Exchanges the long-lived refresh token for a short-lived access token."""
    resp = requests.post(f"{LS_URL}/api/token/refresh", json={"refresh": LS_TOKEN}) # showing the permanent ID - apply for "day pass"
    resp.raise_for_status() # check for errors
    return resp.json()["access"] # extract the access token from json response file access e.g {"access": "eyJhbGciOiJIUz........, "refresh": "eyJhbGciO....}. The long eyJ...JWT tokens...contains encoded data like user id, expirty time...


HEADERS = {"Authorization": f"Bearer {get_access_token()}"} # store in HEADERS of the HTTP request - the "day pass"


# ── HELPERS ───────────────────────────────────────────────────────────────────

def get_project_id(name: str) -> int:
    """Looks up a LabelStudio project by name and returns its numeric id."""
    resp = requests.get(f"{LS_URL}/api/projects", headers=HEADERS)   # gets the full response...{"count": 3,"results": [{ "id": 1,   "title": "AquaMind_IMG_0350"}, {"id": 2,"title": "AquaMind_IMG_0651",...}]
    resp.raise_for_status()
    for p in resp.json()['results']:
        if p['title'] == name:
            logger.info(f"project '{name}' → id={p['id']}")
            return p['id']
    raise ValueError(f"project '{name}' not found in Label Studio")


def fetch_all_tasks(project_id: int) -> list[dict]:
    """Fetches all tasks (frames) from a LabelStudio project, labeled or not."""
    url  = f"{LS_URL}/api/tasks?project={project_id}&page_size=1000"
    resp = requests.get(url, headers=HEADERS)
    resp.raise_for_status()
    return resp.json()['tasks']


def get_labeled_task_ids(project_id: int, min_id: int | None = None, max_id: int | None = None) -> list[int]:
    """Returns ids of labeled tasks only, optionally filtered to a task id range."""
    tasks = fetch_all_tasks(project_id)
    ids   = [t['id'] for t in tasks if t['is_labeled']] # For each dict t in the list — check is_labeled — if True, pull out t['id'] and add it to ids.-->    tasks is a list of dics t [{"id": 51, "is_labeled": True,  "data": {...}}, {"id": 52, "is_labeled": False, "data": {...}} - > creates a list
        
    if min_id:
        ids = [i for i in ids if i >= min_id]
    if max_id:
        ids = [i for i in ids if i <= max_id]
    logger.info(f"found {len(ids)} labeled tasks (min_id={min_id}, max_id={max_id})")
    return ids


def search_task(project_id: int, term: str) -> None:
    """Searches all tasks for a matching image filename and prints the results."""
    tasks = fetch_all_tasks(project_id)
    results = [t for t in tasks if term in t['data'].get('image', '')]
    if not results:
        print(f"no tasks found matching '{term}'")
        return
    for t in results:
        print(f"task id: {t['id']}  |  {t['data']['image']}  |  labeled: {t['is_labeled']}")


def write_sidecar(output_dir: str, project_name: str, project_id: int, task_ids: list[int], min_id: int | None, max_id: int | None) -> None:
    """Writes download metadata alongside the labels folder for provenance."""
    sidecar = {
        'project_name':  project_name,
        'project_id':    project_id,
        'num_tasks':     len(task_ids),
        'min_task_id':   min_id,
        'max_task_id':   max_id,
        'export_labels': EXPORT_LABELS,
        'export_images': EXPORT_IMAGES,
        'downloaded_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }
    path = os.path.join(output_dir, 'download_params.yaml')
    with open(path, 'w') as f:
        yaml.dump(sidecar, f, default_flow_style=False, sort_keys=False)
    logger.info(f"sidecar written → {path}")


def download_images(project_id: int, task_ids: list[int], output_dir: str) -> int:
    """Downloads source images for the given task ids into images/ subfolder."""
    images_dir = os.path.join(output_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)
    tasks    = fetch_all_tasks(project_id)
    task_map = {t['id']: t for t in tasks}
    downloaded = 0
    for task_id in task_ids:
        task      = task_map.get(task_id)
        image_url = task['data'].get('image', '') if task else ''
        if not image_url:
            continue
        if image_url.startswith('/data/'):
            image_url = LS_URL + image_url
        filename = os.path.basename(image_url.split('?')[0])
        if '-' in filename:                        # strip LabelStudio UUID prefix
            filename = filename.split('-', 1)[1]
        resp = requests.get(image_url, headers=HEADERS)
        resp.raise_for_status()
        with open(os.path.join(images_dir, filename), 'wb') as f:
            f.write(resp.content)
        downloaded += 1
    logger.info(f"downloaded {downloaded} images → {images_dir}")
    return downloaded


def download_from_labelstudio(project_id: int, task_ids: list[int], output_dir: str) -> None:
    """Downloads YOLO annotations for the given task ids, extracts the zip, and prints a summary."""
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
    print(f"  Images               : {'see below' if EXPORT_IMAGES else 'not downloaded (export_images: false)'}")
    print(f"  Output               : {output_dir}")
    print("─" * 50)


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main() -> None:
    """Runs export or search mode depending on config.yaml."""
    project_id = get_project_id(PROJECT_NAME)
    if MODE == "search":
        search_task(project_id, SEARCH_TERM)
    elif MODE == "export":
        task_ids = get_labeled_task_ids(project_id, MIN_TASK_ID, MAX_TASK_ID)
        if EXPORT_LABELS:
            download_from_labelstudio(project_id, task_ids, OUTPUT_DIR)
        if EXPORT_IMAGES:
            n = download_images(project_id, task_ids, OUTPUT_DIR)
            print(f"  Images downloaded    : {n}")
            print("─" * 50)
        write_sidecar(OUTPUT_DIR, PROJECT_NAME, project_id, task_ids, MIN_TASK_ID, MAX_TASK_ID)
    else:
        raise ValueError(f"unknown MODE '{MODE}' — use 'export' or 'search'")


# ── ENTRY POINT ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    from scripts.logger import setup_logging
    setup_logging()
    main()
