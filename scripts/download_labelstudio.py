"""
Exports YOLO annotations from a LabelStudio project.

Usage: python -m scripts.download_labelstudio
"""

# IMPORTS───────────────────────────────────────────

import os
import yaml
import requests

# module imports
from scripts.console import banner, banner_sub
from dotenv import load_dotenv

# logging imports
import logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# CONSTANTS───────────────────────────────────────────


# ── STEP 1: READ config.yaml + AUTH ───────────────────────────────────────────

banner("STEP 1 - READ config.yaml")

load_dotenv()
LS_URL = os.getenv("LABEL_STUDIO_URL") # http://localhost:8080
LS_TOKEN = os.getenv("LABEL_STUDIO_API_KEY") # stored in env
if not LS_TOKEN:
    raise RuntimeError("LABEL_STUDIO_API_KEY not found - need to check .env file ")

banner_sub("load download_labelstudio section of config.yaml")
with open("config.yaml") as f:
    cfg = yaml.safe_load(f)
cfg = cfg["download_labelstudio"]
logger.info(cfg) # e.g {'project_name': 'ghost_frames_hard_mining_IMG_1839_20260804_14h09'}

banner_sub("download_labelstudio specific parameters")
project_name = cfg["project_name"]
logger.info(f"project_name={project_name}")

banner_sub("exchange refresh token for access token")
resp = requests.post(f"{LS_URL}/api/token/refresh", json={"refresh": LS_TOKEN})
logger.info(f"token refresh status: {resp.status_code}")
resp.raise_for_status()
access_token = resp.json()["access"]
logger.info(f"access_token starts with: {access_token[:15]}...")
HEADERS = {"Authorization": f"Bearer {access_token}"}

# ── STEP 2: LOOKUP project by name ────────────────────────────────────────────

banner("STEP 2 - LOOKUP project id by name")

resp = requests.get(f"{LS_URL}/api/projects", headers=HEADERS)
resp.raise_for_status()
projects = resp.json()["results"]  #  "results" is a key in the JSON dict LabelStudio sends back for list-type endpoints (like /api/projects) EG. result ->  {'count': 12, 'next': None, 'previous': None, 'results': [{'id': 1, 'title': 'AquaMind_IMG_0350', ...}, {'id': 2, 'title': 'sampl
logger.info(f"found {len(projects)} project(s) in LabelStudio")

project_id = None
for p in projects:
    logger.info(f"checking: {p['title']} (id={p['id']})")
    if p["title"] == project_name:
        project_id = p["id"]
        break

if project_id is None:
    raise ValueError(f"project '{project_name}' not found in LabelStudio")

logger.info(f"project '{project_name}' -> id={project_id}")

# ── STEP 3: FETCH tasks + filter to labeled ───────────────────────────────────
banner("STEP 3 -  FETCH tasks + filter to labeled")
url = f"{LS_URL}/api/tasks?project={project_id}&page_size=1000" # if had lets say 10k samples then i would need to loop over it
resp = requests.get(url, headers=HEADERS)
resp.raise_for_status()
tasks = resp.json()["tasks"] # his /api/tasks call gives you lightweight metadata per image: its id, whether it's labeled (is_labeled), some info about the image itself. It's not carrying the full box coordinates/class for each annotation 
                             # {"total": 10, "tasks": [{"id": 51, "is_labeled": False, "data": {...}}, {"id": 52, "is_labeled": True, "data": {...}}, ...]}




# ── STEP 4: EXPORT labels (YOLO zip -> extract) ───────────────────────────────


# ── STEP 5: SIDECAR + DONE ─────────────────────────────────────────────────────
