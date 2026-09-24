"""
Exports YOLO annotations from a LabelStudio project.

Usage: python -m scripts.download_labelstudio
"""

# IMPORTS───────────────────────────────────────────

import os
import yaml
import requests
import zipfile
from datetime import datetime

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
logger.info(f"found {len(tasks)} total tasks in project")
labeled_ids = [t["id"] for t in tasks if t["is_labeled"]]
logger.info(f"found {len(labeled_ids)} labeled tasks")
if not labeled_ids:
    raise ValueError(f"no labeled tasks in project '{project_name}'")

# ── STEP 4: EXPORT labels (YOLO zip -> extract) ───────────────────────────────
banner("STEP 4 - EXPORT labels (YOLO zip -> extract)")

banner_sub("build output folder path")
timestamp = datetime.now().strftime("%d%m%Y_%Hh%M")
output_dir = f"labelstudio_download/{project_name}_{timestamp}"
os.makedirs(output_dir, exist_ok=True)
logger.info(f"output_dir={output_dir}")

banner_sub("build export request")
params = "exportType=YOLO&" + "&".join([f"ids[]={i}" for i in labeled_ids])
                                # "exportType=YOLO&" + prepends the export format flag => giving the final result: "exportType=YOLO&ids[]=51&ids[]=53&ids[]=54....." # exportType=YOLO_WITH_IMAGES if i want the images
                                # [f"ids[]={i}" for i in labeled_ids] => ["ids[]=51", "ids[]=53", "ids[]=54"] # ids[]= is a convention some APIs use for "this parameter can repeat multiple times" (array-style query param).
                                # "&".join([...]) glues that list together with & between each item: ids[]=51&ids[]=53&ids[]=54    
                                # => exportType=YOLO&ids[]=51&ids[]=53&ids[]=54   + willl aslo expo retrieve class id danio rerio and reflection
url = f"{LS_URL}/api/projects/{project_id}/export?{params}" 
logger.info(f"exporting {len(labeled_ids)} tasks as YOLO -> {output_dir}")

banner_sub("download export zip")
resp = requests.get(url, headers=HEADERS)
resp.raise_for_status()
zip_path = os.path.join(output_dir, "export.zip")
with open(zip_path, "wb") as f: # write bites - images
    f.write(resp.content)
logger.info(f"zip downloaded -> {zip_path}")

banner_sub("extract zip")
with zipfile.ZipFile(zip_path, "r") as z:
    z.extractall(output_dir)
os.remove(zip_path)
logger.info(f"extracted yolo labels -> {output_dir}")

# ── STEP 5: SIDECAR + DONE ─────────────────────────────────────────────────────
# lineage: download_params.yaml -> store_annotations.py -> annotation_sets (ls_* columns)
banner("STEP 5 - SIDECAR + DONE")

banner_sub("write download_params.yaml")
params = { # read by store_annotations.py
    "project_name": project_name,           
    "project_id": project_id,              
    "min_task_id": min(labeled_ids),       
    "max_task_id": max(labeled_ids),      
    "downloaded_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),  
    "labeled_tasks": len(labeled_ids),     
    "total_tasks": len(tasks),
}

with open(f"{output_dir}/download_params.yaml", "w") as f:
    yaml.safe_dump(params, f, sort_keys=False)
logger.info(f"sidecar written -> {output_dir}/download_params.yaml")

banner_sub("summary")
logger.info(f"done: {len(labeled_ids)}/{len(tasks)} labeled tasks exported to {output_dir}")
