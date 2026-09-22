"""
Creates a new LabelStudio project and imports crossing frames for hard-negative mining.

Usage: python -m scripts.upload_labelstudio
"""

# IMPORTS

import requests
import yaml
from dotenv import load_dotenv
import os

# module imports
from scripts.console import banner, banner_sub

# logging imports
import logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# ── STEP 1: READ config.yaml + AUTH ───────────────────────────────────────────

banner("STEP 1 - READ config.yaml")

load_dotenv()
LS_URL = os.getenv("LABEL_STUDIO_URL") # http://localhost:8080
LS_TOKEN = os.getenv("LABEL_STUDIO_API_KEY") # stored in env
if not LS_TOKEN:
    raise RuntimeError("LABEL_STUDIO_API_KEY not found - need to check .env file ")

banner_sub("load upload_labelstudio section of config.yaml")
with open("config.yaml") as f:
    cfg = yaml.safe_load(f)
cfg = cfg["upload_labelstudio"]
logger.info(cfg) # e.g {'project_name': 'sample_test_IMG_0867', 'frames_dir': 'frames/frames_IMG_0867_20260921_1357'}

banner_sub("upload_labelstudio specific parameters")
project_name = cfg["project_name"]
frames_dir = cfg["frames_dir"]
logger.info(f"project_name={project_name}, frames_dir={frames_dir}")

banner_sub("exchange refresh token for access token")
resp = requests.post(f"{LS_URL}/api/token/refresh", json={"refresh": LS_TOKEN})
logger.info(f"token refresh status: {resp.status_code}") # token refresh status: 200 => OK, success
resp.raise_for_status() #  it checks the status code, and if it's in the "bad" range (4xx or 5xx), it raises an exceptiosn, but if good like 200 is ok an dwil lpass
access_token = resp.json()["access"] # parses the json respobse of end point as a dictionary...and we pull out the access token using ["access"]
logger.info(f"access_token starts with: {access_token[:15]}...") # for saftery reason just pritning part of it 
HEADERS = {"Authorization": f"Bearer {access_token}"} # will be used in next steps when requesting using the access token.


# ── STEP 2: LIST frames ───────────────────────────────────────────────────────


# ── STEP 3: CREATE labelstudio project ────────────────────────────────────────


# ── STEP 4: UPLOAD images ─────────────────────────────────────────────────────


# ── STEP 5: DONE ───────────────────────────────────────────────────────────────
