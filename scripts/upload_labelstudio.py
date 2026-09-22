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


# ── STEP 2: LIST frames ───────────────────────────────────────────────────────


# ── STEP 3: CREATE labelstudio project ────────────────────────────────────────


# ── STEP 4: UPLOAD images ─────────────────────────────────────────────────────


# ── STEP 5: DONE ───────────────────────────────────────────────────────────────
