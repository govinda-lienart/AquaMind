"""
Creates a new LabelStudio project and imports crossing frames for hard-negative mining.

Usage: python -m scripts.upload_labelstudio
"""

# ── STEP 0: IMPORTS ───────────────────────────────────────────────────────────
# (we add each import only when the step that needs it appears)

import os

import requests
import yaml
from dotenv import load_dotenv

# module imports
from scripts.console import banner, banner_sub

# logging imports
import logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# ── STEP 1: READ config.yaml + AUTH ───────────────────────────────────────────

banner("STEP 1 - READ config.yaml")











# ── STEP 2: LIST frames ───────────────────────────────────────────────────────


# ── STEP 3: CREATE labelstudio project ────────────────────────────────────────


# ── STEP 4: UPLOAD images ─────────────────────────────────────────────────────


# ── STEP 5: DONE ───────────────────────────────────────────────────────────────
