"""
Creates a new LabelStudio project and imports crossing frames for hard-negative mining.

Usage: python -m scripts.upload_labelstudio
"""

# IMPORTS───────────────────────────────────────────

import requests
import yaml
from dotenv import load_dotenv
import os
import glob

# module imports
from scripts.console import banner, banner_sub

# logging imports
import logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

#CONSTANTS───────────────────────────────────────────

# this will be send to labelstudio t's just pre-defining what the new project's labeling interface (defauflt values for the proejct established in the way the fish will be labeled) e.g pink frame for fish refelction
LABEL_CONFIG = """
<View>
  <Image name="image" value="$image" zoom="true"/>
  <RectangleLabels name="label" toName="image" strokeWidth="1">
    <Label value="danio_rerio" background="#00a3d7" hotkey="d"/>
    <Label value="reflection" background="#d357fe" hotkey="r"/>
  </RectangleLabels>
</View>
"""

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

# ── STEP 2: LIST frames ─────────────────────────────────────────────────────── # find every image file that needs uploading.
banner("STEP 2 - LIST frames")

pattern = os.path.join(frames_dir, "*.jpg") # os.path.join("frames/frames_IMG_0867_20260921_1357", "*.jpg") # => "frames/frames_IMG_0867_20260921_1357/*.jpg" => finds all files that finsihes with jpg in that very folder
matched_files = glob.glob(pattern) # acts on pattern - looks for the files and creates a list e,.g ['frames/frames_IMG_0867_20260921_1357/frame_0001.jpg', 'frames/frames_IMG_0867_20260921_1357/fr...
images = sorted(matched_files) # lexicographically (alphabetical/dictionary order) list
logger.info(f"found {len(images)} images")
if not images:
    raise RuntimeError(f"no files found in {frames_dir}")

# ── STEP 3: CREATE labelstudio project ────────────────────────────────────────
banner("STEP 3 - CREATE labelstudio project")

resp = requests.post(
    f"{LS_URL}/api/projects",
    headers=HEADERS,
    json={"title": project_name, "label_config": LABEL_CONFIG},
)
resp.raise_for_status()
logger.info(resp.json())

project_id = resp.json()["id"]
logger.info(f"created project '{project_name}' -> id={project_id}")

# ── STEP 4 - UPLOAD images ────────────────────────────────────────
banner("STEP 4 - UPLOAD images")

total = 0
for i, img in enumerate(images, 1): # starts enumerating at 1    # enumating is for the progress log otherwise use =>  for img in images:
    with open(img,"rb") as f: # 'rb' = read + binar for images
        resp = requests.post(
            f"{LS_URL}/api/projects/{project_id}/import",
            headers=HEADERS,
            files=[('file', (os.path.basename(img), f))], # different HTTP encoding than json # requests picks the right transport encoding based on which parameter you use (json= vs files= vs data=),
                                                         #  files= expects a list of file entries
                                                         #  os.path.basename(img)  e.g frame_0001.jpg
                                                         #  f is the acutla content - the actual image
        )
    resp.raise_for_status()
    total += 1
    logger.info(f"{i}/{len(images)} uploaded")
    
# ── STEP 5 - DONE ────────────────────────────────────────

banner("STEP 5 - DONE")

logger.info(f"done: {total} images uploaded into '{project_name}' (id={project_id})")
logger.info(f"open: {LS_URL}/projects/{project_id}/")
