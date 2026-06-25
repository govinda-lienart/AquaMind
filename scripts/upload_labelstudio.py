"""
Creates a new LabelStudio project and imports crossing frames for hard-negative mining.
Auto-detects the latest crossing_frames folder unless frames_dir is set in config.

Needs: LABEL_STUDIO_URL and LABEL_STUDIO_API_KEY in .env
"""

# ── IMPORTS ───────────────────────────────────────────────────────────────────

import logging
import os
from datetime import datetime
from pathlib import Path

import requests
import yaml
from dotenv import load_dotenv

CONFIG_PATH = 'config.yaml' #  project_name · frames_dir

LABEL_CONFIG = """
<View>
  <Image name="image" value="$image" zoom="true"/>
  <RectangleLabels name="label" toName="image" strokeWidth="1">
    <Label value="danio_rerio" background="#00a3d7" hotkey="d"/>
    <Label value="reflection" background="#d357fe" hotkey="r"/>
  </RectangleLabels>
</View>
"""
# configation set up is send to labelstudio to already predefine the 2 labels used for tagging - reflection and dario.

logger = logging.getLogger(__name__)

load_dotenv()

LS_URL   = os.getenv("LABEL_STUDIO_URL")
LS_TOKEN = os.getenv("LABEL_STUDIO_API_KEY")

if not LS_TOKEN:
    raise RuntimeError("LABEL_STUDIO_API_KEY not found — check your .env file")


# ── CONFIG + AUTH ──────────────────────────────────────────────────────────────

def load_config():
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    c  = cfg['import_labelstudio']
    ts = datetime.now().strftime('%Y%m%d_%Hh%M')
    frames_dir = c.get('frames_dir')
    if not frames_dir:
        raise ValueError("frames_dir must be set in config.yaml under import_labelstudio")
    return {
        'project_name': f"{c['project_name']}_{ts}",
        'frames_dir':   frames_dir,
    }


def get_access_token() -> str:
    """Exchanges the long-lived refresh token for a short-lived access token."""
    resp = requests.post(f"{LS_URL}/api/token/refresh", json={"refresh": LS_TOKEN})
    resp.raise_for_status()
    return resp.json()["access"]


HEADERS = {"Authorization": f"Bearer {get_access_token()}"}


# ── LABELSTUDIO API ───────────────────────────────────────────────────────────

def create_project(name: str) -> int:
    """Creates a new LabelStudio project and returns its id."""
    resp = requests.post(
        f"{LS_URL}/api/projects",
        headers=HEADERS,
        json={"title": name, "label_config": LABEL_CONFIG},
    )
    resp.raise_for_status()
    project_id = resp.json()['id']
    logger.info(f"Created project '{name}' → id={project_id}")
    return project_id


def upload_images(project_id: int, images: list) -> int:
    """Uploads JPG frames to a LabelStudio project one by one. Returns total uploaded."""
    total = 0
    for i, img in enumerate(images, 1):
        with open(img, 'rb') as f:
            resp = requests.post(
                f"{LS_URL}/api/projects/{project_id}/import",
                headers=HEADERS,
                files=[('file', (Path(img).name, f, 'image/jpeg'))], # send image via HTTP POST ->   files (convention) = The structure ('file' (convention), (filename -> e.g frame_0060_IMG_0350.jpg, file_object f -> the actual bites , content_type chared with server)) is what LabelStudio's API expects specificall
            )
        resp.raise_for_status()
        total += 1
        if i % PROGRESS_EVERY == 0 or i == len(images):
            print(f"  Uploaded {i}/{len(images)} images...")
    return total


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main() -> None:
    """Creates a LabelStudio project and uploads frames from the configured folder."""
    p = load_config()  # project_name - frames_dir

    images = sorted(Path(p['frames_dir']).glob('*.jpg')) # contains a list of path objects [ Path('frames/crossing_frames_IMG_0350_20260624_1641/frame_0060_IMG_0350.jpg'), Path('frames/crossing_frames_IMG_0350_20260624_1641/frame_0120_IMG_0350.jpg'),....]
    if not images:
        print(f"  No .jpg files found in {p['frames_dir']}")
        return

    print("=" * 50)
    print(f"  Project:  {p['project_name']}")
    print(f"  Frames:   {p['frames_dir']} ({len(images)} images)")
    print("=" * 50)

    project_id = create_project(p['project_name'])
    total = upload_images(project_id, images)
    print(f"\nDone. {total} tasks imported into '{p['project_name']}'")
    print(f"Open: {LS_URL}/projects/{project_id}/")


# ── ENTRY POINT ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    from scripts.logger import setup_logging
    setup_logging()
    main()
