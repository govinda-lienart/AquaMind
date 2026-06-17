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

CONFIG_PATH = 'config.yaml'
PROGRESS_EVERY = 20

LABEL_CONFIG = """
<View>
  <Image name="image" value="$image" zoom="true"/>
  <RectangleLabels name="label" toName="image" strokeWidth="1">
    <Label value="danio_rerio" background="#00a3d7" hotkey="d"/>
    <Label value="reflection" background="#d357fe" hotkey="r"/>
  </RectangleLabels>
  <KeyPointLabels name="keypoint" toName="image" opacity="0.9" strokeWidth="3">
    <Label value="eye" background="#ff0000" hotkey="e" model_index="0"/>
  </KeyPointLabels>
</View>
"""

logger = logging.getLogger(__name__)


# ── CONFIG + AUTH ──────────────────────────────────────────────────────────────

def load_config():
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    c  = cfg['import_labelstudio']
    ts = datetime.now().strftime('%Y%m%d_%Hh%M')
    frames_dir = c.get('frames_dir') or _latest_crossing_dir()
    return {
        'project_name': f"{c['project_name']}_{ts}",
        'frames_dir':   frames_dir,
    }


def _latest_crossing_dir():
    base = Path('frames')
    dirs = sorted(base.glob('crossing_frames_*'), key=os.path.getmtime, reverse=True)
    if not dirs:
        raise FileNotFoundError("No crossing_frames_* folder found. Run make extract-crossings first.")
    return str(dirs[0])


def get_access_token(url, api_key):
    resp = requests.post(f"{url}/api/token/refresh", json={"refresh": api_key})
    resp.raise_for_status()
    return resp.json()["access"]


# ── LABELSTUDIO API ───────────────────────────────────────────────────────────

def create_project(url, headers, name):
    resp = requests.post(
        f"{url}/api/projects",
        headers=headers,
        json={"title": name, "label_config": LABEL_CONFIG},
    )
    resp.raise_for_status()
    project_id = resp.json()['id']
    logger.info(f"Created project '{name}' → id={project_id}")
    return project_id


def upload_images(url, headers, project_id, images):
    total = 0
    for i, img in enumerate(images, 1):
        with open(img, 'rb') as f:
            resp = requests.post(
                f"{url}/api/projects/{project_id}/import",
                headers=headers,
                files=[('file', (Path(img).name, f, 'image/jpeg'))],
            )
        resp.raise_for_status()
        total += 1
        if i % PROGRESS_EVERY == 0 or i == len(images):
            print(f"  Uploaded {i}/{len(images)} images...")
    return total


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    load_dotenv(Path(__file__).parent.parent / ".env")
    ls_url   = os.getenv("LABEL_STUDIO_URL")
    ls_token = os.getenv("LABEL_STUDIO_API_KEY")
    if not ls_token:
        raise RuntimeError("LABEL_STUDIO_API_KEY not found — check your .env file")

    p       = load_config()
    headers = {"Authorization": f"Bearer {get_access_token(ls_url, ls_token)}"}

    images = sorted(Path(p['frames_dir']).glob('*.jpg'))
    if not images:
        print(f"  No .jpg files found in {p['frames_dir']}")
        return

    print("=" * 50)
    print(f"  Project:  {p['project_name']}")
    print(f"  Frames:   {p['frames_dir']} ({len(images)} images)")
    print("=" * 50)

    project_id = create_project(ls_url, headers, p['project_name'])
    total = upload_images(ls_url, headers, project_id, images)
    print(f"\nDone. {total} tasks imported into '{p['project_name']}'")
    print(f"Open: {ls_url}/projects/{project_id}/")


# ── ENTRY POINT ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    from scripts.logger import setup_logging
    setup_logging()
    main()
