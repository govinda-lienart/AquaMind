"""
Triggers the AquaMind training kernel on Kaggle.

Input  : kernel_ref from config.yaml
Needs  : kaggle authenticated (run: kaggle auth login)
         dataset pushed to DVC + git (run: make push-dataset_dvc first)
Output : new kernel run triggered on Kaggle
"""

# ── IMPORTS ───────────────────────────────────────────────────────────────────

import json
import logging
import os
import subprocess
import tempfile
import time

import yaml


# ── CONSTANTS ─────────────────────────────────────────────────────────────────

CONFIG_PATH = 'config.yaml'

logger = logging.getLogger(__name__)


# ── HELPERS ───────────────────────────────────────────────────────────────────

def load_config():
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    return {
        'kernel_ref': cfg['push_to_kaggle']['kernel_ref'],
    }


def trigger_kernel(kernel_ref):
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(
            ['kaggle', 'kernels', 'pull', kernel_ref, '-p', tmp, '--metadata'],
            check=True
        )
        meta_path = os.path.join(tmp, 'kernel-metadata.json')
        with open(meta_path) as f:
            meta = json.load(f)
        meta['enable_internet'] = True
        meta['enable_gpu'] = True
        creds = 'govindalienart/aquamind-credentials'
        if creds not in meta.get('dataset_sources', []):
            meta.setdefault('dataset_sources', []).append(creds)
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
        subprocess.run(
            ['kaggle', 'kernels', 'push', '-p', tmp],
            check=True
        )
    logger.info(f"kernel triggered → {kernel_ref}")

def watch_kernel(kernel_ref):
    print("Watching kernel progress...")
    while True:
        result = subprocess.run(
            ['kaggle', 'kernels', 'status', kernel_ref],
            capture_output=True, text=True
        )
        status = result.stdout.strip()
        print(f"{status}")
        if 'complete' in status.lower() or 'error' in status.lower():
            break
        time.sleep(60)

# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    cfg = load_config()
    trigger_kernel(cfg['kernel_ref'])

    print("\n" + "─" * 50)
    print("  KAGGLE TRIGGER SUMMARY")
    print("─" * 50)
    print(f"  Kernel   : {cfg['kernel_ref']}")
    print(f"  Monitor  : https://www.kaggle.com/{cfg['kernel_ref']}")
    print("─" * 50)

    watch_kernel(cfg['kernel_ref'])

# ── ENTRY POINT ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    from scripts.logger import setup_logging
    setup_logging()
    main()
