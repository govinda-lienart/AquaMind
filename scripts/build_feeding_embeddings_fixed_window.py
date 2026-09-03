"""
- Loads the crop-sequence manifests (train_crops.parquet / test_crops.parquet) built by build_feeding_crops_fixed_window.py
- For each window (grouped by event_id): loads its 45 crops in frame order, runs them through the FROZEN DINOv2 backbone as one batch -> a (45, embedding_dim) tensor
- Saves the per-window embedding sequences (.pt, not parquet) so the LSTM step can train on them without re-running the backbone every epoch (the backbone never changes -> compute once here)

usage: python -m scripts.build_feeding_embeddings_fixed_window
"""
import os
import torch
import pandas as pd
from PIL import Image
import logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)
from scripts.console import banner, banner_sub
from scripts.video_utils import grab_video_name
from scripts.reid_features import transform, load_backbone

VIDEO_RUN_NAME = 'IMG_2349_appearance_2026_08_12_1926'
BACKBONE_NAME = 'dinov2_vits14'

# STEP 1 — load the crop-sequence manifests (train_crops / test_crops)
banner("STEP 1 — load train_crops / test_crops")
parquet_path, pixels_per_cm, calibration_secs, surface_y_px, bottom_y_px, frame_number_end = grab_video_name(VIDEO_RUN_NAME) 
output_folder = os.path.dirname(parquet_path)
print(output_folder)


# STEP 2 — load the frozen backbone once (eval mode, no grad — it never trains)
# anner(f"STEP 2 — load frozen backbone ({BACKBONE_NAME} on {DEVICE})")


# STEP 3 — embed ONE window: load its 45 crops in frame order, run through backbone as one batch
banner("STEP 3 — embed one window (flat version)")


# STEP 3b — wrap it into a function


# STEP 4 — loop every window (grouped by event_id), embed its sequence, keep label + fish_id



# STEP 5 — save the per-window embedding sequences as .pt
banner("STEP 5 — save embeddings")
