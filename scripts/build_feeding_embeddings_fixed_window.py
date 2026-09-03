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
DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'

# STEP 1 — load the crop-sequence manifests (train_crops / test_crops) from parquet and convert to a dataframe
banner("STEP 1 — load train_crops / test_crops")
parquet_path, *_ = grab_video_name(VIDEO_RUN_NAME)
crop_sequences = os.path.join(os.path.dirname(parquet_path), "feeding_train_test", "crop_sequences")
train_crops_path = os.path.join(crop_sequences, "train_crops.parquet")
test_crops_path = os.path.join(crop_sequences, "test_crops.parquet")
train_crops = pd.read_parquet(train_crops_path)
test_crops = pd.read_parquet(test_crops_path)
logger.info(f"train_crops: {train_crops.shape}, test_crops: {test_crops.shape}")
logger.info(train_crops.head().to_string())

# STEP 2 — load the frozen backbone once (eval mode, no grad — it never trains)
banner(f"STEP 2 — load frozen backbone ({BACKBONE_NAME} on {DEVICE})")
backbone = load_backbone(BACKBONE_NAME, device = DEVICE)

# STEP 3 — embeding one window as test load its 45 crops in frame order, run through backbone as one batch
banner("STEP 3 — embed one window (flat version)")

sample_event = train_crops["event_id"].iloc[0]
window_rows = train_crops[train_crops["event_id"] == sample_event]
window_rows = window_rows.sort_values("frame_position") 
crop_paths = window_rows["crop_path"].to_list() # throw away the pandas wrapper - converts it into a list
images = [transform(Image.open(p).convert("RGB")) for p in crop_paths]
            # comprehension list - creates a list of tensors (transform) - and forced it to split into RGB channels = (3, 244, 244)

logger.info(f"images: {len(images)} x {tuple(images[0].shape)}")

# STEP 3b — wrap it into a function





images = [transform(Image.open(p).convert("RGB")) for p in crop_paths]



# STEP 4 — loop every window (grouped by event_id), embed its sequence, keep label + fish_id



# STEP 5 — save the per-window embedding sequences as .pt
banner("STEP 5 — save embeddings")
