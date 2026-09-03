"""
- Loads the crop-sequence manifests (train_crops.parquet / test_crops.parquet) built by build_feeding_crops_fixed_window.py
- For each window (grouped by event_id): loads its 45 crops in frame order, transform(tensor) runs them through the FROZEN DINOv2 backbone as one batch -> a (45, embedding_dim) tensor
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
banner(f"STEP 2 — lad frozen backbone ({BACKBONE_NAME} on {DEVICE})")
backbone = load_backbone(BACKBONE_NAME, device = DEVICE)

# STEP 3 a — flat - embeding one window as test load its 45 crops in frame order, run through backbone as one batch
banner("STEP 3 — embed one window (flat version)")
sample_event = train_crops["event_id"].iloc[0]
window_rows = train_crops[train_crops["event_id"] == sample_event]
window_rows = window_rows.sort_values("frame_position") 
crop_paths = window_rows["crop_path"].to_list() # throw away the pandas wrapper - converts it into a list
images = [transform(Image.open(p).convert("RGB")) for p in crop_paths]            # [] collects into a list -  list of tensors (transform) - and forced it to split into RGB channels = (3, 244, 244)
logger.info(f"images: {len(images)} x {tuple(images[0].shape)}") # tuple(...) just makes it print as (3, 224, 224) insrtad of   torch.Size([3, 224, 224]).
batch = torch.stack(images).to(DEVICE) #  batching - list of 45 - bundling into one tensor  #(45, 3, 224, 224)  # DINOv2's forward expects a 4-D tensor
with torch.no_grad(): # "with "" flips a switch -on- and guarantees it flips -off////no need to build computaitnal graph during forward (waste) cause the backbone is frozen and is not trained
  embeddings = backbone(batch) # this is the resulting embedding after passing throuht the neural network
embeddings = embeddings.cpu() #  copies the tensor out of GPU memory into normal RAM and hands you a CPU tensor.
logger.info(f'embeddings: {tuple(embeddings.shape)}')

# STEP 3 b - wrapping the flat version into a function - one window
@torch.no_grad()
def embed_window(event_id, crops_df): # crops_df --- takes in train_df or test_df
  """function takes in train_crops or test_crops and convert one single event into an tensor embedding (45, 3, 244, 244) """
  window_rows = crops_df[crops_df["event_id"] == event_id]
  window_rows = window_rows.sort_values("frame_position") 
  crop_paths = window_rows["crop_path"].to_list() 
  images = [transform(Image.open(p).convert("RGB")) for p in crop_paths]          
  batch = torch.stack(images).to(DEVICE) 
  # with torch.no_grad():  # is replaced by a decorator. Not the case here...but if my function has several branching returns (if) ..it affects all returns 
  embeddings = backbone(batch) 
  return embeddings.cpu()

# STEP 4 — loop every window (grouped by event_id), embed its sequence, keep label + fish_id
# banner("STEP 4 — loop every window ")
# train_window = [] 
#               # -=-?list of per-window dicts: event_id, label, fish_id, embeddings
#               # -> embeddings = model input, label = training target
#               # -> event_id / fish_id = bookkeeping for reporting + debugging (never enter the model)
# for event_id in train_crops["event_id"].unique(): # unique? train_crops is the exploded manifest —?  45 rows per window so  the event_id column looks like: 2, 2, 2, 2, ... (45 times) ..., 2, 5, 5, 
#     rows = train_crops[train_crops["event_id"] == event_id]
#     emb = embed_window(event_id, train_crops)
#     train_window.append({
#         "event_id": event_id, # a name tag so you I trace a window back 
#         "label": int(rows["label"].iloc[0]),
#         "fish_id": int(rows["fish_id"].iloc[0]),
#         "embeddings": emb,
#     })
#     logger.info(f"train: {len(train_window)} windows, emb {tuple(train_window[0]['embeddings'].shape)}")

# STEP 4 - wrap up in a function # looping over each event of both dataframes

def build_embeddings(crops_df, split_name):
  """function uses single event embedding function and loops it over each event for either crops_train or crops_test"""
  windows = [] 
                # -=-?list of per-window dicts: event_id, label, fish_id, embeddings
                # -> embeddings = model input, label = training target
                # -> event_id / fish_id = bookkeeping for reporting + debugging (never enter the model)
  for event_id in crops_df["event_id"].unique(): # unique? train_crops is the exploded manifest —?  45 rows per window so  the event_id column looks like: 2, 2, 2, 2, ... (45 times) ..., 2, 5, 5, 
    rows = crops_df[crops_df["event_id"] == event_id]
    emb = embed_window(event_id, crops_df)
    windows.append({
        "event_id": event_id, # a name tag so you I trace a window back 
        "label": int(rows["label"].iloc[0]),
        "fish_id": int(rows["fish_id"].iloc[0]),
        "embeddings": emb,
    })
  logger.info(f"{split_name}: {len(windows)} windows, emb {tuple(windows[0]['embeddings'].shape)}")
  return windows
train_windows = build_embeddings(train_crops, "train")
test_windows  = build_embeddings(test_crops, "test")


# STEP 5 — save the per-window embedding sequences as .pt
banner("STEP 5 — save embeddings")
emb_dir = os.path.join(os.path.dirname(parquet_path), "feeding_train_test", "embeddings")
os.makedirs(emb_dir, exist_ok=True)
train_path = os.path.join(emb_dir, f"train_embeddings_{BACKBONE_NAME}.pt")
test_path  = os.path.join(emb_dir, f"test_embeddings_{BACKBONE_NAME}.pt")

torch.save(train_windows, train_path)
torch.save(test_windows, test_path)

logger.info(f"saved -> {train_path}")
logger.info(f"saved -> {test_path}")