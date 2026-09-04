"""
train_feeding_lstm_fixed_window.py — Stage 7 Part 2, fixed-window pipeline (final step).

- Loads the per-window embedding sequences (train_embeddings_*.pt / test_embeddings_*.pt) built by
  build_feeding_embeddings_fixed_window.py — each window is a (45, 384) tensor + label
- Wraps them in a Dataset / DataLoader
- Defines a small LSTM + linear classifier head (the only part that trains — the DINOv2 backbone
  was frozen and already run in the embeddings step)
- Trains with early stopping: keep the checkpoint at min test loss, not the last epoch
  (the fix that caught the 200-epoch overfitting — see diary.md, Stage 7 Part 2)
- Reports accuracy / precision / recall on the test split

usage: python -m scripts.train_feeding_lstm_fixed_window
"""
import os
import torch
import logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)
from scripts.console import banner, banner_sub
from scripts.video_utils import grab_video_name
from torch.utils.data import Dataset

VIDEO_RUN_NAME = 'IMG_2349_appearance_2026_08_12_1926'
BACKBONE_NAME = 'dinov2_vits14'
DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'

# STEP 1 — load the per-window embedding sequences (.pt)
banner("STEP 1 — load train / test embedding sequences")
parquet_path, *_ = grab_video_name(VIDEO_RUN_NAME)
emb_dir = os.path.join(os.path.dirname(parquet_path), "feeding_train_test", "embeddings")
train_path = os.path.join(emb_dir, f"train_embeddings_{BACKBONE_NAME}.pt")
test_path  = os.path.join(emb_dir, f"test_embeddings_{BACKBONE_NAME}.pt")
train_windows = torch.load(train_path, weights_only=False)  # weights_only=False -> full pickle, needed because this is my own list-of-dicts, not a flat state_dict (safe loader only handles tensors/state_dicts)
test_windows  = torch.load(test_path,  weights_only=False)
logger.info("")
logger.info(f"train: {len(train_windows)} windows | test: {len(test_windows)} windows")
logger.info(f"window keys: {list(train_windows[0].keys())}")
logger.info(f"first embeddings shape: {tuple(train_windows[0]['embeddings'].shape)}")

# STEP 2 — Dataset + DataLoader (serve one window: (45, 384) tensor + label)
banner("STEP 2 — Dataset / DataLoader")
class FeedingWindowDataset(Dataset):
    def __init__(self, windows):
      self.windows = windows
    def __len__(self):
      return len(self.windows)

train_ds = FeedingWindowDataset(train_windows)
test_ds = FeedingWindowDataset(test_windows) 
logger.info(f"train_ds: {len(train_ds)} samples | test_ds: {len(test_ds)} samples")

# STEP 3 — define the LSTM + classifier head (nn.Module)
banner("STEP 3 — model")


# STEP 4 — one training epoch + one eval pass (helper functions)
banner("STEP 4 — train / eval helpers")
# your code here


# STEP 5 — training loop with early stopping (keep min-test-loss checkpoint)
banner("STEP 5 — training loop")
# your code here


# STEP 6 — load best checkpoint, report accuracy / precision / recall on test
banner("STEP 6 — final evaluation")
