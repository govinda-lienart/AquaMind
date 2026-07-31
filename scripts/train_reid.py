
"""
Stage 6 — per-video re-ID head (DINOv2 backbone + trained classifier head).

Cache-or-compute:
  - config.yaml `train_reid.videos.<name>.features_path` SET   -> LOAD that cache, skip the slow backbone
  - features_path BLANK                                        -> BUILD a fresh timestamped cache, then train

Usage:
    python -m scripts.train_reid --video_name IMG_1839
"""
#============================================================
# IMPORTS
#============================================================
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset # TensorDataset lets us batch tensors directly (feats already cached, no preprocessing)
import os
import glob
import re
import logging
import yaml
import argparse
import subprocess
import datetime
from PIL import Image
from torchvision import transforms
from scripts.console import banner, banner_sub  # readable console section headers
import torch.nn as nn

logger = logging.getLogger(__name__)  # module logger; setup_logging() configures format/level in the entry point

CONFIG_PATH = 'config.yaml'
with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)['train_reid']   # this script's config section

#============================================================
# PREPROCESSING BELT: PIL image - Model ready tensor
#============================================================

transform = transforms.Compose([
        transforms.Resize((224, 224)), # DINOV2 wants 224 x 224 pixels
        transforms.ToTensor(),           # PIL to tensor , pixel 0-1
        transforms.Normalize(mean=[0.485, 0.456, 0.406],     # centre on ImageNet stats - ranging between about -2 and 2
                         std=[0.229, 0.224, 0.225]),])

#============================================================
# DATASET
#============================================================

class FishCropDataset(Dataset):# the class inherits the DATASET structure set up by pytroch
    def  __init__(self, crops_globs): # __init__ STORES a BIG list of e.g. 1949 paths and a TINY dict of 5 entries # launched once therefore init
        if isinstance(crops_globs, str):                       # accept a single pattern OR a list of patterns (one per stretch)
            crops_globs = [crops_globs]
        self.paths = sorted(p for g in crops_globs for p in glob.glob(g)) # union of every pattern, sorted so path order is stable (better for debugging)
        raw_ids = {int(re.search(r"fish_(\d+)", p).group(1)) for p in self.paths}  # goes through each path and pulls the fish_id out for each path - here we call it raw cause this is exactly how it appears in the original, unprocessed. # note ythay also that {} make it a set {... for ...} (no colons ;) so duplicated vanishes cause fish_1 appears hudneres of time but the set only keeps 1
        self.label_map = {raw_id: i for i, raw_id in enumerate(sorted(raw_ids))}   # 1) sorted changes the set into a list [1,2,3,4,5] - a set has no order, so we sorted to guarantee order. 2) enumerate pairs each item with its position (0,1), (1,2),  (2, 3), (3, 4), (4, 5) - 3) raw id converts it all in a dict comprehension {1:0, 2:1, 3:2, 4:3, 5:4}.
                                                                                    # the converstion from just fish_id 1-5 to 0-4 has to do with the classifiers output
                                                                                    # head output:   [ 2.1 ,  0.3 , -1.0 ,  4.7 ,  0.8 ]
                                                                                    # position:         0      1      2      3      4
                                                                                    # this is to feed CrossEntropyLoss
    def __len__(self):
        return len(self.paths) # len method - count number of paths stored in paths object

    def __getitem__(self, i): # pairing of label and fish_id # launched for evry crop/epoch thousands of time therefore not in init
        path = self.paths[i]
        image = Image.open(path).convert("RGB") # GET IMAGE - open jpg and convert this compress3ed file into a pixed grid with RBG value (color,height and width)
        raw_id = int(re.search(r"fish_(\d+)", path).group(1))   # pull fish id from filename
        label = self.label_map[raw_id]                          #  for example read fish 5 off this crop's filename — asks table map, what slot number is that? → slot 4. That's my label."
        tensor = transform(image) # run the image down the pre-processing belt - (3,254,244) normalized tensor.
        return tensor, label

#============================================================
# CONFIG LOADER
#============================================================

def grab_video_name(video_name):
    "pull this video's re-ID params from config.yaml"
    video_cfg     = cfg['videos'][video_name]
    crops_run     = video_cfg['crops_run']
    stretches     = video_cfg['stretches']           # which curated stretch(es) to train on (identity-safety)
    backbone_name = video_cfg['backbone']
    features_path = video_cfg.get('features_path')   # optional — None/blank means BUILD a fresh cache
    num_epochs    = video_cfg['num_epochs']
    lr            = video_cfg['lr']
    batch_size    = video_cfg['batch_size']
    banner('LOADING CONFIGURATION')
    logger.info(f"loaded cfg: {video_cfg}")
    return crops_run, stretches, backbone_name, features_path, num_epochs, lr, batch_size

#============================================================
# FEATURES: cache-or-compute
#============================================================

def build_or_load_features(crops_run, stretches, backbone_name, features_path):
    """Return (all_feats, all_labels, label_map).
       FAST path: features_path exists -> load it, backbone never runs.
       SLOW path: otherwise run the backbone over the SELECTED stretches ONCE, save a timestamped cache + sidecar."""

    # ---------- FAST PATH: load a prebuilt cache ----------
    if features_path and os.path.exists(features_path):
        banner_sub("LOADING CACHED FINGERPRINTS")
        data = torch.load(features_path)
        logger.info(f"loaded cache <- {features_path}  feats: {tuple(data['feats'].shape)}")
        return data["feats"], data["labels"], data["label_map"]

    # ---------- SLOW PATH: build the cache from the selected stretches ----------
    banner_sub("BUILDING FINGERPRINTS (one-time backbone pass)")
    # one glob per selected stretch; int(s):02d so '2', 2, '02' all become 'stretch02' (matches the folder names)
    crops_globs = [f"{crops_run}/curated_crops/stretch{int(s):02d}_fish*/*.jpg" for s in stretches]
    logger.info(f"training stretches: {stretches}  ->  {crops_globs}")
    ds = FishCropDataset(crops_globs)
    logger.info(f"total crops found: {len(ds)}")
    logger.info(f"label map (fish_id -> slot): {ds.label_map}")
    loader = DataLoader(ds, batch_size=32, shuffle=False)   # order irrelevant for caching (feat & label travel together)

    backbone = torch.hub.load("facebookresearch/dinov2", backbone_name)
    backbone.eval() # switches layers from training behaviour to inference behaviour

    all_feats = []      # will collect fingerprint batches: [ (32,384), (32,384), ..., (29,384) ]
    all_labels = []     # will collect the matching labels
    with torch.no_grad():                       # frozen — no gradients, no backward, no training of the DINOv2 backbone
        for tensors, labels in loader:          # grab the next 32 crops
            all_feats.append(backbone(tensors)) # turn those 32 crops into 32 fingerprints, stash them
            all_labels.append(labels)           # stash their labels in a second bucket

    all_feats = torch.cat(all_feats)            # cat glues the ~61 (32,384) tensors into one (1949, 384) table
    all_labels = torch.cat(all_labels)          # ~61 (32,) tensors -> one (1949,)
    logger.info(f"built feats: {tuple(all_feats.shape)}  labels: {tuple(all_labels.shape)}")

    # versioned subfolder so a new cache never overwrites an old one (same idea as tracker_basic's timestamped run folder)
    stamp    = datetime.datetime.now().strftime("%Y_%m_%d_%H%M")
    reid_dir = f"{crops_run}/reid_cache_{stamp}"
    os.makedirs(reid_dir, exist_ok=True)

    feat_path = f"{reid_dir}/reid_features.pt"
    torch.save({"feats": all_feats, "labels": all_labels, "label_map": ds.label_map}, feat_path)  # label_map travels with the feats
    logger.info(f"saved fingerprints -> {feat_path}")

    # sidecar (provenance: how this cache was created — mirrors tracker_basic.py's config sidecar)
    stretches_used = sorted({re.search(r"stretch(\d+)", p).group(1) for p in ds.paths})  # which stretches actually contributed
    sidecar = {
        "tracker_run": os.path.basename(crops_run),     # which tracker run produced these crops
        "crops_globs": crops_globs,                     # the exact source-crop patterns (no drift — same list the dataset used)
        "stretches":   stretches_used,                  # the actual stretch ids harvested (e.g. ['02'])
        "backbone":    backbone_name,                   # fingerprints from a different backbone are NOT comparable
        "n_crops":     all_feats.shape[0],              # rows = number of crops cached
        "feat_dim":    all_feats.shape[1],              # cols = fingerprint length (384)
        "label_map":   ds.label_map,                    # fish_id -> slot, so slot 0 always means the same fish
        "git_commit":  subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip(),
        "created_at":  datetime.datetime.now().isoformat(timespec="seconds"),
    }
    with open(f"{reid_dir}/reid_features.yaml", "w") as f:
        yaml.dump(sidecar, f, sort_keys=False)          # same yaml.dump pattern as tracker_basic.py
    logger.info(f"saved sidecar -> {reid_dir}/reid_features.yaml")
    logger.info(f"REUSE: set  features_path: {feat_path}  in config.yaml to skip the backbone next time")

    return all_feats, all_labels, ds.label_map

#============================================================
# MAIN
#============================================================

def main(crops_run, stretches, backbone_name, features_path, num_epochs, lr, batch_size):

    banner("FISH RE-ID — TRAIN HEAD")

    # ---------- features: load a cache or build one ----------
    all_feats, all_labels, label_map = build_or_load_features(crops_run, stretches, backbone_name, features_path)

    n_classes = len(label_map)          # number of fish (NOT hardcoded — comes from the data)
    feat_dim  = all_feats.shape[1]      # 384

    # ---------- head + loss + optimizer (build once) ----------
    head = nn.Linear(feat_dim, n_classes)   # (in-size, out-size) fingerprint -> one score per fish
    head.train()                            # head in training mode
    loss_fin = nn.CrossEntropyLoss()        # the grader: takes raw logits, does softmax internally
    optimizer = torch.optim.Adam(head.parameters(), lr=lr)   # nudges ONLY the head

    # ---------- training loop over cached fingerprints ----------
    banner_sub("TRAINING THE HEAD")
    feat_ds = TensorDataset(all_feats, all_labels)                       # wrap the cached tensors as a dataset (no files, no backbone)
    feat_loader = DataLoader(feat_ds, batch_size=batch_size, shuffle=True)   # batches of (fingerprint, label)

    for epoch in range(num_epochs):
        running_loss = 0.0 # accumulator — reset each epoch (reporting only)
        for feats, labels in feat_loader: # batches of CACHED fingerprints — no crops, no backbone -> instant
            logits = head(feats) # fingerprints -> N scores (THIS has gradients)
            loss = loss_fin(logits, labels) # how wrong?
            optimizer.zero_grad() # reset old gradients
            loss.backward() # backprop: which way to nudge each head weight
            optimizer.step() # optimizer: actually nudge them
            running_loss += loss.item() # add this batch's loss into the bucket
        logger.info(f"epoch {epoch+1}/{num_epochs}  avg loss: {running_loss / len(feat_loader):.4f}")

#============================================================
# ENTRY POINT
#============================================================

if __name__ == "__main__":
    from scripts.logger import setup_logging   # configures level (LOG_LEVEL env) + format
    setup_logging()                            # un-mutes logger.info so the output shows

    parser = argparse.ArgumentParser(description="Train per-video re-ID head")
    parser.add_argument("--video_name", default=cfg['video'], help="key under train_reid.videos in config.yaml")
    args = parser.parse_args()
    main(*grab_video_name(args.video_name))    # unpack the config tuple straight into main()
