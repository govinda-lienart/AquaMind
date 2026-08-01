"""
Shared re-ID feature extraction — crops -> DINOv2 fingerprints.

Owns the pieces every re-ID script needs, so they live in ONE place instead of being copied:
  - transform        : the PIL-image -> model-ready-tensor preprocessing belt
  - load_backbone    : the ONLY place DINOv2 is loaded (torch.hub)
  - FishCropDataset  : reads curated crop .jpgs, yields (tensor, label, frame)
  - build_features   : run the frozen backbone over chosen stretches -> feats/labels/frames/label_map

No feature cache: at this data scale one DINOv2 pass is seconds, and MLflow records each run's params.
(Fine-tuning can't cache anyway — the backbone changes every step.)

Imported by: train_reid.py (frozen head), embed_crops.py (baseline), finetune_reid.py (6.4).
"""
import torch
from torch.utils.data import Dataset, DataLoader
import glob
import re
import logging
from PIL import Image
from torchvision import transforms
from scripts.console import banner_sub

logger = logging.getLogger(__name__)

#============================================================
# PREPROCESSING BELT: PIL image - Model ready tensor
#============================================================

transform = transforms.Compose([
        transforms.Resize((224, 224)), # DINOV2 wants 224 x 224 pixels
        transforms.ToTensor(),           # PIL to tensor , pixel 0-1
        transforms.Normalize(mean=[0.485, 0.456, 0.406],     # centre on ImageNet stats - ranging between about -2 and 2
                         std=[0.229, 0.224, 0.225]),])

#============================================================
# BACKBONE LOADER (shared: one place that loads DINOv2)
#============================================================

def load_backbone(name, device=None):
    """Load a frozen DINOv2 backbone in eval mode; optionally move it to a device (e.g. 'mps')."""
    model = torch.hub.load("facebookresearch/dinov2", name)
    model.eval()                          # inference behaviour — the backbone never trains
    if device is not None:
        model.to(device)                  # e.g. 'mps' for the M1 GPU
    return model

#============================================================
# DATASET
#============================================================

class FishCropDataset(Dataset):# the class inherits the DATASET structure set up by pytroch
    def  __init__(self, crops_globs, tf=None): # __init__ STORES a BIG list of e.g. 1949 paths and a TINY dict of 5 entries # launched once therefore init
        self.tf = tf if tf is not None else transform          # custom transform (e.g. augmentation for fine-tuning); defaults to the shared eval belt
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
        frame = int(re.search(r"frame_(\d+)", path).group(1))   # pull frame number — needed for the TEMPORAL train/val split
        tensor = self.tf(image) # run the image down the pre-processing belt (or the augmented one) - (3,224,224) tensor.
        return tensor, label, frame

#============================================================
# FEATURES: run the frozen backbone -> fingerprints
#============================================================

def build_features(crops_run, stretches, backbone_name):
    """Run the frozen backbone over the selected stretches' crops.
       Returns (all_feats, all_labels, all_frames, label_map). No cache — recomputed each run."""
    banner_sub("BUILDING FINGERPRINTS (backbone pass)")
    # one glob per selected stretch; int(s):02d so '2', 2, '02' all become 'stretch02' (matches the folder names)
    crops_globs = [f"{crops_run}/curated_crops/stretch{int(s):02d}_fish*/*.jpg" for s in stretches]
    logger.info(f"stretches: {stretches}  ->  {crops_globs}")
    ds = FishCropDataset(crops_globs)
    logger.info(f"total crops found: {len(ds)}")
    logger.info(f"label map (fish_id -> slot): {ds.label_map}")
    loader = DataLoader(ds, batch_size=32, shuffle=False)   # order irrelevant (feat, label, frame travel together)

    backbone = load_backbone(backbone_name)   # frozen DINOv2, eval mode, CPU

    all_feats, all_labels, all_frames = [], [], []
    with torch.no_grad():                       # frozen — no gradients, no backward, no training of the backbone
        for tensors, labels, frames in loader:  # grab the next 32 crops
            all_feats.append(backbone(tensors)) # turn those 32 crops into 32 fingerprints
            all_labels.append(labels)           # stash their labels
            all_frames.append(frames)           # stash their frame numbers

    all_feats  = torch.cat(all_feats)           # ~61 (32,384) tensors -> one (N, 384) table
    all_labels = torch.cat(all_labels)          # ~61 (32,) tensors -> one (N,)
    all_frames = torch.cat(all_frames)          # ~61 (32,) tensors -> one (N,)
    logger.info(f"built feats: {tuple(all_feats.shape)}  labels: {tuple(all_labels.shape)}  frames: {tuple(all_frames.shape)}")
    return all_feats, all_labels, all_frames, ds.label_map
