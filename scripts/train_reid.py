
"""

Usage:
    python -m scripts.train_reid
"""
#============================================================
# IMPORTS
#============================================================
import torch
from torch.utils.data import Dataset, DataLoader
import glob
import re
import logging
from PIL import Image
from torchvision import transforms
from scripts.console import banner, banner_sub  # readable console section headers]
import torch.nn as nn

logger = logging.getLogger(__name__)  # module logger; setup_logging() configures format/level in the entry point

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
    def  __init__(self, crops_glob): # __init__ STORES a BIG list of e.g. 1949 paths and a TINY dict of 5 entries # launched once therefore init 
        self.paths = sorted(glob.glob(crops_glob)) # collect every crop path matching the pattern * and store it in a list of paths - note sorted not 100 percent needed but it helps to make sure it always stores the paths in the same way - better for debugging 
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
# MAIN
#===========================================================

def main():

    # ============================================================
    # SETUP 
    # ============================================================

    banner("FISH CROP DATASET")

    ds = FishCropDataset("output_fish_tracker/tracker_IMG_1839_basic_2026_07_23_1202/curated_crops/stretch*_fish*/*.jpg")

    banner_sub("DATASET OVERVIEW")
    logger.info(f"total crops found: {len(ds)}")
    logger.info(f"label map (fish_id -> slot): {ds.label_map}")

    banner_sub("FIRST SAMPLE")
    sample_tensor, sample_label = ds[0]      #  this triggers getitem in the class FishCropDataset -  ds[0] calls __getitem__(0) -> returns (tensor, label)
    logger.info(f"first path: {ds.paths[0]}")
    logger.info(f"label: {sample_label}")

    banner_sub("FIRST SAMPLE SHAPE")
    logger.info(f"sample tensor shape: {tuple(sample_tensor.shape)}")   # (3, 224, 224)

    banner_sub("DATALOADER — ONE BATCH")
    loader = DataLoader(ds, batch_size=32, shuffle=True)
                                                            # batch_size=32: common default — stable gradients, small enough for memory
                                                            # shuffle=True: breaks the sorted fish_1,fish_1,...,fish_2 ordering so each batch mixes fish

    # backbone — THE PRE_TRAINED LAYER - converting to finger print - output layer
    backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    backbone.eval()                       # inference mode — not training the backbone

    # head — THE EXPERT LAYER
    head = nn.Linear(384, 5) # (in-size, out-size) 384 fingerprint numbers in -> 5 fish out # here we buld the machine - the layers and staches in the head
    head.train() # set the head in training mode

    # loss + optimizer
    loss_fin = nn.CrossEntropyLoss()  # build the measuring tool - the grader # takes raw logits # softmax internally
    optimizer = torch.optim.Adam(head.parameters(), lr=1e-3) # build-once setup  # the optimizer only affects the head. lr - 0.01 is training rate #

    banner_sub("SANITY CHECK — ONE BATCH THROUGH THE PIPELINE")
    tensors, labels = next(iter(loader))     # pull ONE batch out of the loader
    with torch.no_grad():                 # no gradient graph — pure forward pass
        feats = backbone(tensors)         # push the SAME batch already pulled
    logger.info(f"fingerprints shape: {tuple(feats.shape)}")      #  (32, 384) 32 fingerprints
    logits = head(feats) # head.__call__(feats)-> then forward - inside nn.Linear - run the 32 fingerprints through the head - Running the data in the machine - in the neural network
    logger.info(f"logits shape: {tuple(logits.shape)}")   # expect (32, 5): 32 crops, 5 scores each
    logger.info(f"first row: {logits[0]}")                # 5 raw scores for crop 0 — highest = head's guess
    logger.info(f"start loss: {loss_fin(logits, labels).item()}") # compare 32 guesses against the 32 true labels # ~2.26 = untrained, random guessing

    # ============================================================
    # TRAINING LOOP
    # ============================================================

    banner_sub("TRAINING THE HEAD")
    NUM_EPOCHS = 5
    for epoch in range(NUM_EPOCHS):
        running_loss = 0.0 # is an accumulator # empty bucket at the start of each epoch
        for tensors, labels in loader: # every batch, all e.g 61 of them *e.g 1949 crops/ 32 per batch = 60.9 # so the head during one epoch gets nodged 61 one times per epoch = 305 nudges - every crop seen 5 times
            with torch.no_grad(): # backbone stays frozen - no gradients
                feats = backbone(tensors) # crops -> fingerprints
            logits = head(feats) # fingerprints -> 5 scores (THIS has gradients)
            loss = loss_fin(logits, labels) # how wrong?
            optimizer.zero_grad() # reset old gradients
            loss.backward() # backprop: which way to nudge each head weight
            optimizer.step() # optimizer: actually nudge them
            running_loss += loss.item() # add this batch's loss into the bucket # this is for reporting and for human user to have approximation info of loss
        logger.info(f"epoch {epoch+1}/{NUM_EPOCHS}  avg loss: {running_loss / len(loader):.4f}")

# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    from scripts.logger import setup_logging   # configures level (LOG_LEVEL env) + format
    setup_logging()                            # un-mutes logger.info so the output shows
    main()


