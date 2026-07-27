# IMPORTS
import logging
import torch
import torch.nn.functional as F  # cosine_similarity lives here
from PIL import Image
from torchvision import transforms
from scripts.console import banner, banner_sub
import glob
import re
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# LOAD THE MODEL ("the brain") - once, at import, reused by every embed() call
banner("LOAD DINOv2 ON MPS")
logger.info("loading DINOv2...")
model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
model.eval()                 # evaluation mode - layers behave in stable/inference mode
device = "mps"               # running on local machine M1 GPU
model.to(device)             # move the model from CPU RAM to M1 GPU memory
logger.info(f"model loaded on {device}")

# BUILD THE PREPROCESSING BELT - once, reused by every embed() call
transform = transforms.Compose([
    transforms.Resize((224, 224)), # reshape into 224 x 224 squared size that matches DINO format ingestion
    transforms.ToTensor(), # convert it to Tensor # pixels value are ranging between 0-1, # but Dino doesnt want 0-1, but wants numbers centred around zero (roughly -2 to +2)
    transforms.Normalize(mean=[0.485, 0.456, 0.406], # normalize following ImageNet statistics = average red/green/blue brightness across the millions of images DINOv2's ancestors trained on
                         std=[0.229, 0.224, 0.225]), # new_value = (old_value − mean) / std e.g pixel value 1.0: (1.0 − 0.485) / 0.229 = +2.25 - so the average pixel lands around 0.0
])

# HELPER FUNCTION

def embed(crop_path):
    """Take a crop path -> return its 384-number DINOv2 fingerprint."""
    # decode the JPEG → PIL picture
    img = Image.open(crop_path).convert("RGB")  # jpg is a compressed recipe, not a pixel grid - Image.open decodes it into a PIL picture, .convert forces 3 RGB channels

    # run the crop down the preprocessing belt -> (3,224,224) model-ready tensor
    tensor = transform(img)

    # add batch axis + move to GPU -> (1,3,224,224)
    batch = tensor.unsqueeze(0).to(device)  # unsqueeze adds the batch dim at the front, .to moves it to mps

    # forward pass through DINOv2 -> (1,384) fingerprint
    with torch.no_grad(): # second guardrail similar to eval() - "don't track gradients - I'm only using, not training
        embedding = model(batch) # calling the model with the batch - forward pass - flows through 21M parameters
    return embedding  # the 384-number fingerprint - the image embedded in a 384-dimensional space

def gather_embeddings(stretch_glob):
    """Embed every crop matching the glob, return a list of tuples: (fish_id, frame,  emb)."""
    banner_sub("GATHER EMBEDDINGS")
    records = []
    for path in sorted(glob.glob(stretch_glob)): #glob finds files whose names match a pattern, and hands back a list of their paths.
        fish_id = int(re.search(r"fish_(\d+)", path).group(1))  # frame_6457_fish_5.jpg -->  /d+ takes one or more digit # r is raw string # group 1 captures what is between parenthisis # int() converts string to int
        frame = int(re.search(r"frame_(\d+)", path).group(1))
        emb = embed(path) # runs DINOv2 forward pass
        records.append((fish_id, frame,  emb))  # 3 value tuple
    logger.info(f"gathered {len(records)} embeddings")
    return records
    # e.g records = [
    # (2, tensor([[ 0.124, -0.873,  1.402,  0.031, -0.556,  ... ]])),   # fish 2, 384 numbers
    # (2, tensor([[ 0.201, -0.790,  1.388,  0.045, -0.601,  ... ]])),   # fish 2


def compare_pairs(records):
    """Compare every pair of embeddings, return a DataFrame of (fish_i, fish_j, is_same, cosine)."""
    banner_sub("COMPARE ALL PAIRS")
    rows = []
    for i in range(len(records)): # records is a list of tuples - loop over its indexes
        for j in range(i+1, len(records)): # j starts AFTER i - each pair once, no self-pairs
            fish_i, frame, emb_i,  = records[i]
            fish_j, frame,  emb_j = records[j]
            cos = F.cosine_similarity(emb_i, emb_j).item()      # item pulls the single python number out of the 1-element tensor([0.7849]) -> 0.7849
            rows.append({"fish_i": fish_i, "fish_j": fish_j,
                         "is_same": fish_i == fish_j, "cosine": cos})
    results = pd.DataFrame(rows)   # built ONCE, after both loops finish
    logger.info(results.head().to_string())
    logger.info(f"compared {len(results)} pairs")
    return results


def rank1_accuracy(records):
    """Split records into early-gallery / late-query, then score rank-1 identification accuracy."""
    logger.info(f"before sort (fish_id, frame): {[(r[0], r[1]) for r in records[:3]]}")
    records_sorted = sorted(records, key=lambda r: r[1]) # without key it would standard sort on first element fish_id
    logger.info(f"after sort  (fish_id, frame): {[(r[0], r[1]) for r in records_sorted[:3]]}")
        # before:
        # [(5, 1943, emb), (2, 1851, emb), (2, 2660, emb)]
        # after  (earliest frame -> latest):
        # [(2, 1851, emb), (5, 1943, emb), (2, 2660, emb)]
    






# MAIN

def main():

    banner("DINOv2 ZERO-TRAINING RE-ID BASELINE")

    # EXPERIMENT 1 — spot-check on 3 hand-picked crops
    #   a & b = SAME fish (fish 2, stretch02); c = DIFFERENT fish (fish 5, stretch02)
    #   cosine similarity = angle between two fingerprint vectors:
    #    small angle -> cosine near 1.0 -> similar   -   big angle -> cosine near 0 -> different
    banner("EXPERIMENT 1: 3-CROP SPOT-CHECK")
    emb_a = embed("output_fish_tracker/tracker_IMG_1839_basic_2026_07_23_1202/curated_crops/stretch02_fish2/frame_1851_fish_2.jpg")   # anchor  - fish 2, early frame
    emb_b = embed("output_fish_tracker/tracker_IMG_1839_basic_2026_07_23_1202/curated_crops/stretch02_fish2/frame_2660_fish_2.jpg")   # SAME    - fish 2, late frame  -> should be CLOSE to a
    emb_c = embed("output_fish_tracker/tracker_IMG_1839_basic_2026_07_23_1202/curated_crops/stretch02_fish5/frame_1943_fish_5.jpg")   # DIFF    - fish 5             -> should be FAR from a
    sim_same = F.cosine_similarity(emb_a, emb_b)
    sim_diff = F.cosine_similarity(emb_a, emb_c)
    logger.info(f"cosine(a, b) SAME fish:      {sim_same.item():.4f}")
    logger.info(f"cosine(a, c) DIFFERENT fish: {sim_diff.item():.4f}")
    # -> barely any gap: weak signal even before we scale up

    # EXPERIMENT 2 — same/different averaged over EVERY pair in the stretch
    banner("EXPERIMENT 2: MEAN COSINE, SAME vs DIFFERENT FISH")
    records = gather_embeddings("output_fish_tracker/tracker_IMG_1839_basic_2026_07_23_1202/curated_crops/stretch02_fish*/*.jpg")
    results = compare_pairs(records)
    logger.info(results.groupby("is_same")["cosine"].mean().to_string())
    # -> False 0.625 / True 0.684 : confirms the weak signal at scale

    # EXPERIMENT 3 — rank-1 identification (enroll early crops = gallery, test late crops = query)
    banner("EXPERIMENT 3: RANK-1 IDENTIFICATION ACCURACY")
    acc = rank1_accuracy(records)

# # ENTRY POINT
if __name__ == '__main__':
    main()


