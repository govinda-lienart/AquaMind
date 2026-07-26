# IMPORTS
import logging
import torch
import torch.nn.functional as F  # cosine_similarity lives here
from PIL import Image
from torchvision import transforms
from scripts.console import banner, banner_sub
import glob
import re

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
    """Embed every crop matching the glob, return a list of (fish_id, embedding)."""
    banner("GATHER EMBEDDINGS")
    records = []
    for path in sorted(glob.glob(stretch_glob)): #glob finds files whose names match a pattern, and hands back a list of their paths.
        fish_id = int(re.search(r"fish_(\d+)", path).group(1))  # /d+ takes one or more digit # r is raw string # group 1 captures what is between parenthisis # int() converts string to int
        emb = embed(path) # runs DINOv2 forward pass
        records.append((fish_id, emb)) # embedding the tuple
    logger.info(f"gathered {len(records)} embeddings")
    return records


def compare_pairs(records):
    "compares every pair of embedding and return a dataframe of (fish i, fish j) is same cosine"
    rows = []
    for i in range(len(record)):

# MAIN

def main():

    banner("SPOT-CHECK: DO SAME-FISH FINGERPRINTS LAND CLOSER?")

    # TESTING WITH A FEW CROPS

    # 3 test crops - a & b are the SAME fish (fish 2, stretch02), c is a DIFFERENT fish (fish 5, stretch02)
    emb_a = embed("output_fish_tracker/tracker_IMG_1839_basic_2026_07_23_1202/curated_crops/stretch02_fish2/frame_1851_fish_2.jpg")  # anchor - fish 2, early frame
    emb_b = embed("output_fish_tracker/tracker_IMG_1839_basic_2026_07_23_1202/curated_crops/stretch02_fish2/frame_2660_fish_2.jpg")  # same fish 2, late frame - should be CLOSE to a
    emb_c = embed("output_fish_tracker/tracker_IMG_1839_basic_2026_07_23_1202/curated_crops/stretch02_fish5/frame_1943_fish_5.jpg")  # different fish - should be FAR from a

    # compare fingerprints by cosine similarity (1.0 = identical direction, higher = more similar) - Cosine similarity measures the ANGLE between two arrows: 
        # small angle → cosine ≈ 1.0 → SAME direction → similar fish
        # big angle → cosine ≈ 0 → different directions → different fish
    sim_same = F.cosine_similarity(emb_a, emb_b)   # comparing the same fish from same stretch but frames far apart
    sim_diff = F.cosine_similarity(emb_a, emb_c)   # different fish - same stretch

    banner_sub("RESULT")
    logger.info(f"cosine(a, b) SAME fish:      {sim_same.item():.4f}") # cosine(a, b) SAME fish:      0.7849
    logger.info(f"cosine(a, c) DIFFERENT fish: {sim_diff.item():.4f}") # cosine(a, c) DIFFERENT fish: 0.7490
    # almost no difference

    # LOOPING OVER ALL THE CROPS AND COMPARE STATS
    records = gather_embeddings("output_fish_tracker/tracker_IMG_1839_basic_2026_07_23_1202/curated_crops/stretch02_fish*/*.jpg")
    results = compare_pairs(records)


# ENTRY POINT
if __name__ == '__main__':
    main()
