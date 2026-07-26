# IMPORTS
import logging
import torch
from PIL import Image
from torchvision import transforms
from scripts.console import banner, banner_sub

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

# MAIN

def main():

    banner("SPOT-CHECK: DO SAME-FISH FINGERPRINTS LAND CLOSER?")
    load_model()

    banner("EMBEDDING")
    embed(crop_path):
        # 3 test crops path confifs
        emb_a = embed("output_fish_tracker/tracker_IMG_1839_basic_2026_07_23_1202/curated_crops/stretch02_fish2/frame_1851_fish_2.jpg")  # anchor - fish 2, early frame
        emb_b = embed("output_fish_tracker/tracker_IMG_1839_basic_2026_07_23_1202/curated_crops/stretch02_fish2/frame_2660_fish_2.jpg")  # same fish 2, late frame - should be CLOSE to a
        emb_c = embed("ooutput_fish_tracker/tracker_IMG_1839_basic_2026_07_23_1202/curated_crops/stretch02_fish5/frame_1943_fish_5.jpg")  # different fish
        logger.info(f"emb_a shape: {emb_a.shape}")
        logger.info(f"emb_b shape: {emb_b.shape}")
        logger.info(f"emb_c shape: {emb_c.shape}")



# ENTRY POINT
if __name__ == '__main__':
    main()
