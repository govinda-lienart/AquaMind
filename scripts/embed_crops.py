# IMPORTS
import logging
import torch
from PIL import Image
from torchvision import transforms
from scripts.console import banner, banner_sub

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# LOAD THE MODEL ("the brain")
banner("LOAD DINOv2 ON MPS")
logger.info("loading DINOv2...")
model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
model.eval()                 # evaluation mode - layers behave in stable/inference mode
device = "mps"               # running on local machine M1 GPU
model.to(device)             # move the model from CPU RAM to M1 GPU memory
logger.info(f"model loaded on {device}")

# TEST WITH ONE CROP AND FEED IT TO THE BRAIN
banner("PREPROCESS ONE CROP FROM JGP TO PIL")
crop_test_path = "output_fish_tracker/tracker_IMG_1839_basic_2026_07_23_1202/curated_crops/stretch06_fish3/frame_5151_fish_3.jpg"
img = Image.open(crop_test_path).convert("RGB")  # jpg is a compressed recipe, not a pixel grid - Image.open decodes it into a PIL picture, .convert forces 3 RGB channels
logger.info(f"raw image size (W, H): {img.size}")  # e.g. (116, 53) - PIL reports width, height only (no channel axis)
logger.info(f"mode: {img.mode}")                   # RGB = proof the 3 colour channels exist

# the preprocessing assembly line: PIL picture -> model-ready tensor
banner("PIL TO MODEL READY TENSOR - BUILDING AND RUNNING THE ASSEMBLY BELT")
    # building the assembly belt
transform = transforms.Compose([
    transforms.Resize((224, 224)), # reshape into 224 x 224 squared size that matches DINO format ingestion
    transforms.ToTensor(), # convert it to Tensor # pixels value are ranging between 0-1, # but Dino doesnt want 0-1, but wants numbers centred around zero (roughly -2 to +2)
    transforms.Normalize(mean=[0.485, 0.456, 0.406], # normalize following ImageNet statistics = average red/green/blue brightness across the millions of images DINOv2's ancestors trained on
                         std=[0.229, 0.224, 0.225]), # new_value = (old_value − mean) / std e.g pixel value 1.0: (1.0 − 0.485) / 0.229 = +2.25 - so the average pixel lands around 0.0
])
    # running the assembly belt
tensor = transform(img)
logger.info(f"tensor shape (C, H, W): {tensor.shape}")   # torch.Size([3, 224, 224]) - the 3 is the RGB
logger.info(f"tensor min: {tensor.min():.2f}  max: {tensor.max():.2f}")  # should land roughly in -2 .. +2 after Normalize
    # adding the batch axis
batch = tensor.unsqueeze(0) # adds one dimension at the beginning - indicating nunber of files in the batch..in this case 1
logger.info(f"batch shape (B,C,H,W) after unsqueeze: {batch.shape}") # torch.Size([1, 3, 224, 224])  = 150,528 numbers

banner("FOWARD PASS - RUNNING THE MODEL")
# moving and running the batch to/with my M1 GPU MPS where the model also lives - the forward pass
batch = batch.to(device) # moving to mps
with torch.no_grad(): # second guardrail similar to eval() - "don't track gradients - I'm only using, not training
    embedding = model(batch) # calling the model with the batch - forward pass - flows through 21M parameters
logger.info(f"embedding shape: {embedding.shape}") # fingerprint embedding: torch.Size([1, 384]) one image consisting of a fingerprint of 384 numbers - massive compression but capturing the essence of the fish as vector - the image embeded in a 384 dimensional space/axes


