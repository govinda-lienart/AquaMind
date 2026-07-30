
"""
Usage:
    python -m scripts.train_reid
"""

# IMPORTS

from torch.utils.data import Dataset, DataLoader
import glob
import re
from PIL import Image
from torchvision import transforms
from scripts.console import banner, banner_sub  # readable console section headers

# PREPROCESSING BELT: PIL image - Model ready tensor

transform = transforms.Compose([
        transforms.Resize((224, 224)), # DINOV2 wants 224 x 224 pixels
        transforms.ToTensor(),           # PIL to tensor , pixel 0-1
        transforms.Normalize(mean=[0.485, 0.456, 0.406],     # centre on ImageNet stats - ranging between about -2 and 2
                         std=[0.229, 0.224, 0.225]),])

# HELP FUNCTIONS

def describe_tensor(t):
    """Print a tensor's key facts"""
    print(f"  shape:   {tuple(t.shape)}")          # e.g. (3, 224, 224)
    print(f"  dtype:   {t.dtype}")                 # e.g. torch.float32
    print(f"  device:  {t.device}")               # cpu / mps
    print(f"  min/max: {t.min():.3f} / {t.max():.3f}")   # value range after Normalize (~ -2 to +2)
    print(f"  mean:    {t.mean():.3f}")            # roughly 0 if normalized well
    print(f"  peek:    {t[0, :2, :3]}")            # tiny corner: channel 0, first 2 rows, first 3 cols
    print(f"  grad?:   {t.requires_grad}")   # False — not part of a graph yet
    print(f"  grad:    {t.grad}")             # None — no gradient computed

# MAIN

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
    
# MAIN 

def main():
    banner("FISH CROP DATASET")

    ds = FishCropDataset("output_fish_tracker/tracker_IMG_1839_basic_2026_07_23_1202/curated_crops/stretch*_fish*/*.jpg")

    banner_sub("DATASET OVERVIEW")
    print("total crops found:", len(ds))
    print("label map (fish_id -> slot):", ds.label_map)

    banner_sub("FIRST SAMPLE")
    sample_tensor, sample_label = ds[0]      # ds[0] calls __getitem__(0) -> returns (tensor, label)
    print("first path:", ds.paths[0])
    print("label:", sample_label)

    banner_sub("TENSOR INSPECTION OF FIRST SAMPLE")
    describe_tensor(sample_tensor)           # hand the tensor to your helper

    banner_sub("DATALOADER — ONE BATCH")
    loader = DataLoader(ds, batch_size=32, shuffle=True)
    # batch_size=32: common default — stable gradients, small enough for memory
    # shuffle=True: breaks the sorted fish_1,fish_1,...,fish_2 ordering so each batch mixes fish

    tensors, labels = next(iter(loader))     # pull ONE batch
    print("batch tensors shape:", tensors.shape)   # predicted (32, 3, 224, 224)?
    print("batch labels shape: ", labels.shape)
    print("labels in batch:    ", labels)          # a mix of slots 0-4 if shuffle worked

# ENTRY POINT

if __name__ == "__main__":
    main()


