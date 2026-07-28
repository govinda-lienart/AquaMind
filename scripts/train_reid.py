
"""
Usage:
    python -m scripts.train_reid
"""

# IMPORTS

from torch.utils.data import Dataset
import glob
import re
from PIL import Image

# MAIN

class FishCropDataset(Dataset):# the class inherits the DATASET structure set up by pytroch
    def  __init__(self, crops_glob): # __init__ STORES the list on the object
        self.paths = sorted(glob.glob(crops_glob)) # collect every crop path matching the pattern * and store it in a list of paths - note sorted not 100 percent needed but it helps to make sure it always stores the paths in the same way - better for debugging 
    
    def __len__(self):
        return len(self.paths) # len method - count number of paths stored in paths object

    def __getitem__(self, i):
        path = self.path[i]
        image = Image.open(path).convert("RGB") # GET IMAGE - open jpg and convert this compress3ed file into a pixed grid with RBG value (color,height and width) 
        label = int(re.search(r"fish(\d+)", path).group(1)) # GET LABEL - pull the fish_id from the filename 

# ENTRY POINT

if __name__ == "__main__":
    ds = FishCropDataset("output_fish_tracker/tracker_IMG_1839_basic_2026_07_23_1202/curated_crops/stretch*_fish*/*.jpg")
    print("total crops found:", len(ds))
    print("first path:", ds.paths[0])


