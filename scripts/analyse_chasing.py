
#---------------
# IMPORTS
#---------------

import os
import yaml
import pandas as pd
import numpy as np
import argparse

import matplotlib
matplotlib.use('Agg') # avoids popup windows of poduced plots
import matplotlib.pyplot as plt 
from scripts.console import banner, banner_sub
import logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger=logging.getLogger(__name__)

#---------------
# CONFIGS
#---------------

# loading tank related data
with open('config.yaml') as f:
    cfg = yaml.safe_load(f)['analyse_behaviour']
config_2349 = grab_video_name('IMG_2349_new')
logger.info(f'configuration data: {cfg}')

#---------------
# HELPER FUNCTIONS 
#---------------



#---------------
# ENTRY POINT
#---------------

if __name__ == "__main__"