
# IMPORTS
from logging import basicConfig
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

# CONFIGS
CONFIG_PATH = 'config.yaml' 
with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)['analyse_behaviour']
logger.info(f'configuration data: {cfg}')

