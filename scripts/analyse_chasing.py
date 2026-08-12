from logging import basicConfig
import os
import yaml
import pandas as pd
import numpy as np
import argparse

import matplotlib
matplotlib.use('Agg') # avoids popup windows of poduced plots
import pyplot as pllt 
from script.console import banner, banner_sub
logging.basicConfig
logger = logging.getLogger(__name__)

CONFIG_PATH = 'config.yaml' 


