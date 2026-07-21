# IMPORT

import os
import yaml
import pandas as pd
import numpy as np
import argparse

import matplotlib.pyplot as plt
plt.use('Agg') # no pop up windows when producing results

from scripts.console import banner, banner_sub  # improves layout when printing in the console
import logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger=logging.getLogger(__name__)

# pairwise distance between each fish pair, per fram


#-------------------
# ENTRY POINT/GUARD
#-------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Imports Parquet data")
    parser