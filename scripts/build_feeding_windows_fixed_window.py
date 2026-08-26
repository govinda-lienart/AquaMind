# IMPORTS

import pandas as pd
import logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


# CONSTANTS

LINK_POS_LABELS = 'output_fish_tracker/feeding_labels.xlsx'

# MAIN

pos_labels = pd.read_excel(LINK_POS_LABELS)
logger.info(pos_labels.head().to_string())

# drop the junk Unnamed columns
pos_labels = pos_labels.drop(columns=["Unnamed: 5","Unnamed: 6"])
logger.info(pos_labels.head().to_string())

# force the two frame columns start and end to numeric, bad values -> NaN
pos_labels ["framenumber_start"] = pd.to_numeric(pos_labels["framenumber_start", errors="coerce"))
logger.info(pos_labels.head().to_string())

# drop rows missing any of the three required columns

# drop the swapped-typo row: end must be >= start
