"""usage:  python -m scripts.train_chase_lstm
Phase F - LSTM on the same labelled chase events as train_chase_classifier.py (Phase E),
but fed raw per-frame sequences instead of hand-summarized mean/min/max stats."""


# IMPORTS

import pandas as pd
import logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

from scripts.console import banner, banner_sub

# CONSTANTS

LABELS_XLS_PATH = 'output_fish_tracker/chase_labels.xlsx'

# MAIN

# STEP 1 - load the human-labeled chase/non-chase events (same source Phase D used)
banner('STEP 1 - LOAD chase_labels.xlsx')
labels = pd.read_excel(LABELS_XLS_PATH)
logger.info(f'{labels.shape[0]} labeled events ({(labels["label"]==1).sum()} positive, {(labels["label"]==0).sum()} negative)')
