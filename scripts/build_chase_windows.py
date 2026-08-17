"""usage:  python -m scripts.build_chase_windows
hardcoded path to curated labels chasing"""

# IMPORTS

import pandas as pd
import logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# MAIN

# converting excel to labels dataframe
labels_xls_path = 'output_fish_tracker/chase_labels.xlsx'
labels = pd.read_excel(labels_xls_path)
logger.info(labels.head().to_string())


