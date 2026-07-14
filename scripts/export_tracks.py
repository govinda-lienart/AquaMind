import pandas as pd
from scripts.db import get_connection
import logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger=logging.getLogger(__name__)

QUERY = """SELECT frame_number, timestamp, fish_id, x, y, confidence, occluded
FROM tracks
ORDER BY frame_number"""

conn = get_connection()
df = pd.read_sql(QUERY, conn)
logger.info(df.head().to_string())
logger.info(df.shape)


