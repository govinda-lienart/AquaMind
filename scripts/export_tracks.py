import pandas as pd
import db
import logging
logging.basicConfig(level=logging.INFO), format="%(message)s"
logger=logging.getLogger(__name__)
# import logging
# logging.basicConfig(level=logging.INFO, format="%(message)s")
# logger = logging.getLogger(__name__)


QUERY = """SELECT frame_number, timestamp, fish_id, x, y, occluded, confidence
FROM tracks
JOIN frames ON tracks.frame_id = frames.id
ORDER BY frame_number"""

conn = db.get_connection()
df = pd.read_sql(QUERY, conn)
print(df.head())
print(df.shape)

