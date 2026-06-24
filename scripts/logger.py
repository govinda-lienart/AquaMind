import logging
import os

def setup_logging():
    level = os.getenv('LOG_LEVEL', 'DEBUG').upper()
    logging.basicConfig(level=getattr(logging, level),
                        format='%(levelname)s | %(name)s | %(funcName)s | %(message)s')
