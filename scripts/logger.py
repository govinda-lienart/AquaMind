import logging
import os

def setup_logging():
    level = os.getenv('LOG_LEVEL', 'DEBUG').upper()
    logging.basicConfig(level=getattr(logging, level),
                        format='%(levelname)s | %(name)s | %(funcName)s | %(message)s')
    # silence noisy third-party DEBUG spam (font scoring / jpeg plugin imports) while keeping OUR debug logs
    for noisy in ("matplotlib", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
