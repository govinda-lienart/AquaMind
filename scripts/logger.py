import logging

def setup_logging(level=logging.DEBUG):
    logging.basicConfig(level=level, format='%(levelname)s | %(name)s | %(funcName)s | %(message)s')
