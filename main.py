from src.logger import logging

from src.exception import CustomException

import sys

from src.event_detection_ap import score

if __name__=='__main__':
    try:
        a = 1/0
    except Exception as e:
        logging.info("Divided by 0")
        raise CustomException(e,sys)