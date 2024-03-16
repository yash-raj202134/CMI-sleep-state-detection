import sys
sys.path.append('')


import os
from src.logger import logging 
from src.exception import CustomException

import pandas as pd
import numpy as np

import random

try:
    train_events = pd.read_csv("data/train_events.csv")

    series_ids = train_events['series_id'].unique()

except Exception as e:
    logging.info("Erro reading train_event file")
    raise CustomException(e,sys)



WINDOW_SIZE = int(2*60*60/5)
STEP_SIZE = int(20*60/5)


def load_data(series_id, window_size, step_size):
    """Method to load the data with series id window size and step size"""

    features = ['awake', 'timestamp', 'anglez', 'enmo', 'step', 'series_id']
    logging.info("reading intermediate data")

    _train_series = pd.read_csv("data/intermediate_data/train_series_{}.csv".format(series_id))
    logging.info("preparing the training data")

    _data = []
    for _start_step in range(0, len(_train_series)-window_size, step_size):
        _data.append(_train_series[features].iloc[_start_step:(_start_step+window_size)].values)

    _data = np.stack(_data)
    train_data = _data[np.all(_data[:,:,0]!=-1, axis=1),:,:]

    data_info = pd.DataFrame(
        train_data[:, int(window_size/2), np.isin(features, ['series_id', 'timestamp', 'step'])],
        columns = ['timestamp', 'step', 'series_id']
    )
    data_info['timestamp'] = pd.to_datetime(data_info['timestamp'], utc=True).dt.tz_convert('America/Los_Angeles')

    X = train_data[:, :, np.isin(features, ['anglez', 'enmo'])].astype(np.float32)

    y = train_data[:, int(window_size/2), np.isin(features, ['awake'])].astype(np.int32)
    y = np.concatenate([y, 1-y], 1)
    logging.info("data loaded and prepared sucessfully")

    
    return X, y, data_info


    
