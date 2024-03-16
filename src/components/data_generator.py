import sys
sys.path.append('')

import pandas as pd
import pyarrow.parquet as pq


import os

from src.exception import CustomException
from src.logger import logging


os.chdir("data")


try:
    logging.info("Reading the train_series and train_event files")
    train_series = pq.read_table("train_series.parquet").to_pandas()
    train_events = pd.read_csv("train_events.csv")
    print(train_events)

    series_ids = train_events['series_id'].unique()

except Exception as e:
    raise CustomException(e,sys)


new_train_series_list = {}
for series_id in series_ids:
    print('\n'+'='*50)
    print(series_id)

    _train_series = train_series[train_series['series_id']==series_id]
    _train_events = train_events[train_events['series_id']==series_id]
    
    _train_series.index = pd.to_datetime(_train_series['timestamp'], utc=True)
    _train_events.index = pd.to_datetime(_train_events['timestamp'], utc=True)
    
    _train_series = _train_series.assign(awake=-1, event_id=-1, night=0)
    
    current_event_time = _train_series.index[0]
    for _event_id, _events in enumerate(_train_events.itertuples()):
        _event = 0 if _events.event=='onset' else 1
        if _event_id == (len(_train_events)-1):
            _event = -1
        
        if not pd.isnull(_events.step):
            _train_series.loc[_train_series.index>=_events.Index, ['awake']] = _event
            current_event_time = _events.Index
        else:
            _train_series.loc[_train_series.index>=current_event_time, ['awake']] = -1
        _train_series.loc[_train_series.index>=_events.Index, ['event_id']] = _event_id
        _train_series.loc[_train_series.index>=_events.Index, ['night']] = _events.night
        
    new_train_series_list[series_id] = _train_series.copy()
    
logging.info("intermediate data generated")

logging.info("saving the intermediate data")

try:

    for series_id, _train_series in new_train_series_list.items():
        _train_series.to_csv("data/intermediate_data/train_series_{}.csv".format(series_id), index=False)
    logging.info(" intermediate data saved")

except Exception as e:
    raise CustomException(e,sys)
