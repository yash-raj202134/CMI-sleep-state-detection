import sys
sys.path.append('')

from src.components.data_loader import load_data
from src.components.data_loader import WINDOW_SIZE , STEP_SIZE

import numpy as np
from keras.utils import Sequence
import random


class MyGenerator(Sequence):
    def __init__(self, series_ids, batch_size, steps_per_epoch):
        self.series_ids = series_ids
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch
        
        self.num_X_used = 0
        self.num_called = 0

        self.X = np.empty(0)
        self.y = np.empty(0)

    def __len__(self):
        return self.steps_per_epoch
    
    def _load_data(self, series_id):
        X, y, _ = load_data(series_id, WINDOW_SIZE, STEP_SIZE)
        
        self.X = X
        self.y = y
        self.num_X_used = 0
        
    def __getitem__(self, idx):
        if (self.num_X_used > int(len(self.X)/self.batch_size)) or (len(self.X)==0):
            series_id = random.choice(self.series_ids)
            self._load_data(series_id)
            
        start_id = self.num_X_used*self.batch_size
        end_id = (self.num_X_used*self.batch_size+self.batch_size)

        batch_x = self.X[start_id:end_id,:,:]
        batch_y = self.y[start_id:end_id,:]
        
        self.num_called += 1
        self.num_X_used += 1
            
        if self.num_called==(self.steps_per_epoch-1):
            self.num_called = 0 
        
        return batch_x, batch_y