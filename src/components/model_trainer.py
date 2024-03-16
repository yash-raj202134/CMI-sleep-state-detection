import sys
sys.path.append('')

from src.components.model_generator import MyGenerator
from src.exception import CustomException
from src.logger import logging
from src.components.data_loader import WINDOW_SIZE , STEP_SIZE
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense ,Dropout ,LSTM ,Activation
from keras.optimizers import Adam


n_epoch = 300

batch_size = 8
steps_per_epoch = 50
try:
    train_events = pd.read_csv("data/train_events.csv")
    series_ids = train_events['series_id'].unique()

except Exception as e:
    logging.info("Erro reading train_event file")
    raise CustomException(e,sys)



my_generator = MyGenerator(series_ids, batch_size, steps_per_epoch)

model = Sequential() 
model.add(LSTM(128,return_sequences=True, input_shape=( WINDOW_SIZE, 2)))
model.add(Dropout(0.2))
model.add(LSTM(units=32))
model.add(Dropout(0.2))
model.add(Dense(2, activation='softmax')) 

model.compile(
    loss='binary_crossentropy', 
    optimizer=Adam(), 
    metrics = ['accuracy']
)
print(model.summary())
