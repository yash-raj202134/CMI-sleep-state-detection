import sys
sys.path.append('')

from src.components.model_generator import MyGenerator
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object , rmse_and_plot
from src.components.data_loader import WINDOW_SIZE , STEP_SIZE
from src.components.data_loader import load_data
from src.components.events_generator import get_events
from src.event_detection_ap import score


import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense ,Dropout ,LSTM ,Activation
from keras.optimizers import Adam
import matplotlib.pyplot as plt


#parameters
n_epoch = 300

batch_size = 8
steps_per_epoch = 50
try:
    train_events = pd.read_csv("data/train_events.csv")
    series_ids = train_events['series_id'].unique()

except Exception as e:
    logging.info("Erro reading train_event file")
    raise CustomException(e,sys)


logging.info("model building")

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



# Model training
try:
    logging.info("model training begains")
    hist = model.fit_generator(
        my_generator,
        epochs=n_epoch,
    #     validation_data=(X_valid, y_valid),
        )


    plt.plot(hist.history['loss'],label="train set")
    # plt.plot(hist.history['val_loss'],label="test set")
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.show()

except Exception as e:
    raise CustomException(e,sys)

# save the model after training
save_object(model,"model")


try:
    # prediction
    series_id = series_ids[0]
    X, y, data_info = load_data(series_id, WINDOW_SIZE, STEP_SIZE)

    pred_y = model.predict(X) 

except Exception as e :

    raise CustomException(e,sys)


y_true = y[:,0]
y_pred = pred_y[:,0]
timestamp = data_info['timestamp']

rms = rmse_and_plot(y_pred,y_true,timestamp)


preds = 1-np.argmax(pred_y, axis=1)
probs = np.max(pred_y, axis=1)

try:
    # processing predictions and probabilities to identify events and their occurrences in a dataset

    predict_events = get_events(preds, probs, data_info)
    predict_events.to_csv("result/result.csv")

except Exception as e:
    raise CustomException(e,sys)




# Tolerence

tolerances = {
    'onset': [12, 36, 60, 90, 120, 150, 180, 240, 300, 360], 
    'wakeup': [12, 36, 60, 90, 120, 150, 180, 240, 300, 360]
}

series_id_column_name = "series_id"
time_column_name = "step"
event_column_name = "event"
score_column_name = "score"
use_scoring_intervals = None

sc = score(
        train_events,
        predict_events,
        tolerances,
        series_id_column_name,
        time_column_name,
        event_column_name,
        score_column_name,
        use_scoring_intervals
)

print(f"Score : {sc}")