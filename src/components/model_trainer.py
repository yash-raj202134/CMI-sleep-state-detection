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
import matplotlib.pyplot as plt

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


logging.info("model training begains")
# Model training
try:

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


