import os
import sys
sys.path.append('')

import numpy as np 
import pandas as pd
import dill
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

from src.exception import CustomException
from src.logger import logging

def save_object(obj, file_path):
    """
    Save the trained model in .h5 format.
    Args:
    - model: Keras model object to be saved.
    - filepath: String. Path to save the model.
    """
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        obj.save(file_path)
        logging.info(f"Model saved successfully as {file_path}")

    except Exception as e:
        raise CustomException(e, sys)
    
    
def rmse_and_plot(pred_y, true_y, timestamp):
    """
    Calculate RMSE between predicted and true values, and visualize the results.

    Args:
    - pred_y: Numpy array. Predicted values.
    - true_y: Numpy array. True values.
    - timestamp: Numpy array or Pandas Series. Timestamps for visualization.

    Returns:
    - rmse_score: Float. Root Mean Squared Error.
    """
    # Calculate RMSE
    rmse_score = np.sqrt(((true_y - pred_y) ** 2).mean())
    print("RMSE Score:", rmse_score)

    # Create DataFrame for visualization
    pred_y_df = pd.DataFrame(pred_y)
    pred_y_df.index = timestamp

    true_y_df = pd.DataFrame(true_y)
    true_y_df.index = timestamp

    # Plot predicted and true values
    fig, ax = plt.subplots(figsize=(20, 3))
    ax.plot(pred_y_df, color='red', label="Predicted")
    ax.plot(true_y_df, color='blue', label="True")
    ax.legend()
    plt.show()

    return rmse_score



def load_object(file_path):
    try:
        with open(file_path ,'rb') as file_obj:
            return pickle.load(file_obj)
        
    except Exception as e:
        raise CustomException(e,sys)
    

