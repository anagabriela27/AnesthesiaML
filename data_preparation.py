import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

class DataProcessor:
    def __init__(self, dataframe, vital_signs, time_window_before, time_window_after=1, test_size=0.2, random_state=42):
        self.dataframe = dataframe
        self.vital_signs = vital_signs
        self.time_window_before = time_window_before
        self.time_window_after = time_window_after
        self.test_size = test_size
        self.random_state = random_state
        self.scalers = {}

def series_to_supervised(self):
    """
    Note: this function is an adapted version of the function from 
    https://github.com/damiandziedzic/ML-Project---Anesthesia-Prediction/blob/master/Anesthesia_Prediction_ML_Project.ipynb
    
    Frame a time series as a supervised learning dataset.
    Arguments:
        df_case: DataFrame of one case
        vital_signs: list, the list of vital signs
        time_window_before: int, the number of lag observations as input (X, in seconds)
        time_window_after: int, the number of observations as output (y, in seconds, default is 1)
        how many seconds after the current time point to predict (default is 1 <=> y=t+1)
    Returns:
        df_all: Pandas DataFrame, the supervised learning dataset
    """
    n_vars = len(self.vital_signs)
    df = self.dataframe[self.vital_signs]
    cols, names = list(), list()
    
    # input sequence (t-n, ... t-1)
    for i in range(self.time_window_before, 0, -1):
        cols.append(df.shift(i))
        names += [f'{self.vital_signs[j]}(t-{i})' for j in range(n_vars)]
    
    # forecast sequence (t, t+1, ... t+n)
    if self.time_window_after == 0:
        cols.append(df)
        names += [f'{self.vital_signs[j]}(t)' for j in range(n_vars)]
    else:
        for i in range(0, self.time_window_after + 1):
            cols.append(df.shift(-i))
            if i == 0:
                names += [f'{self.vital_signs[j]}(t)' for j in range(n_vars)]
            else:
                names += [f'{self.vital_signs[j]}(t+{i})' for j in range(n_vars)]
    
    # put it all together
    df_all = pd.concat(cols, axis=1)
    df_all.columns = names
    
    # drop rows with NaN values (the first time_window_before rows)
    df_all.dropna(inplace=True)
    
    return df_all
