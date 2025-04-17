"""
This module contains a class to help prepare the data for LSTM models.
"""

import pandas as pd
import numpy as np

class DataPreparator():
    """
    A class to generate sequences of data for LSTM from time series and clinical information.
    """
    def __init__(self, df_time_series, df_clinical_info, time_window_before=10,
                 target_col='insp_sevo', static_features=None, test_size=0.2):
        self.df_time_series = df_time_series
        self.df_clinical_info = df_clinical_info
        self.time_window_before = time_window_before  # Number of time steps before the target
        self.target_col = target_col
        self.static_features = static_features  # Static features to be used in the model (biometric data)
        self.test_size = test_size  # Size of the test set
        self.X = None
        self.y = None
        self.caseids_extended = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.train_mask = None
        self.test_mask = None
        self.train_ids = None
        self.test_ids = None
    
    def normalize_static_features(self):
        """
        Normalize static features to be on the same scale as time series data
        """
        if 'sex' in self.static_features:
            self.df_clinical_info['sex'] = self.df_clinical_info['sex'].replace({'M': 1, 'F': 0})

        if self.static_features:
            # Simple min-max normalization to range [0,1] to match your time series
            for feature in self.static_features:
                if feature in self.df_clinical_info.columns and feature != 'sex':  # Skip sex if already converted
                    min_val = self.df_clinical_info[feature].min()
                    max_val = self.df_clinical_info[feature].max()
                    if max_val > min_val:  # Avoid division by zero
                        self.df_clinical_info[feature] = (self.df_clinical_info[feature] - min_val) / (max_val - min_val)

    def generate_sequences(self):
        """
        Creates the X, y arrays directly in format (samples, window, features) for LSTM
        
        Returns:
            X (np.array): Input sequences of shape (num_samples, window_size, num_features).
            y (np.array): Target values of shape (num_samples, 1).
            caseids_extended (list): Patient IDs corresponding to each sequence.
        """
        X, y, caseids_extended = [], [], []


        # Normalize static features if any
        if self.static_features:
            self.normalize_static_features()
            
        self.df_clinical_info.set_index('caseid', inplace=True)
        # Group by patient ID
        for caseid, df_patient in self.df_time_series.groupby("caseid"):
            df_patient = df_patient.sort_values("time")

            time_series_features = [col for col in df_patient.columns if col not in ["caseid", "time"]]
            data_values = df_patient[time_series_features].values
            target_values = df_patient[self.target_col].values

            # Get the values of the static features for the patient
            if self.static_features:
                static_row = self.df_clinical_info.loc[caseid, self.static_features]
                static_values = static_row.values if isinstance(static_row, pd.Series) else static_row.to_numpy()
            else:
                static_values = None

            # Calculate the number of sequences considering the time window
            max_t = len(df_patient) - self.time_window_before
            for i in range(max_t):
                X_seq = data_values[i:i+self.time_window_before]
                y_seq = target_values[i+self.time_window_before]

                if static_values is not None:
                    static_expanded = np.tile(static_values, (self.time_window_before, 1))
                    X_seq = np.hstack([X_seq, static_expanded])

                X.append(X_seq)
                y.append(y_seq)
                caseids_extended.append(caseid)

        self.X = np.array(X)
        self.y = np.array(y)
        self.caseids_extended = caseids_extended

        return self.X, self.y, self.caseids_extended

    def split_train_test(self, random_state=42):
        """
        Split the data into train and test sets
        
        Parameters:
        - X_all: list of np.array shape (n_samples, window, n_features)
        - y_all: list of np.array shape (n_samples,)
        - caseids_extended: list of caseids for each sample

        Returns:
        - X_train: np.array shape (n_samples, window, n_features)
        - X_test: np.array shape (n_samples, window, n_features)
        - y_train: np.array shape (n_samples,)
        - y_test: np.array shape (n_samples,)
        - train_mask: np.array shape (n_samples,) boolean mask 
        - test_mask: np.array shape (n_samples,) boolean mask
        - train_ids: list of caseids for the training set
        - test_ids: list of caseids for the testing set        
        """
        from sklearn.model_selection import train_test_split

        unique_caseids = np.unique(self.caseids_extended)
        train_ids, test_ids = train_test_split(unique_caseids, test_size=self.test_size, random_state=random_state)

        # Create masks for train and test sets
        train_mask = np.isin(self.caseids_extended, train_ids)
        test_mask = np.isin(self.caseids_extended, test_ids)

        # Convert the data to numpy arrays and apply the masks
        # Convert to float32 for compatibility with TensorFlow
        self.X_train = np.array(self.X, dtype=np.float32)[train_mask]
        self.X_test = np.array(self.X, dtype=np.float32)[test_mask]
        self.y_train = np.array(self.y, dtype=np.float32)[train_mask]
        self.y_test = np.array(self.y, dtype=np.float32)[test_mask]
        self.train_mask = train_mask
        self.test_mask = test_mask
        self.train_ids = train_ids
        self.test_ids = test_ids

        return self.X_train, self.X_test, self.y_train, self.y_test, self.train_mask, self.test_mask, self.train_ids, self.test_ids

