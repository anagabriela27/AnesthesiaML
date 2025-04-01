"""
This module contains helper classes and functions to preprocess and visualize the data.
"""

import pandas as pd
import numpy as np
import vitaldb
from sklearn.preprocessing import MinMaxScaler

class DataPreprocessor:
    """
    A class to preprocess the data for a given caseid.
    """
    def __init__(self, caseid, vital_signs,clinical_info):
        self.caseid = caseid
        self.vital_signs = vital_signs
        self.clinical_info = clinical_info

        # Initialize the min and max values for each vital sign
        self.min_values = None
        self.max_values = None

    def create_case_df(self):
        """
        Load the data corresponding to one caseid and create a dataframe from the case values.
        """
        # Load the data
        case_values = vitaldb.load_case(self.caseid, ['Solar8000/ART_DBP', 'Solar8000/ART_SBP',
                                                      'Solar8000/ART_MBP', 'Solar8000/HR',
                                                        'Solar8000/PLETH_SPO2', 'BIS/BIS',
                                                          'Primus/EXP_SEVO','Primus/INSP_SEVO'], 1)
        
        case_values = np.round(case_values, 4)
        # Create a dataframe
        case_df = pd.DataFrame(case_values, columns=self.vital_signs)
        case_df['caseid'] = self.caseid
        case_df['time'] = np.arange(0, len(case_df), 1)
        case_df = case_df[['caseid','time'] + self.vital_signs]
        return case_df

    def check_all_signs(self, case_df):
        """
        To check if there is data different from 0 in all the signs
        (i.e.: if there is data different from 0 and null for each column)
        """
        flag = True
        for sign in self.vital_signs:
            # Check if there are values different from 0 and null
            if not case_df[sign].dropna().ne(0).any():
                flag = False
                break
        return flag

    def get_maintenance_phase(self, case_df):
        """
        Get the maintenance phase of the anesthesia for each case.
        """
        # Get operations start and end time for each case
        df_merged = case_df.merge(self.clinical_info[['caseid', 'opstart', 'opend']], on='caseid')

        # Get the data corresponding only to the operation time (maintenance part of the anesthesia)
        df_maintenance = df_merged.query("opstart <= time <= opend").drop(columns=['opstart', 'opend'])

        return df_maintenance

    def set_outliers_to_null(self, case_df):
        """
        Set the outliers to null values.
        """
        case_df_nooutliers = case_df.copy()

        # For bis column, set values <0 and >100 as NaN:
        case_df_nooutliers['bis'] = np.where((case_df_nooutliers['bis'] < 0) | (case_df_nooutliers['bis'] > 100), np.nan, case_df_nooutliers['bis'])

        # For spo2 column, set values <50 and >100 as NaN:
        case_df_nooutliers['spo2'] = np.where((case_df_nooutliers['spo2'] < 50) | (case_df_nooutliers['spo2'] > 100), np.nan, case_df_nooutliers['spo2'])

        # For exp_sevo column, set values <0 and >30 as NaN:
        case_df_nooutliers['exp_sevo'] = np.where((case_df_nooutliers['exp_sevo'] < 0) | (case_df_nooutliers['exp_sevo'] > 8), np.nan, case_df_nooutliers['exp_sevo'])

        # For insp_sevo column, set values <0 and >30 as NaN:
        case_df_nooutliers['insp_sevo'] = np.where((case_df_nooutliers['insp_sevo'] < 0) | (case_df_nooutliers['insp_sevo'] > 8), np.nan, case_df_nooutliers['insp_sevo'])

        # For Pulso column, set values <20 and >220 as NaN:
        case_df_nooutliers['hr'] = np.where((case_df_nooutliers['hr'] < 20) | (case_df_nooutliers['hr'] > 220), np.nan, case_df_nooutliers['hr'])

        # For sbp column, set values <20 and >300 as NaN:
        case_df_nooutliers['sbp'] = np.where((case_df_nooutliers['sbp'] < 20) | (case_df_nooutliers['sbp'] > 300), np.nan, case_df_nooutliers['sbp'])

        # For dbp column, set values <5 and >300 as NaN:
        case_df_nooutliers['dbp'] = np.where((case_df_nooutliers['dbp'] < 5) | (case_df_nooutliers['dbp'] > 225), np.nan, case_df_nooutliers['dbp'])

        # For mbp column, set values <20 and >300 as NaN:
        case_df_nooutliers['mbp'] = np.where((case_df_nooutliers['mbp'] < 20) | (case_df_nooutliers['mbp'] > 300), np.nan, case_df_nooutliers['mbp'])

        # Conditions to set sbp, dbp and mbp as NaN
        condition1 = (case_df_nooutliers['sbp'] - case_df_nooutliers['dbp']) <= 5
        condition2 = (case_df_nooutliers['sbp'] - case_df_nooutliers['dbp']) >= 200
        condition3 = (case_df_nooutliers['sbp'] - case_df_nooutliers['mbp']) <= 3

        case_df_nooutliers.loc[condition1 | condition2 | condition3, ['mbp', 'sbp', 'dbp']] = np.nan

        return case_df_nooutliers

    def data_imputation(self, case_df):
        """
        Impute the missing values in the dataframe.
        Using linear interpolation to fill the missing values.
        """
        # Impute the missing values using linear interpolation
        case_df_imputed = case_df.copy()
        case_df_imputed = case_df_imputed.interpolate(method='linear', limit_direction='both')

        # Forward fill and backward fill to handle boundary conditions
        case_df_imputed = case_df_imputed.ffill().bfill()

        return case_df_imputed

    def normalize_data(self, case_df):
        """
        Normalize the data in the dataframe using MinMaxScaler.
        """
        # Initialize the MinMaxScaler
        scaler = MinMaxScaler()
        case_df_normalized = case_df.copy()
        
        # Normalize the data (except the caseid and time columns)
        case_df_normalized[self.vital_signs] = scaler.fit_transform(case_df[self.vital_signs])
        
        return case_df_normalized, scaler

    def denormalize_data(self, normalized_data, scalers):
        """
        Denormalize the data using the scaler.
        """
        denormalized_data = normalized_data.copy()
        scaler = scalers[self.caseid]
        denormalized_data[self.vital_signs] = scaler.inverse_transform(normalized_data[self.vital_signs])
        
        return denormalized_data

    def preprocess_data(self):
        """
        Preprocess the data for a given caseid
        """
        # Create a dataframe for the case
        case_df = self.create_case_df()

        # Get the maintenance phase of the anesthesia for each case
        case_df_maintenance = self.get_maintenance_phase(case_df)

        # Set the outliers to null values
        case_df_nooutliers = self.set_outliers_to_null(case_df_maintenance)

        # Check if there is data different from 0 in all the signs after removing the outliers
        if self.check_all_signs(case_df_nooutliers):
            # Impute the missing values in the dataframe
            case_df_imputed = self.data_imputation(case_df_nooutliers)
            case_df_normalized, scaler = self.normalize_data(case_df_imputed)
            return case_df_normalized, (self.caseid,scaler)
        return None, None

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
        

    def sex_to_numerical(self):
        """
        If sex is one of the static features, convert it to numerical values.
        M = 1, F = 0
        """
        if 'sex' in self.static_features:
            self.df_clinical_info['sex'] = self.df_clinical_info['sex'].replace({'M': 1, 'F': 0})

    def generate_sequences(self):
        """
        Creates the X, y arrays directly in format (samples, window, features) for LSTM
        
        Returns:
            X (np.array): Input sequences of shape (num_samples, window_size, num_features).
            y (np.array): Target values of shape (num_samples, 1).
            caseids_extended (list): Patient IDs corresponding to each sequence.
        """
        X, y, caseids_extended = [], [], []

        if 'sex' in self.static_features:
            self.sex_to_numerical()

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

        # Máscaras booleanas
        train_mask = np.isin(self.caseids_extended, train_ids)
        test_mask = np.isin(self.caseids_extended, test_ids)

        self.X_train = self.X[train_mask]
        self.y_train = self.y[train_mask]
        self.X_test = self.X[test_mask]
        self.y_test = self.y[test_mask]
        self.train_mask = train_mask
        self.test_mask = test_mask
        self.train_ids = train_ids
        self.test_ids = test_ids

        return self.X_train, self.X_test, self.y_train, self.y_test, self.train_mask, self.test_mask, self.train_ids, self.test_ids

class CreateLSTM():
    """
    A class to create and train an LSTM model.
    """
    def __init__(self, input_shape, output_shape, lstm_units=50, dropout_rate=0.2):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate

    def build_model(self):
        """
        Build the LSTM model.
        """
        from keras.models import Sequential
        from keras.layers import LSTM, Dense, Dropout

        model = Sequential()
        model.add(LSTM(self.lstm_units, input_shape=self.input_shape, return_sequences=True))
        model.add(Dropout(self.dropout_rate))
        model.add(LSTM(self.lstm_units))
        model.add(Dropout(self.dropout_rate))
        model.add(Dense(self.output_shape))

        model.compile(optimizer='adam', loss='mean_squared_error')

        return model