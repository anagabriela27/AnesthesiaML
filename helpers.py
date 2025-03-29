"""
This module contains helper classes and functions to preprocess and visualize the data.
"""

import pandas as pd
import numpy as np
import vitaldb
import joblib
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


class DataPreparation(DataPreprocessor):
    """
    Prepares data for the models
    """
    def __init__(self, caseid, vital_signs, clinical_info,time_window_before=10, time_window_after=1,
                 target = 'insp_sevo', test_size=0.2, random_state=42):
        super().__init__(caseid, vital_signs,clinical_info)
        self.time_window_before = time_window_before
        self.time_window_after = time_window_after
        self.test_size = test_size
        self.random_state = random_state
        self.target = target
        self.scalers = {}

       
    def build_lstm_case_arrays(self, case_df):
        """
        Creates the X, y arrays directly in format (samples, window, features) for LSTM
        from a patient DataFrame (after normalization and imputation).
        
        Returns:
        - X: np.array shape (n_samples, window, n_features)
        - y: np.array shape (n_samples,)
        """
        input_signals = self.vital_signs
        target_signal = self.target

        data = case_df[input_signals + [target_signal]].values
        X_list = []
        y_list = []

        time_steps = self.time_window_before
        for i in range(len(data) - time_steps - self.time_window_after + 1):
            x_seq = data[i:i+time_steps, :-1]  # sinais vitais (exclui target)
            y_val = data[i+time_steps + self.time_window_after - 1, -1]  # target no t+1
            X_list.append(x_seq)
            y_list.append(y_val)

        return np.array(X_list), np.array(y_list)

    def create_lstm_dataset(self,caseids):
        """
        Creates the X, y arrays directly in format (samples, window, features) for LSTM
        from a patient DataFrame (after normalization and imputation).
        
        Returns:
        - X: np.array shape (n_samples, window, n_features)
        - y: np.array shape (n_samples,)
        - caseids_extended: list of caseids for each sample
        """
        X_all = []
        y_all = []
        caseids_extended = []

        for cid in caseids:
            self.caseid = cid
            df = df[df['caseid'] == cid]

            if df is not None:
                X, y = self.build_lstm_case_arrays(df)
                X_all.append(X)
                y_all.append(y)
                caseids.extend([cid]*len(y))  # keep track of caseids
        return X_all, y_all, caseids_extended
    
    def split_train_test(self, X_all, y_all, caseids_extended):
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

        X_all = np.concatenate(X_all, axis=0)   # shape: (total_samples, 300, 7)
        y_all = np.concatenate(y_all, axis=0)   # shape: (total_samples,)
        caseids = np.array(caseids_extended)             # shape: (total_samples,)

        unique_caseids = np.unique(caseids)
        train_ids, test_ids = train_test_split(unique_caseids, test_size=self.test_size, random_state=self.random_state)

        # Máscaras booleanas
        train_mask = np.isin(caseids, train_ids)
        test_mask = np.isin(caseids, test_ids)

        X_train = X_all[train_mask]
        y_train = y_all[train_mask]
        X_test = X_all[test_mask]
        y_test = y_all[test_mask]

        return X_train, X_test, y_train, y_test, train_mask, test_mask, train_ids, test_ids


class SequenceGenerator:
    def __init__(self, df_time_series, df_clinical_info, time_window_before = 10,time_window_after = 1, target_col = 'insp_sevo', static_features=['age','sex','weight','height','asa']):
        self.df_time_series = df_time_series
        self.df_clinical_info = df_clinical_info.set_index("caseid")  # Assumes 'caseid' is the key
        self.time_window_before = time_window_before # Number of time steps before the target
        self.time_window_after = time_window_after # Number of time steps after the target
        self.target_col = target_col

        #Static features to be used in the model (biometric data)
        self.static_features = static_features if static_features else []

    def sex_to_numerical(self):
        """
        If sex is one of the static features, convert it to numerical values.
        M = 1, F = 0
        """
        if 'sex' in self.static_features:
            self.df_clinical_info['sex']=self.df_clinical_info['sex'].map({'M': 1, 'F': 0})


    def generate_sequences(self):
        """
        Creates the X, y arrays directly in format (samples, window, features) for LSTM
            
        Returns:
            X (np.array): Input sequences of shape (num_samples, window_size, num_features).
            y (np.array): Target values of shape (num_samples, 1).
            patient_ids (list): Patient IDs corresponding to each sequence.
        """
        X, y, patient_ids = [], [], []

        if 'sex' in self.static_features:
            self.sex_to_numerical()

        # Group by patient ID
        for caseid, df_patient in self.df_time_series.groupby("caseid"):
            df_patient = df_patient.sort_values("time")

            time_series_features = [col for col in df_patient.columns if col not in ["caseid", "time"]]
            data_values = df_patient[time_series_features].values
            target_values = df_patient[self.target_col].values

            #Get the values of the static features for the patient
            if self.static_features:
                static_row = self.df_clinical_info.loc[caseid, self.static_features]
                static_values = static_row.values if isinstance(static_row, pd.Series) else static_row.to_numpy()
            else:
                static_values = None

            # Calculate the number of sequences considering the time window
            # Example: if len(df) = 10, self.time_window_before = 3:
            # First sequence: data[0:3], target[4]; last sequence: data[7:10], target[10]
            max_t = len(df_patient) - self.time_window_before
            for i in range(max_t):
                X_seq = data_values[i:i+self.time_window_before]
                y_seq = target_values[i+self.time_window_before]

                if static_values is not None:
                    static_expanded = np.tile(static_values, (self.time_window_before, 1))
                    X_seq = np.hstack([X_seq, static_expanded])

                X.append(X_seq)
                y.append(y_seq)
                patient_ids.append(caseid)

        return np.array(X), np.array(y), patient_ids




def save_scalers(scaler, filename):
    """
    Save the scaler to a file using joblib.
    """
    joblib.dump(scaler, filename)

def load_scalers(filename):
    """
    Load the scaler from a file using joblib.
    """
    return joblib.load(filename)