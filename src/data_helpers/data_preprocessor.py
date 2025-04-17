"""
Scope: Data Preprocessing
Brief: This module contains a helper class with functions to preprocess the data.
"""

import pandas as pd
import numpy as np
import vitaldb
from sklearn.preprocessing import MinMaxScaler


def select_caseids(df_trks, clinical_info):
    
    """
    Select the caseids based on:
    - The availability of all the vital signs
    - Clinical information (age >18, weight > 35, asa < 4, ane_type = General,
      initial bolus of propofol > 0)
    - Filter out the cases with desflurane, propofol, and remifentanil (we are focusing on sevoflurane)

    Args:
        df_trks (pd.DataFrame): Dataframe containing the vital signs data.
        clinical_info (pd.DataFrame): Dataframe containing the clinical information.

    Returns:
        caseids (List): List of caseids that meet the criteria.
    """
    # Filter the caseids based on the availability of all the vital signs
    # Also filter based on the clinical information (age >18, weight > 35, asa < 4, ane_type = General,
    # initial bolus of propofol > 0)
    caseids = (
        set(df_trks.loc[df_trks['tname'] == 'Solar8000/ART_DBP', 'caseid']) &
        set(df_trks.loc[df_trks['tname'] == 'Solar8000/ART_SBP', 'caseid']) &
        set(df_trks.loc[df_trks['tname'] == 'Solar8000/ART_MBP', 'caseid']) &
        set(df_trks.loc[df_trks['tname'] == 'Solar8000/HR', 'caseid']) &
        set(df_trks.loc[df_trks['tname'] == 'Solar8000/PLETH_SPO2', 'caseid']) &
        set(df_trks.loc[df_trks['tname'] == 'BIS/BIS', 'caseid']) & 
        set(df_trks.loc[df_trks['tname'] == 'Primus/INSP_SEVO', 'caseid']) &
        set(df_trks.loc[df_trks['tname'] == 'Primus/EXP_SEVO', 'caseid']) &
        set(clinical_info.loc[clinical_info['age'] > 18, 'caseid']) & 
        set(clinical_info.loc[clinical_info['weight'] > 35, 'caseid']) &
        set(clinical_info.loc[clinical_info['asa'] < 4, 'caseid']) &
        set(clinical_info.loc[clinical_info['ane_type'] == 'General', 'caseid']) &
        set(clinical_info.loc[clinical_info['intraop_ppf'] > 0, 'caseid'])
    )
    # Filter out the cases with desflurane, prpofol, and remifentanil (we are focusing on sevoflurane)
    caseids -= set(df_trks.loc[df_trks['tname'] == 'Primus/EXP_DES', 'caseid'])
    caseids -= set(df_trks.loc[df_trks['tname'] == 'Primus/INSP_DES', 'caseid'])
    caseids -= set(df_trks.loc[df_trks['tname'] == 'Orchestra/PPF20_CE', 'caseid'])
    caseids -= set(df_trks.loc[df_trks['tname'] == 'Orchestra/RFTN50_CE', 'caseid'])
    caseids = list(caseids)
    return caseids

def data_loader(caseids):
    """
    Load the data for the given caseids.

    Args:
        caseids (List): List of caseids to load data for.

    Returns:
        df (pd.DataFrame): Dataframe containing the loaded data.
    """
    # Load the data for the given caseids
    df = vitaldb.load_case(caseids, ['Solar8000/ART_DBP', 'Solar8000/ART_SBP',
                                     'Solar8000/ART_MBP', 'Solar8000/HR',
                                     'Solar8000/PLETH_SPO2', 'BIS/BIS',
                                     'Primus/EXP_SEVO','Primus/INSP_SEVO'], 1)
    
    return df

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
