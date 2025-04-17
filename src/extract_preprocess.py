"""
Scope: Pre-modeling
Brief: Extract and preprocess data from the VitalDB database.
"""

# Standard library imports
import time
import os
from datetime import datetime
import multiprocessing

# Third-party imports
import pandas as pd
import psutil
from sqlalchemy import create_engine
import joblib

# Local imports
from data_helpers import DataPreprocessor
from data_helpers.data_preprocessor import select_caseids
from utils import file_utils

SQL_STORAGE = "mysql://root:Ana.mysql.18@127.0.0.1/vitaldb_anesthesiaml2"
PROJECT_ROOT = file_utils.get_project_root()
OUTPUTS_PATH = os.path.join(PROJECT_ROOT, "outputs")

# Connect to the database
engine = create_engine(SQL_STORAGE)

# Load and save the data
clinical_info = pd.read_csv("https://api.vitaldb.net/cases")
df_trks = pd.read_csv("https://api.vitaldb.net/trks")

# Vital signs to be extracted
vital_signs = ['dbp', 'sbp', 'mbp', 'hr', 'spo2', 'bis', 'exp_sevo', 'insp_sevo']


caseids = select_caseids(df_trks, clinical_info)
#caseids = caseids[:5] 

def preprocess_case(caseid, clinical_info_df, signs):
    """
    Preprocess the data for a single caseid
    """
    preprocessor = DataPreprocessor(caseid, signs,clinical_info_df)
    normalized_data, id_scaler = preprocessor.preprocess_data()
    return normalized_data, id_scaler

def main(clinical_info_df):
    """
    Main function to extract and preprocess the data
    """
    # Preprocessing start time
    start_time = time.time()
    print(f"Preprocessing started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Get optimal number of worker processes
    num_physical_cores = psutil.cpu_count(logical=False)
    
    with multiprocessing.Pool(num_physical_cores) as pool:
        results = pool.starmap(preprocess_case, [(caseid, clinical_info_df, vital_signs) for caseid in caseids])
    
    # Filter out None results and separate dataframes and scalers
    processed_data = [result for result in results if result is not None]
    dataframes = [data[0] for data in processed_data if data[0] is not None]
    scalers_list = [data[1] for data in processed_data if data[1] is not None]
    
    # Concatenate all the processed DataFrames
    preprocessed_df = pd.concat(dataframes, ignore_index=True)

    # Preprocessing end time
    end_time_preprocess = time.time()
    print(f"Total time taken for preprocessing: {end_time_preprocess - start_time:.2f} seconds")

    # Save the final dataframe to sqlite database
    start_time_save = time.time()
    preprocessed_df.to_sql('vitaldb_preprocessed_delete', con=engine, if_exists='replace', index=False)
    end_time_save = time.time()
    print(f"Total time taken for saving the dataframe: {end_time_save - start_time_save:.2f} seconds")
    
    # Save the scalers as a dictionary to a pickle file
    joblib.dump(dict(scalers_list), os.path.join(OUTPUTS_PATH, 'scalers2.pkl'))
    
    return preprocessed_df

if __name__ == '__main__':
    multiprocessing.freeze_support()
    df = main(clinical_info)

    print('\n Final number of case ids:')
    print(df['caseid'].nunique())