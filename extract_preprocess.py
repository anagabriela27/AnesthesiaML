import time
from datetime import datetime
import multiprocessing

import pandas as pd
import psutil
from sqlalchemy import create_engine

import joblib
from helpers import DataPreprocessor

# Connect to the database
engine = create_engine("mysql://root:Ana.mysql.18@127.0.0.1/vitaldb_anesthesiaml")

# Load and save the data
clinical_info = pd.read_csv("https://api.vitaldb.net/cases")
df_trks = pd.read_csv("https://api.vitaldb.net/trks")

# Vital signs to be extracted
vital_signs = ['dbp', 'sbp', 'mbp', 'hr', 'spo2', 'bis', 'exp_sevo', 'insp_sevo']

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
#caseids = caseids[:100]

def preprocess_case(caseid, clinical_info_df, signs):
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
    preprocessed_df.to_sql('vitaldb_preprocessed', con=engine, if_exists='replace', index=False)
    end_time_save = time.time()
    print(f"Total time taken for saving the dataframe: {end_time_save - start_time_save:.2f} seconds")
    
    # Save the scalers as a dictionary to a pickle file
    joblib.dump(dict(scalers_list), 'scalers.pkl')
    
    return preprocessed_df

if __name__ == '__main__':
    multiprocessing.freeze_support()
    df = main(clinical_info)
    
    print('\n Number of null values in the dataframe:')
    print(df.isnull().sum())

    print('\n Final number of case ids:')
    print(df['caseid'].nunique())