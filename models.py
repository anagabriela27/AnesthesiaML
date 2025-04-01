import pandas as pd
import joblib
from sqlalchemy import create_engine
from helpers import DataPreparator

#Load data from MySQL database
engine = create_engine("mysql://root:Ana.mysql.18@127.0.0.1/vitaldb_anesthesiaml")
df = pd.read_sql('SELECT * FROM vitaldb_preprocessed', con=engine)
df_clinical_info = pd.read_sql('SELECT * FROM vitaldb_clinical_info', con=engine)

# Create an instance of DataPreparator with the loaded data
data_prep = DataPreparator(df, df_clinical_info, time_window_before=10, 
                           static_features=['age', 'sex', 'asa'])

# Get the data in the required format for LSTM
data_prep.generate_sequences()

# Split the data into training and testing sets
data_prep.split_train_test()

# Save the object with the generated sequences and splitted data to a file
joblib.dump(data_prep, 'data_prep.pkl')


# Get the training and testing data into variables
X_train = data_prep.X_train
X_test = data_prep.X_test
y_train = data_prep.y_train
y_test = data_prep.y_test
train_mask = data_prep.train_mask
test_mask = data_prep.test_mask
train_ids = data_prep.train_ids
test_ids = data_prep.test_ids


# Save the data 