import pandas as pd
import joblib
from sqlalchemy import create_engine
from helpers import LSTMModelTrainer

#Load data from MySQL database
engine = create_engine("mysql://root:Ana.mysql.18@127.0.0.1/vitaldb_anesthesiaml")
df = pd.read_sql('SELECT * FROM vitaldb_preprocessed', con=engine)
df_clinical_info = pd.read_sql('SELECT * FROM vitaldb_clinical_info', con=engine)


# Create an instance of LSTMModelTrainer with the loaded data
model_trainer = LSTMModelTrainer(df, df_clinical_info, time_window_before=10, static_features=['age', 'sex', 'asa'])

# Get the data in the required format for LSTM
model_trainer.generate_sequences()

# Split the data into training and testing sets
model_trainer.split_train_test()

# Optimize hyperparameters
study = model_trainer.optimize_hyperparameters(n_trials=25, timeout=600)

# Show the results of the study
model_trainer.show_result(study)

# Save the object with the generated sequences and split data to a file
joblib.dump(model_trainer, 'model_trainer.pkl')

# Get the training and testing data into variables
X_train = model_trainer.X_train
X_test =y_train = model_trainer.y_train
y_test = model_trainer.y_test
train_mask = model_trainer.train_mask
test_mask = model_trainer.test_mask
train_ids = model_trainer.train_ids
test_ids = model_trainer.test_ids