import pandas as pd
import joblib
import time
from sqlalchemy import create_engine
from helpers import LSTMModelTrainer

# Load data from MySQL database
print("Loading data from MySQL database...")
start = time.time()
engine = create_engine("mysql://root:Ana.mysql.18@127.0.0.1/vitaldb_anesthesiaml")
df = pd.read_sql('SELECT * FROM vitaldb_preprocessed', con=engine)
df_clinical_info = pd.read_sql('SELECT * FROM vitaldb_clinical_info', con=engine)
end = time.time()
print(f"Data loaded successfully (took {end - start:.2f} seconds)")

caseids = df['caseid'].unique()[:15]  # Limit to 15 caseids for testing
df = df[df['caseid'].isin(caseids)]

print("Data shape:", df.shape)
print("Clinical info shape:", df_clinical_info.shape)

print("\nCreating LSTMModelTrainer instance...")
start = time.time()
model_trainer = LSTMModelTrainer(df, df_clinical_info, time_window_before=10, static_features=['age', 'sex', 'asa'])
end = time.time()

print("\nGenerating sequences...")
start = time.time()
model_trainer.generate_sequences()
end = time.time()
print(f"Sequences generated successfully (took {end - start:.2f} seconds)")

print("\nSplitting data into training and testing sets...")
start = time.time()
model_trainer.split_train_test()
end = time.time()
print(f"Data split successfully (took {end - start:.2f} seconds)")

# Save the object with the generated sequences and split data to a file
print('Saving model_trainer object...')
start = time.time()
joblib.dump(model_trainer, 'model_trainer.pkl')
end = time.time()
print(f"model_trainer object saved successfully (took {end - start:.2f} seconds)")

print("\nCreating LSTM model (hyperparameter tuning)...")
start = time.time()
study = model_trainer.optimize_hyperparameters(storage="mysql://root:Ana.mysql.18@127.0.0.1/vitaldb_anesthesiaml")
end = time.time()
print(f"Hyperparameters optimized successfully (took {end - start:.2f} seconds)")

print('Saving study...')
start = time.time()
joblib.dump(study, 'study.pkl')
end = time.time()
print(f"Study saved successfully (took {end - start:.2f} seconds)")

print('\nResults:')
model_trainer.show_result(study)

best_params = study.best_trial.params
print("Best parameters:", best_params)

print("\nCreating the best model...")
start = time.time()
model_trainer.create_model(best_params)
end = time.time()
print(f"Best model created successfully (took {end - start:.2f} seconds)")
