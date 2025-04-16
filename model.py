import pandas as pd
import joblib
import time
import os
from sqlalchemy import create_engine
from helpers import LSTMModelTrainer

SQL_STORAGE = "mysql://root:Ana.mysql.18@127.0.0.1/vitaldb_anesthesiaml"
RESULTS_PATH = os.path.join(os.getcwd(), "training_results")

engine = create_engine(SQL_STORAGE)


def main():
    # Load data from MySQL database
    print("Loading data from MySQL database...")
    start = time.time()
    df = pd.read_sql('SELECT * FROM vitaldb_preprocessed', con=engine)
    df_clinical_info = pd.read_sql('SELECT * FROM vitaldb_clinical_info', con=engine)
    end = time.time()
    print(f"Data loaded successfully (took {(end - start) / 60:.2f} minutes)")

    # caseids = df['caseid'].unique()[:15]  # Limit to 15 caseids for testing
    # df = df[df['caseid'].isin(caseids)]

    print("Data shape:", df.shape)
    print("Clinical info shape:", df_clinical_info.shape)

    print("\nCreating LSTMModelTrainer instance...")
    model_trainer = LSTMModelTrainer(df, df_clinical_info, time_window_before=10, static_features=['age', 'sex', 'asa'])

    print("\nGenerating sequences...")
    start = time.time()
    model_trainer.generate_sequences()
    end = time.time()
    print(f"Sequences generated successfully (took {(end - start) / 60:.2f} minutes)")

    print("\nSplitting data into training and testing sets...")
    model_trainer.split_train_test()


    # Save the object with the generated sequences and split data to a file
    print('Saving model_trainer object...')
    start = time.time()
    joblib.dump(model_trainer, 'model_trainer.pkl')
    end = time.time()
    print(f"model_trainer object saved successfully (took {(end - start) / 60:.2f} minutes)")

    print("\nCreating LSTM model (hyperparameter tuning)...")
    start = time.time()
    study = model_trainer.optimize_hyperparameters(storage=SQL_STORAGE)
    end = time.time()
    print(f"Hyperparameters optimized successfully (took {(end - start) / 60:.2f} minutes)")

    for t in study.trials:
        print(f"Trial {t.number} - State: {t.state} - Value: {t.value}")

    print('Saving study...')
    joblib.dump(study, 'study.pkl')

       
    print('Results:')
    best_params = study.best_trial.params
    print("Best parameters:", best_params)


    print("\nCreating the best model...")
    best_model = model_trainer.create_model_architecture(best_params)

    print("\nTraining the best model on full training set...")
    start = time.time()
    best_model.fit(
        model_trainer.X_train, model_trainer.y_train,
        validation_split=0.2,
        epochs=15,
        batch_size=best_params["batch_size"],
        verbose=1
    )
    end = time.time()
    print(f"Model trained successfully (took {(end - start) / 60:.2f} minutes)")

    print("\nSaving the trained model...")
    best_model.save(os.path.join(RESULTS_PATH,'best_model.keras'))  # or "best_model.h5"



if __name__ == "__main__":
    main()