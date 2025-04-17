"""
Scope:_ Modeling
Brief: Prepare the data for modeling and train the LSTM model with hyperparameter tuning.
"""

# Standard library imports
import os
import time

# Third-party imports
import pandas as pd
import joblib
from sqlalchemy import create_engine

# Local application imports
from models_helpers import LSTMModelTrainer
from utils import file_utils

SQL_STORAGE = "mysql://root:Ana.mysql.18@127.0.0.1/vitaldb_anesthesiaml"
PROJECT_ROOT = file_utils.get_project_root()
OUTPUTS_PATH = os.path.join(PROJECT_ROOT, "outputs")
RUN_PATH = file_utils.create_run_folder()

engine = create_engine(SQL_STORAGE)
logger = file_utils.configure_logger(RUN_PATH, log_name="run.log")

def main():
    logger.info("Loading data from MySQL database...")
    start = time.time()
    df = pd.read_sql('SELECT * FROM vitaldb_preprocessed', con=engine)
    df_clinical_info = pd.read_sql('SELECT * FROM vitaldb_clinical_info', con=engine)
    end = time.time()
    logger.info("Data loaded successfully (took %.2f minutes)", (end - start) / 60)

    logger.info("Data shape: %s", df.shape)
    logger.info("Clinical info shape: %s", df_clinical_info.shape)

    logger.info("Creating LSTMModelTrainer instance...")
    model_trainer = LSTMModelTrainer(df, df_clinical_info, time_window_before=10, 
                                     static_features=['age', 'sex', 'asa'], logger=logger)

    logger.info("Generating sequences...")
    start = time.time()
    model_trainer.generate_sequences()
    end = time.time()
    logger.info("Sequences generated successfully (took %.2f minutes)", (end - start) / 60)

    logger.info("Splitting data into training and testing sets...")
    model_trainer.split_train_test()

    logger.info("Saving model_trainer object...")
    start = time.time()
    joblib.dump(model_trainer, os.path.join(RUN_PATH, 'model_trainer.pkl'))
    end = time.time()
    logger.info("model_trainer object saved successfully (took %.2f minutes)", (end - start) / 60)

    logger.info("Starting hyperparameter tuning...")
    start = time.time()
    study = model_trainer.optimize_hyperparameters(storage=SQL_STORAGE)
    end = time.time()
    logger.info("Hyperparameters optimized successfully (took %.2f minutes)", (end - start) / 60)

    for t in study.trials:
        logger.info("Trial %d - State: %s - Value: %s", t.number, t.state, t.value)

    logger.info("Saving study...")
    joblib.dump(study, os.path.join(RUN_PATH, 'study.pkl'))

    best_params = study.best_trial.params
    logger.info("Best parameters: %s", best_params)

    logger.info("Saving model_trainer object...")
    start = time.time()
    joblib.dump(model_trainer, os.path.join(RUN_PATH, 'model_trainer_after.pkl'))
    end = time.time()
    logger.info("model_trainer object saved successfully (took %.2f minutes)", (end - start) / 60)

    logger.info("Creating the best model...")
    best_model = model_trainer.create_model_architecture(best_params)

    logger.info("Training the best model on full training set...")
    start = time.time()
    history = best_model.fit(
        model_trainer.X_train, model_trainer.y_train,
        validation_split=0.2,
        epochs=15,
        batch_size=best_params["batch_size"],
        verbose=1
    )
    end = time.time()
    logger.info("Model trained successfully (took %.2f minutes)", (end - start) / 60)

    logger.info("Saving the trained model...")
    best_model.save(os.path.join(RUN_PATH, 'best_model.keras'))

    logger.info("Saving the training history...")
    with open(os.path.join(RUN_PATH, 'history.pkl'), 'wb') as f:
        joblib.dump(history.history, f)
    logger.info("Training history saved successfully")

if __name__ == "__main__":
    main()
