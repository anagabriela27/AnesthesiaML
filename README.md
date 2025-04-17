# Sevoflurane Forecasting with LSTM

This project aims to forecast the inspired sevoflurane concentration using time-series data of vital signs and clinical information during general anesthesia. The pipeline includes data extraction, preprocessing, LSTM model training, hyperparameter tuning, and result visualization.
---
## 📊 Data

This project uses real-world physiological and clinical data retrieved from the open-source [VitalDB dataset](https://vitaldb.net/dataset/).

Reference: Lee HC, Park Y, Yoon SB, Yang SM, Park D, Jung CW. VitalDB, a high-fidelity multi-parameter vital signs database in surgical patients. Sci Data. 2022 Jun 8;9(1):279. doi: 10.1038/s41597-022-01411-5. PMID: 35676300; PMCID: PMC9178032.

## 🧪 Pipeline Overview

### 1. **Data Extraction & Preprocessing**
- Handled in `extract_preprocess.py`.
- Extracts data for selected `caseids`, saves a MySQL-ready dataframe, and a `scalers.pkl` file used for later re-normalization.
- Scalers are used for converting predictions back to the original range for comparison and visualization.

### 2. **Model Training**
- Managed by `model.py`.
- Loads preprocessed data, splits it into sequences and in training aand test groups and defines an LSTM model.
- Hyperparameters are tuned using Optuna and the best model is after trained and saved.

### 3. **Outputs**
- The scalers.pkl file originated in `extract_preprocess.py`  is saved here
- Every run is saved in a  `run-N` folder inside `outputs/`, containing:
  - Trained model (`best_model.keras`)
  - Pickled model trainer (`model_trainer.pkl`)
  - Optimization study (`study.pkl`)
  - Log files and plots

---

## 🔧 Modules Descriptions

### `data_helpers/data_preprocessor.py`
Contains helper functions and the main class for preprocessing:
- `select_caseids(df_trks, clinical_info)`: Filters patients based on:
  - Availability of all vital signs
  - Age > 18, Weight > 35, ASA < 4, anesthesia type = General, and initial bolus of propofol > 0
  - Excludes use of desflurane, propofol, remifentanil
  Note: These criteria were applied in agreement with specialized doctors.
- `data_loader(caseids)`: Loads time-series data for the selected cases
- `DataPreprocessor` class:
  - `create_case_df()`: Creates a dataframe for one case
  - `check_all_signs(case_df)`: Ensures all vital signs have valid values
  - `get_maintenance_phase(case_df)`: Extracts the maintenance phase of anesthesia
  - `set_outliers_to_null(case_df)`: Detects and nullifies outliers
  - `data_imputation(case_df)`: Uses linear interpolation for missing values
  - `normalize_data(case_df)`: MinMaxScaler-based normalization
  - `preprocess_data()`: Calls all the above steps in a pipeline

### `data_helpers/data_preparator.py`
Contains helper functions and the main class for preparing the data that goes into the model.
- `DataPreparator` class:
  - `normalize_static_features`: Converts and scales static features to be compatible with the dynamic time series input.



### `models_helpers/lstm_model_trainer.py`
- Contains helper functions and the mainclass for LSTM model definition, training, evaluation, and hyperparameter tuning with Optuna.
- `LSTMModelTrainer` class (extends `DataPreparator`):
  - `create_model_architecture(params)`: Builds and compiles an LSTM model based on hyperparameters.
  - `objective(trial)`: Defines the Optuna objective function for tuning hyperparameters.
  - `optimize_hyperparameters(storage, n_trials, study_name)`: Runs Optuna to find the best set of hyperparameters.

### `utils/file_utils.py`
- Includes utility functions for creating timestamped output folders and saving files.
- `get_project_root()`: get the absolute path to the project root directory.
- `create_run_folder(base_path="outputs")`: Create a new numbered run folder with a subfolder for plots.
- `configure_logger(run_path, log_name="run.log")`: Configure a logger to log messages.
---

## 📊 Notebooks
- `eda.ipynb`: Performs exploratory data analysis on the dataset.
- `results_analysis.ipynb`: Analyzes model predictions and performance metrics.

---

## 📌 Notes
- All preprocessing is performed per `caseid` to ensure personalized normalization.
- Model predictions are always re-normalized using the original `scalers.pkl` values for true comparison.
- Data is filtered based on medically guided inclusion/exclusion criteria.
