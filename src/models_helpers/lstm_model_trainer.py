"""
Scope: Modeling
Brief: This module contains a class to help train and optimize an LSTM model using Optuna for hyperparameter tuning.
"""
import os
import optuna
from optuna.integration import TFKerasPruningCallback
from optuna.visualization import plot_param_importances, plot_optimization_history, plot_parallel_coordinate

from keras.backend import clear_session
from keras.callbacks import EarlyStopping
from keras.layers import LSTM, Dense, Dropout
from keras.metrics import RootMeanSquaredError
from keras.models import Sequential
from keras.optimizers import Adam

from data_helpers import DataPreparator

# pylint: disable=broad-except 

class LSTMModelTrainer(DataPreparator):
    """
    A class that extends DataPreparator to include methods for creating, training, and optimizing an LSTM model.
    """
    def __init__(self, df_time_series, df_clinical_info, time_window_before=10, 
                static_features=None, validation_split=0.2, epochs=30,study=None, best_params=None, logger=None):
        """
        Initialize the LSTMModelTrainer with the provided data and parameters.
        """
        super().__init__(df_time_series, df_clinical_info, time_window_before, 
                        target_col='insp_sevo', static_features=static_features)
        self.validation_split = validation_split
        self.epochs = epochs
        self.logger = logger
        self.study = study
        self.best_params = best_params

    def create_model_architecture(self, params):
        """
        Create and compile an LSTM model based on provided parameters.
        Args:
            params (dict): Dictionary containing hyperparameters for the model.
        Returns:
            model (keras.Model): Compiled LSTM model.
        """
        clear_session()

        model = Sequential()
        model.add(LSTM(params['lstm_units'],
                       input_shape=(self.X_train.shape[1], self.X_train.shape[2]),
                       return_sequences=(params['num_lstm_layers'] > 1)))
        model.add(Dropout(params['dropout_rate']))

        for i in range(1, params['num_lstm_layers']):
            model.add(LSTM(params['lstm_units'], return_sequences=(i < params['num_lstm_layers'] - 1)))
            model.add(Dropout(params['dropout_rate']))

        model.add(Dense(1))

        model.compile(
            optimizer=Adam(learning_rate=params['learning_rate']),
            loss='mse',
            metrics=['mae', RootMeanSquaredError()]
        )

        return model

    def objective(self, trial):
        """
        Objective function for Optuna hyperparameter optimization.
        Args:
            trial: The Optuna trial object used for hyperparameter optimization.
        Returns:
            val_loss: Validation loss of the model.
        """
        params = {
            "batch_size": trial.suggest_int("batch_size", 64, 128, step=32),
            "lstm_units": trial.suggest_int("lstm_units", 32, 64),
            "dropout_rate": trial.suggest_float("dropout_rate", 0.1, 0.3),
            "learning_rate": trial.suggest_categorical("learning_rate", [1e-4, 1e-3, 1e-2]),
            "num_lstm_layers": trial.suggest_int("num_lstm_layers", 1, 2),
        }

        model = self.create_model_architecture(params)

        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        pruning_callback = TFKerasPruningCallback(trial, monitor='val_loss')

        history = model.fit(
            self.X_train, self.y_train,
            validation_split=self.validation_split,
            epochs=self.epochs,
            batch_size=params["batch_size"],
            callbacks=[early_stopping, pruning_callback],
            verbose=1
        )

        return history.history['val_loss'][-1]

    def optimize_hyperparameters(self, storage, n_trials=25, study_name="lstm_optuna_study"):
        """
        Optimize hyperparameters using Optuna.

        Args:
            storage (str): Database URL for storing the study.
            n_trials (int): Number of trials to run.
            study_name (str): Name of the Optuna study.
            seed (int): Random seed for reproducibility.

        Returns:
            optuna.study.Study: The Optuna study object containing the results.
        """
        clear_session()

        try:
            self.study = optuna.load_study(study_name=study_name, storage=storage)
            self.logger.info("Loaded existing study '%s'.", study_name)
        except KeyError:
            self.study = optuna.create_study(
                study_name=study_name,
                direction="minimize",
                storage=storage,
                pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
            )
            self.logger.info("Created new study '%s'.", study_name)

        self.study.optimize(self.objective, n_trials=n_trials)
        self.best_params = self.study.best_params

        return self.study

    def plot_and_save(self, plot_func, name, saving_path=None):
        """
        Generate and display a plot using the provided Optuna plotting function.
        Optionally saves the plot to a PNG file.

        Args:
            plot_func (callable): Optuna plotting function (e.g., plot_param_importances).
            name (str): Name to use for the saved plot file.
            saving_path (str or None): If provided, saves the plot to this directory.
        """
        try:
            self.logger.info("Plotting %s...", name.replace("_", " "))
            fig = plot_func(self.study)
            fig.show()

            # Save figure if a path is provided
            if saving_path:
                os.makedirs(saving_path, exist_ok=True)
                fig_path = os.path.join(saving_path, f"{name}.png")
                fig.figure.savefig(fig_path)
                self.logger.info("Saved %s plot to %s", name.replace("_", " "), fig_path)

        except Exception as e:
            self.logger.warning("Error generating %s plot: %s", name.replace("_", " "), str(e))


    def show_result(self, top_n=5, saving_path=None):
        """
        Display statistics and plots from the Optuna study results.
        Optionally saves the plots to disk.

        Args:
            top_n (int): Number of top trials to display in the logs.
            saving_path (str or None): Path to save generated plots. If None, only displays them.
        """
        self.logger.info("Study statistics:")
        self.logger.info("  Number of finished trials: %d", len(self.study.trials))
        self.logger.info("  Best trial: %d", self.study.best_trial.number)

        # Log details about the best trial
        best_trial = self.study.best_trial
        self.logger.info("\nBest Trial:")
        self.logger.info("  Value (Validation Loss): %.4f", best_trial.value)
        self.logger.info("  Hyperparameters:")
        for key, value in best_trial.params.items():
            self.logger.info("    %s: %s", key, value)

        # Log details about the top N trials
        self.logger.info("\nTop %d Trials:", top_n)
        for i, trial in enumerate(sorted(self.study.trials, key=lambda t: t.value)[:top_n]):
            self.logger.info("  Trial %d:", i + 1)
            self.logger.info("    Value: %.4f", trial.value)
            for key, value in trial.params.items():
                self.logger.info("    %s: %s", key, value)

        # Generate and optionally save visualizations
        self.plot_and_save(plot_param_importances, "param_importances", saving_path)
        self.plot_and_save(plot_optimization_history, "optimization_history", saving_path)
        self.plot_and_save(plot_parallel_coordinate, "parallel_coordinates", saving_path)      