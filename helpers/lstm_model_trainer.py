"""
This module provides helper functions for creating and training LSTM models
with hyperparameter optimization using Optuna.
"""
import optuna
from optuna.integration import TFKerasPruningCallback
from optuna.trial import TrialState

from keras.backend import clear_session
from keras.callbacks import EarlyStopping
from keras.layers import LSTM, Dense, Dropout
from keras.metrics import RootMeanSquaredError
from keras.models import Sequential
from keras.optimizers import Adam

from .data_preparator import DataPreparator


class LSTMModelTrainer(DataPreparator):
    """
    A class that extends DataPreparator to include methods for creating, training, and optimizing an LSTM model.
    """
    def create_model(self, trial):
        """
        Create and train an LSTM model using the given hyperparameters.
        Args:
            trial: The Optuna trial object used for hyperparameter optimization.
        """
        # Clear the session to avoid memory issues
        clear_session()

        # Define the hyperparameters to be optimized
        batchsize = trial.suggest_int("batchsize", 64, 128, step=32)
        lstm_units = trial.suggest_int("lstm_units", 32, 64)
        dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.3)
        learning_rate = trial.suggest_categorical("learning_rate", [1e-4, 1e-3, 1e-2])
        num_lstm_layers = trial.suggest_int('num_lstm_layers', 1, 2)

        # Define the model architecture using the suggested hyperparameters
        model = Sequential()
        model.add(LSTM(lstm_units, input_shape=(self.X_train.shape[1], self.X_train.shape[2]), return_sequences=(num_lstm_layers > 1)))
        model.add(Dropout(dropout_rate))

        for _ in range(num_lstm_layers - 1):
            model.add(LSTM(lstm_units, return_sequences=False))
            model.add(Dropout(dropout_rate))

        model.add(Dense(1))  # Output layer

        # Compile the model
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse', metrics=['mae', RootMeanSquaredError(), 'mape'])

        # Define callbacks for early stopping and pruning
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        pruning_callback = TFKerasPruningCallback(trial, monitor='val_loss')

        # Train the model
        history = model.fit(
            self.X_train, self.y_train,
            validation_split=0.2,
            epochs=30,
            batch_size=batchsize,
            callbacks=[early_stopping, pruning_callback],
            verbose=1
        )

        # Return the validation loss for optimization
        val_loss = history.history['val_loss'][-1]
        return val_loss

    def optimize_hyperparameters(self, n_trials=25, timeout=600):
        """
        Optimize hyperparameters using Optuna.
        Args:
            n_trials: Number of trials for optimization.
            timeout: Timeout for the optimization process.
        """
        def objective(trial):
            return self.create_model(trial)

        study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner(n_startup_trials=2))
        study.optimize(objective, n_trials=n_trials, timeout=timeout)
        #self.best_params = study.best_params
        return study

    def show_result(self, study):
        pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
        complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

        print("Study statistics: ")
        print("  Number of finished trials: ", len(study.trials))
        print("  Number of pruned trials: ", len(pruned_trials))
        print("  Number of complete trials: ", len(complete_trials))

        print("Best trial:")
        trial = study.best_trial

        print("  Value: ", trial.value)

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))
