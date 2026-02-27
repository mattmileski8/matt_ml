import pandas as pd
import numpy as np
import re
import datetime
import os
import subprocess
from pathlib import Path
import joblib
import seaborn as sns
import optuna
import matplotlib.pyplot as plt
import statsmodels.api as sm
import tensorflow as tf
import keras 
from scipy import stats
from sklearn.metrics import mean_squared_error
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from optuna.integration import TFKerasPruningCallback

def objective(trial):

    # ----- Hyperparameters to Search -----
    n_layers = trial.suggest_int("n_layers", 1, 3)
    n_units = trial.suggest_int("n_units", 4, 16, step=4)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [4, 8, 16])
    activation = trial.suggest_categorical("activation", ["relu", "swish", "leaky_relu"])
    optimizer_name = trial.suggest_categorical("optimizer", ["adam", "rmsprop", "sgd"])
    dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.3, step=0.1)

    # ----- 5-Fold Cross Validation -----
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    val_losses = []

    for train_idx, val_idx in kf.split(X):

        X_train_raw = X.iloc[train_idx].values
        X_val_raw   = X.iloc[val_idx].values
        y_train_raw = y.iloc[train_idx].values
        y_val_raw   = y.iloc[val_idx].values

        scaler_X = StandardScaler()
        X_train = scaler_X.fit_transform(X_train_raw)
        X_val = scaler_X.transform(X_val_raw)

        scaler_y = StandardScaler()
        y_train = scaler_y.fit_transform(y_train_raw.reshape(-1,1))
        y_val = scaler_y.transform(y_val_raw.reshape(-1,1))

        # ---- Build model ----
        model = keras.Sequential()
        model.add(keras.layers.Input(shape=(X_train.shape[1],)))

        for _ in range(n_layers):

            if activation == "leaky_relu":
                model.add(keras.layers.Dense(n_units))
                model.add(keras.layers.LeakyReLU())
            else:
                model.add(keras.layers.Dense(n_units, activation=activation))

            if dropout_rate > 0:
                model.add(keras.layers.Dropout(dropout_rate))

        model.add(keras.layers.Dense(1))

        # ---- Optimizer (recreate each fold!) ----
        if optimizer_name == "adam":
            optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer_name == "rmsprop":
            optimizer = keras.optimizers.RMSprop(learning_rate=learning_rate)
        else:
            optimizer = keras.optimizers.SGD(learning_rate=learning_rate)

        model.compile(
            optimizer=optimizer,
            loss="mse"
        )

        early_stopping = keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=30,
            restore_best_weights=True,
            verbose=0
        )

        #pruning_callback = TFKerasPruningCallback(trial, "val_loss")

        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=400,
            batch_size=batch_size,
            verbose=0,
            callbacks=[early_stopping]#, pruning_callback]
        )

        val_losses.append(min(history.history["val_loss"]))

        keras.backend.clear_session()  # VERY important for memory

    return np.mean(val_losses)

columns = [
    "Molecule",
    "Vibrational ZPE",
    "Polarizability",
    "Dipole Moment",
    "Adiabatic IE",
    "Cohesive Energy",
    "Breakdown Voltage", 
    "Molecular Mass",
    "Number e-",
    "Molecular Volume"
]

# Load the dataframe saved from preprocessing
df = pd.read_csv("./data/molecular_data_sorted.txt", sep="\t")
df_names = pd.read_csv("./data/molecular_names_sorted.txt", sep="\t")

# verify
#print(df.head())
print(df.dtypes)



# Separate features (X) and target (y)
X = df.drop(columns=["Breakdown Voltage"])  # all columns except target
y = df["Breakdown Voltage"]                 # target column

# X_input = np.array(X)
# y_input = np.array(y)



# Optuna parameter search

study = optuna.create_study(
    direction="minimize",
    study_name="molecular_nn",
    storage="sqlite:///molecular_optuna.db",
    load_if_exists=True
    #pruner=optuna.pruners.MedianPruner(n_startup_trials=20, n_warmup_steps=20)
)

study.optimize(objective, n_trials=200)




# study = optuna.load_study(
#     study_name="molecular_nn",
#     storage="sqlite:///molecular_optuna.db"
# )


df_trials = study.trials_dataframe()
df_trials.to_csv("./logs/optuna_trials.csv", index=False)

print(df_trials.head())

print("Best Parameters:")
print(study.best_params)

print("Best CV MSE:")
print(study.best_value)