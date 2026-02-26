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

def objective(trial):

    # ----- Hyperparameters to Search -----
    n_layers = trial.suggest_int("n_layers", 1, 3)
    n_units = trial.suggest_int("n_units", 4, 32, step=4)
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True)
    batch_size = trial.suggest_categorical("batch_size", [4, 8, 16])
    activation = trial.suggest_categorical("activation", ["relu", "swish", "leaky_relu"])
    optimizer_name = trial.suggest_categorical("optimizer", ["adam", "rmsprop", "sgd"])

    # ----- Choose optimizer -----
    if optimizer_name == "adam":
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_name == "rmsprop":
        optimizer = keras.optimizers.RMSprop(learning_rate=learning_rate)
    else:
        optimizer = keras.optimizers.SGD(learning_rate=learning_rate)

    # ----- 5-Fold Cross Validation -----
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    val_losses = []

    for train_idx, val_idx in kf.split(X_input):

        X_train, X_val = X_input[train_idx], X_input[val_idx]
        y_train, y_val = y_input[train_idx], y_input[val_idx]

        # Build model
        model = keras.Sequential()
        model.add(keras.layers.Input(shape=(X_input.shape[1],)))

        for _ in range(n_layers):
            model.add(keras.layers.Dense(n_units, activation=activation))

        model.add(keras.layers.Dense(1))

        model.compile(
            optimizer=optimizer,
            loss="mse"
        )

        early_stopping = keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=40,
            restore_best_weights=True,
            verbose=0
        )

        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=500,
            batch_size=batch_size,
            verbose=0,
            callbacks=[early_stopping]
        )

        val_losses.append(min(history.history["val_loss"]))

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
print(df.head())
print(df.dtypes)



# Separate features (X) and target (y)
X = df.drop(columns=["Breakdown Voltage"])  # all columns except target
y = df["Breakdown Voltage"]                 # target column

scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)

scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))

X_input = np.array(X_scaled)
y_input = np.array(y_scaled)



