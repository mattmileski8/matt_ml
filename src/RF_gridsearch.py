import os
import re
import joblib
import numpy as np
import pandas as pd
from itertools import product
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from scipy.stats import linregress
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import make_scorer
import optuna

# Define the Optuna objective function
def objective(trial):
    # Suggest hyperparameters
    n_estimators = trial.suggest_int("n_estimators", 100, 10000)
    max_depth = trial.suggest_int("max_depth", 5, 50)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 20)

    # Create RF model
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
        n_jobs=-1,
        oob_score=True,
        bootstrap=True
    )

    # 10-fold CV
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(
        rf,
        X_train_input,
        y_train_input,
        scoring=neg_mse_scorer,
        cv=kf,
        n_jobs=-1
    )

    # Return mean RMSE (Optuna maximizes by default, so we negate RMSE)
    mean_rmse = np.mean(np.sqrt(-scores))
    return -mean_rmse  # minimize RMSE

# Define a scorer
neg_mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)


# Define the column names based on the header
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



#----------Train initial model
data_train, data_test = train_test_split(df, test_size=0.1, random_state=42)

X_train = data_train.drop(columns=['Breakdown Voltage'])
X_test = data_test.drop(columns=['Breakdown Voltage'])

y_train = data_train[['Breakdown Voltage']]
y_test = data_test[['Breakdown Voltage']]


# Convert to np arrays
X_train_input = np.array(X_train)
X_test_input = np.array(X_test)
y_train_input = np.array(y_train)
y_test_input = np.array(y_test)

# Flatten input for BaggingRegressor
y_train_input = y_train_input.ravel()
y_test_input = y_test_input.ravel()


# Create Optuna study
study = optuna.create_study(direction="minimize",
                            study_name="molecular_rf",
                            storage="sqlite:///molecular_rf_optuna.db",
                            load_if_exists=True
                            )
study.optimize(objective, n_trials=1000, show_progress_bar=True)

# Best hyperparameters
print("Best hyperparameters:", study.best_params)

# Train final model with best hyperparameters
best_params = study.best_params
rf_best = RandomForestRegressor(
    n_estimators=best_params["n_estimators"],
    max_depth=best_params["max_depth"],
    min_samples_split=best_params["min_samples_split"],
    min_samples_leaf=best_params["min_samples_leaf"],
    random_state=42,
    n_jobs=-1,
    oob_score=True,
    bootstrap=True
)

rf_best.fit(X_train_input, y_train_input)

# Evaluate on test set
y_pred = rf_best.predict(X_test_input)
test_rmse = np.sqrt(mean_squared_error(y_test_input, y_pred))
test_r2 = r2_score(y_test_input, y_pred)

rf_trials = study.trials_dataframe()
rf_trials.to_csv("./logs/optuna_rf_trials.csv", index=False)


print(f"Test RMSE: {test_rmse:.3f}")
print(f"Test R2: {test_r2:.3f}")