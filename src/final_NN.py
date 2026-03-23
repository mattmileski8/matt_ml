import pandas as pd
import numpy as np
import re
import datetime
import os
import subprocess
from pathlib import Path
import joblib
import seaborn as sns
import random
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
from sklearn.metrics import r2_score

# class LrChangePrinter(keras.callbacks.Callback):
#     def on_epoch_end(self, epoch, logs=None):
#         logs = logs or {}
#         lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
#         if not hasattr(self, "prev_lr"):
#             self.prev_lr = lr
#         if lr != self.prev_lr:
#             print(f"\n Learning rate changed from {self.prev_lr:.5f} → {lr:.5f} at epoch {epoch+1}")
#             self.prev_lr = lr


# #SEED = 3
RMSE_array = []
seed_array = []
best_epoch_array = []
external_rmse_array = []
r2_train_array = []
r2_external_test_array = []

for i in range(150):
    SEED = i

    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

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

    # verify
    print(df.head())
    print(df.dtypes)

    # --- NEW: Split 10% for testing ---
    n_samples = len(df)
    train_val_indices, test_indices = train_test_split(
        np.arange(n_samples), test_size=0.1, random_state=SEED
    )

    # Train+val data (90%)
    X_train_val = df.iloc[train_val_indices].drop(columns=["Breakdown Voltage", "Dipole Moment"])
    y_train_val = df.iloc[train_val_indices]["Breakdown Voltage"]

    # Test data (10%)
    X_test = df.iloc[test_indices].drop(columns=["Breakdown Voltage", "Dipole Moment"])
    y_test = df.iloc[test_indices]["Breakdown Voltage"]

    # --- Scale only on train+val data ---
    scaler_X = StandardScaler()
    X_train_val_scaled = scaler_X.fit_transform(X_train_val)

    scaler_y = StandardScaler()
    y_train_val_scaled = scaler_y.fit_transform(y_train_val.values.reshape(-1, 1))

    X_input = np.array(X_train_val_scaled)
    y_input = np.array(y_train_val_scaled)

    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3500, restore_best_weights=True)

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    # tensorboard --logdir logs/fit

    numerical_input = keras.layers.Input(shape=(X_input.shape[1],))
    hidden1 = keras.layers.Dense(16, activation='swish')(numerical_input)
    hidden1 = keras.layers.Dropout(0.1)(hidden1)
    hidden2 = keras.layers.Dense(16, activation='swish')(hidden1)
    hidden2 = keras.layers.Dropout(0.1)(hidden2)
    #concat = keras.layers.Concatenate()([numerical_input,hidden2])
    output = keras.layers.Dense(1)(hidden2)
    model = keras.Model(inputs=numerical_input, outputs=output)


    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.006807936096668156), 
        loss='mean_squared_error',
        metrics=['mae']
    )
    #learning_rate=0.006807936096668156

    hist1 = model.fit(
        X_input, y_input,
        epochs=10000,
        batch_size=4,
        validation_split=0.15,  # Now ~13.5% of total data for validation
        callbacks=[early_stopping],#tensorboard_callback],  etc.
        verbose=1
    )

    # --- NEW: Evaluate on internal test set ---
    X_test_scaled = scaler_X.transform(X_test)
    y_pred_scaled = model.predict(X_test_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled).flatten()
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    RMSE_array.append(rmse)


    r2_test = r2_score(y_test, y_pred)

    print(f"Loop {i+1} Test RMSE: {rmse:.4f}")

    seed_array.append(SEED)
    best_epoch_array.append(np.argmin(hist1.history['val_mae']) + 1)


    # ------External test set ---------
    # Load test data
    df_test = pd.read_csv("./data/test_seven_sorted.txt", sep="\t")
    #df_test_names = pd.read_csv("./data/test_seven_names_sorted.txt", sep="\t")

    # Separate features and target
    X_test_external = df_test.drop(columns=["Breakdown Voltage", "Dipole Moment"])
    y_test_external = df_test["Breakdown Voltage"]

    # Import scalers used in training
    X_test_external_scaled = scaler_X.transform(X_test_external)

    # Predict test set
    y_pred_external_scaled = model.predict(X_test_external_scaled)

    # Convert predictions back into relative DS units
    y_pred_external = scaler_y.inverse_transform(y_pred_external_scaled)
    ypred_external = y_pred_external.flatten()

    # Calculate Test RMSE
    external_rmse = np.sqrt(mean_squared_error(y_test_external, y_pred_external))
    external_rmse_array.append(external_rmse)
    #print(f"Test RMSE: {rmse:.4f}")

    # df_train_check = pd.read_csv("./data/molecular_data_sorted.txt", sep="\t")
    # X_train_check = df_train_check.drop(columns=["Breakdown Voltage", "Dipole Moment"])
    # y_train_check = df_train_check["Breakdown Voltage"]

    # X_train_check_scaled = scaler_X.transform(X_train_check)
    # y_train_pred_scaled = model.predict(X_train_check_scaled)
    # y_train_pred = scaler_y.inverse_transform(y_train_pred_scaled).flatten()

    # R² scores
    
    r2_external_test = r2_score(y_test_external, y_pred_external)

    r2_train_array.append(r2_test)
    r2_external_test_array.append(r2_external_test)


# --- NEW: After the loop, save RMSE_array ---
rmse_df = pd.DataFrame({"Seed": seed_array, 
                        "Test_RMSE": RMSE_array,
                        "External_Test_RMSE": external_rmse_array,
                        "R2_Train": r2_train_array,
                        "R2_External_Test": r2_external_test_array,
                        "Best_Epoch": best_epoch_array})
rmse_df.to_csv("./results/nn_test_rmse_per_loop_7.csv", index=False)
print("Saved test RMSE per loop to ./results/nn_test_rmse_per_loop_7.csv")



    # ---- Save training history to .csv ---------------
    # history_dict = hist1.history

    # # Convert to DataFrame
    # history_df = pd.DataFrame(history_dict)

    # # Add epoch column
    # history_df['epoch'] = np.arange(1, len(history_df) + 1)

    # # Save to CSV
    # history_df.to_csv("./data/training_histories/5400_training_3_history.csv", index=False)
    # -------------------------------------


#model.save("models/final_NN_model.keras")

#print("Best Stage 1 val_mae:", min(hist1.history['val_mae']))



# ------------- Train Final Model -------------------------------------


# SEED = 93

# random.seed(SEED)
# np.random.seed(SEED)
# tf.random.set_seed(SEED)

# # Define the column names based on the header
# columns = [
#     "Molecule",
#     "Vibrational ZPE",
#     "Polarizability",
#     "Dipole Moment",
#     "Adiabatic IE",
#     "Cohesive Energy",
#     "Breakdown Voltage", 
#     "Molecular Mass",
#     "Number e-",
#     "Molecular Volume"
# ]

# # Load the dataframe saved from preprocessing
# df = pd.read_csv("./data/molecular_data_sorted.txt", sep="\t")
# df_names = pd.read_csv("./data/molecular_names_sorted.txt", sep="\t")

# # verify
# print(df.head())
# print(df.dtypes)

# # --- NEW: Split 10% for testing ---
# n_samples = len(df)
# train_val_indices, test_indices = train_test_split(
#     np.arange(n_samples), test_size=0.1, random_state=SEED
# )

# # Train+val data (90%)
# X_train_val = df.iloc[train_val_indices].drop(columns=["Breakdown Voltage", "Dipole Moment"])
# y_train_val = df.iloc[train_val_indices]["Breakdown Voltage"]

# # Test data (10%)
# X_test = df.iloc[test_indices].drop(columns=["Breakdown Voltage", "Dipole Moment"])
# y_test = df.iloc[test_indices]["Breakdown Voltage"]

# # --- Scale only on train+val data ---
# scaler_X = StandardScaler()
# X_train_val_scaled = scaler_X.fit_transform(X_train_val)

# scaler_y = StandardScaler()
# y_train_val_scaled = scaler_y.fit_transform(y_train_val.values.reshape(-1, 1))

# X_input = np.array(X_train_val_scaled)
# y_input = np.array(y_train_val_scaled)

# early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3500, restore_best_weights=True)

# log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
# # tensorboard --logdir logs/fit

# numerical_input = keras.layers.Input(shape=(X_input.shape[1],))
# hidden1 = keras.layers.Dense(16, activation='swish')(numerical_input)
# hidden1 = keras.layers.Dropout(0.1)(hidden1)
# hidden2 = keras.layers.Dense(16, activation='swish')(hidden1)
# hidden2 = keras.layers.Dropout(0.1)(hidden2)
# #concat = keras.layers.Concatenate()([numerical_input,hidden2])
# output = keras.layers.Dense(1)(hidden2)
# model = keras.Model(inputs=numerical_input, outputs=output)


# model.compile(
#     optimizer=keras.optimizers.Adam(learning_rate=0.006807936096668156), 
#     loss='mean_squared_error',
#     metrics=['mae']
# )
# #learning_rate=0.006807936096668156

# hist1 = model.fit(
#     X_input, y_input,
#     epochs=10000,
#     batch_size=4,
#     validation_split=0.15,  # Now ~13.5% of total data for validation
#     #callbacks=[early_stopping],#tensorboard_callback],  etc.
#     verbose=1
# )

# # --- NEW: Evaluate on internal test set ---
# X_test_scaled = scaler_X.transform(X_test)
# y_pred_scaled = model.predict(X_test_scaled)
# y_pred = scaler_y.inverse_transform(y_pred_scaled).flatten()
# rmse = np.sqrt(mean_squared_error(y_test, y_pred))
# print(f"Final Model Test RMSE: {rmse:.4f}")

# model.save("./models/seven_descriptors/nn_avg_model.keras")
# joblib.dump(scaler_X, "./models/seven_descriptors/nn_avg_scaler_X.pkl")
# joblib.dump(scaler_y, "./models/seven_descriptors/nn_avg_scaler_y.pkl")







# -------------------------- K-fold validation ------------------------------------------------------
# # Define model builder
# def build_model(input_dim):
#     numerical_input = keras.layers.Input(shape=(input_dim,))
#     hidden1 = keras.layers.Dense(16, activation='swish')(numerical_input)
#     hidden1 = keras.layers.Dropout(0.1)(hidden1)
#     hidden2 = keras.layers.Dense(16, activation='swish')(hidden1)
#     hidden2 = keras.layers.Dropout(0.1)(hidden2)
#     #concat = keras.layers.Concatenate()([numerical_input, hidden2])
#     output = keras.layers.Dense(1)(hidden2)

#     model = keras.Model(inputs=numerical_input, outputs=output)

#     model.compile(
#         optimizer=keras.optimizers.Adam(learning_rate=0.006807936096668156),
#         loss='mean_squared_error'
#     )

#     return model

# kf = KFold(n_splits=7, shuffle=True, random_state=SEED)

# rmse_scores = []
# best_epochs = []



# for fold, (train_idx, val_idx) in enumerate(kf.split(X)):

#     print(f"\n===== Fold {fold+1} =====")

#     # Split raw data
#     X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
#     y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

#     # Fit scalers ONLY on training data (prevents leakage)
#     scaler_X = StandardScaler()
#     scaler_y = StandardScaler()

#     X_train_scaled = scaler_X.fit_transform(X_train)
#     X_val_scaled = scaler_X.transform(X_val)

#     y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1,1))
#     y_val_scaled = scaler_y.transform(y_val.values.reshape(-1,1))

#     # Build fresh model
#     model = build_model(X_train_scaled.shape[1])

#     early_stopping = keras.callbacks.EarlyStopping(
#         monitor='val_loss',
#         patience=1500,
#         restore_best_weights=True
#     )

#     history = model.fit(
#         X_train_scaled,
#         y_train_scaled,
#         validation_data=(X_val_scaled, y_val_scaled),
#         epochs=5400,
#         batch_size=4,
#         #callbacks=[early_stopping],
#         verbose=0
#     )

#     # Record best epoch
#     best_epoch = np.argmin(history.history['val_loss']) + 1
#     best_epochs.append(best_epoch)

#     # Predict
#     y_pred_scaled = model.predict(X_val_scaled)
#     y_pred = scaler_y.inverse_transform(y_pred_scaled)

#     # Compute RMSE in original voltage units
#     rmse = np.sqrt(mean_squared_error(y_val, y_pred))
#     rmse_scores.append(rmse)

#     print(f"Fold {fold+1} RMSE: {rmse:.4f}")
#     print(f"Best epoch: {best_epoch}")

# print("\n===== Cross-Validation Results =====")
# print("Average RMSE:", np.mean(rmse_scores))
# print("Std RMSE:", np.std(rmse_scores))
# print("Average Best Epoch:", int(np.mean(best_epochs)))

# -----------------Test Model ---------------------------------------------------------

# # Load trained model
# model = keras.models.load_model("models/final_NN_model.keras")

# # Load saved scalers
# scaler_X = joblib.load("models/final_NN_scaler_X.pkl")
# scaler_y = joblib.load("models/final_NN_scaler_y.pkl")


# # Load test data
# df_test = pd.read_csv("./data/test_seven_sorted.txt", sep="\t")
# df_test_names = pd.read_csv("./data/test_seven_names_sorted.txt", sep="\t")

# # Load prediction dataset
# df_pred = pd.read_csv("./data/molecular_tm_data_sorted.txt", sep="\t")
# df_pred_names = pd.read_csv("./data/molecular_tm_names_sorted.txt", sep="\t")


# # Separate features and target
# X_test = df_test.drop(columns=["Breakdown Voltage"])
# y_test = df_test["Breakdown Voltage"]

# # Import scalers used in training
# X_test_scaled = scaler_X.transform(X_test)

# # Predict test set
# y_pred_scaled = model.predict(X_test_scaled)

# # Convert predictions back into relative DS units
# y_pred = scaler_y.inverse_transform(y_pred_scaled)
# ypred = y_pred.flatten()

# # Calculate Test RMSE
# rmse = np.sqrt(mean_squared_error(y_test, y_pred))
# print(f"Test RMSE: {rmse:.4f}")

# # Separate training features/target again
# X_train = df.drop(columns=["Breakdown Voltage"])
# y_train = df["Breakdown Voltage"]

# # Scale using saved scalers
# X_train_scaled = scaler_X.transform(X_train)

# # Predict (scaled)
# y_train_pred_scaled = model.predict(X_train_scaled)

# # Convert back to real units
# y_train_pred = scaler_y.inverse_transform(y_train_pred_scaled).flatten()

# # R² scores
# r2_train = r2_score(y_train, y_train_pred)
# r2_test = r2_score(y_test, y_pred)

# # Parity plot
# plt.figure(figsize=(6,6))

# # Plot training
# plt.scatter(y_train, y_train_pred, alpha=0.6, label=f"Train (R² = {r2_train:.3f})")

# # Plot test
# plt.scatter(y_test, y_pred, alpha=0.9, label=f"Test (R² = {r2_test:.3f})")

# plt.plot(
#     [min(y_train), max(y_test)],
#     [min(y_train), max(y_test)],
# )  # 45-degree line

# plt.xlabel("Actual Breakdown Voltage")
# plt.ylabel("Predicted Breakdown Voltage")
# plt.title("Parity Plot: Training vs Test")
# plt.legend()
# #plt.gca().set_aspect('equal', adjustable='box')
# plt.tight_layout()
# plt.savefig("./images/test_parity.png")

#------------------------------------------------------------------------------------------
# model = keras.models.load_model('./models/final_NN.keras')


# # --- Scale features the same way as during training ---
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# # Load the label scaler that was used during training
# # (You need to re-fit it the same way)
# scaler_label = StandardScaler()
# scaler_label.fit(y.values.reshape(-1, 1))

# # Predict scaled values
# preds_scaled = model.predict(X_scaled)

# # Inverse transform to get real breakdown voltages
# preds = scaler_label.inverse_transform(preds_scaled).flatten()
# y_actual = y.values.flatten()


# from sklearn.metrics import r2_score


# # Fit a regression line
# slope, intercept = np.polyfit(y_actual, preds, 1)
# y_fit = slope * y_actual + intercept

# # Compute R^2
# r2 = r2_score(y_actual, preds)

# # Plot
# plt.figure(figsize=(3.5,3.5))
# plt.scatter(y_actual, preds, alpha=0.7, label="Predictions")
# plt.plot(y_actual, y_actual, color='red', linestyle='-', 
#          label=f"Fit: y={slope:.2f}x + {intercept:.2f}\n$R^2$={r2:.2f}")

# plt.xlabel("Actual Breakdown Field (MV/m)", fontsize=8.5, fontweight='bold')
# plt.ylabel("Predicted Breakdown Field (MV/m)", fontsize=8.5, fontweight='bold')
# plt.xticks(fontsize=8.5)
# plt.yticks(fontsize=8.5)
# #plt.title("RF: Actual vs Predicted Electric Field at Breakdown")
# #plt.legend(fontsize=8.5, labelspacing=0.2, loc="upper left")
# plt.grid(True)

# plt.text(
#     0.05, 0.95,                   # (x, y) position in axes coordinates
#     f"$R^2 = {r2:.3f}$",
#     transform=plt.gca().transAxes,
#     fontsize=8.5,
#     fontweight='bold',
#     verticalalignment='top',
# )

# # marks outlier
# plt.text(
#     0.07, 0.48,                   # (x, y) position in axes coordinates
#     "O",
#     transform=plt.gca().transAxes,
#     fontsize=8.5,
#     #fontweight='bold',
#     verticalalignment='top',
# )

# # Save figure
# os.makedirs("./images", exist_ok=True)
# plt.savefig("./images/final_actual_vs_predicted_fitline.png", dpi=300, bbox_inches="tight")
# plt.close()

# residuals = y_actual - preds
# z = np.abs((residuals - np.mean(residuals)) / np.std(residuals))

# outliers = z > 3
# #print(outliers)

# for i in range(len(outliers)):
#     if outliers[i] == True: 
#         print(names[i])
#         print(y_actual[i])
#         print(preds[i])