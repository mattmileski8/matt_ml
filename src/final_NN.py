import pandas as pd
import numpy as np
import re
import datetime
import os
import subprocess
from pathlib import Path
import joblib
import seaborn as sns

import matplotlib.pyplot as plt
import statsmodels.api as sm
import tensorflow as tf
import keras 
from scipy import stats
from sklearn.metrics import mean_squared_error
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class LrChangePrinter(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        if not hasattr(self, "prev_lr"):
            self.prev_lr = lr
        if lr != self.prev_lr:
            print(f"\n Learning rate changed from {self.prev_lr:.5f} → {lr:.5f} at epoch {epoch+1}")
            self.prev_lr = lr


# Define the column names based on the header
columns = [
    "Molecule",
    "Vibrational ZPE (cm^-1)",
    "Polarizability (Å^3)",
    "Dipole Moment (Debye)",
    "Adiabatic IE (eV)",
    "Cohesive Energy (kJ/mol)",
    "Breakdown Voltage (MV/m)"
]

# Load the file
with open('./data/molecular_data.txt', "r") as file:
    lines = file.readlines()

data = []

for line in lines:
    # Remove leading/trailing whitespace
    line = line.strip()
    if not line or line.startswith("Molecule"):
        continue  # Skip empty and header lines

    # Match molecule name (non-numeric part at the start)
    match = re.match(r'^(\S+)', line)
    if match:
        molecule = match.group(1)
        # Extract all numbers (scientific notation or float)
        values = re.findall(r'[-+]?\d*\.\d+e[+-]?\d+|[-+]?\d+\.\d+|[-+]?\d+', line[len(molecule):])
        # Fill missing values with None (so all rows have 6 columns)
        while len(values) < 6:
            values.append(None)
        data.append([molecule] + values)

# Convert to DataFrame
df = pd.DataFrame(data, columns=columns)
df = df.replace("", pd.NA).dropna()

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
print(len(df))

names = df["Molecule"]

df = df.drop(columns=["Molecule"])

# Convert numeric columns to float
for col in columns[1:]:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Preview the result
print(df)

#-----------------------------------------------------------------------------
# Compute Pearson correlations between each feature and target and full matrix
#-----------------------------------------------------------------------------

clean_columns = [
    r"$\varepsilon_{V}$",   # Vibrational ZPE 
    r"$\alpha$",                   # Polarizability
    r"$\mu$",                      # Dipole Moment
    r"$\varepsilon_{I}$",              # Adiabatic IE
    r"$\varepsilon_{c}$",            # Cohesive Energy
    "DS"
]

df.columns = clean_columns

feature_names = clean_columns[0:-1]   # all input features
target_col = clean_columns[-1]        # breakdown strength, not used right now

# correlations = df.corr(method='pearson')[[target_col]].loc[feature_names]

# print("\n Pearson Correlation (r) with Breakdown Strength:")
# print(correlations)

# corr_matrix = df.corr(method='pearson')


# select only the five feature columns
feature_df = df[feature_names]

# Pearson correlation matrix of features only
corr_matrix = feature_df.corr(method='pearson')

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", square=True, annot_kws={"size": 18})
plt.xticks(rotation=0, ha='right', fontsize=18)
plt.yticks(rotation=0, ha='right', fontsize=18)
# plt.xlabel(fontsize=8.5)
# plt.ylabel(fontsize=8.5)
#plt.title("Pearson Correlation Matrix")

cbar = plt.gcf().axes[-1]
cbar.tick_params(labelsize=18)

plt.tight_layout()
plt.savefig(f"./results/pearson_correlation_heatmap.png", dpi=300)
plt.close()


#------------------------------------------------------------------------------------



# # Separate features (X) and target (y)
# X = df.drop(columns=["Breakdown Voltage (MV/m)"])  # all columns except target
# y = df["Breakdown Voltage (MV/m)"]                 # target column

# scaler_X = StandardScaler()
# X_scaled = scaler_X.fit_transform(X)

# scaler_y = StandardScaler()
# y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))

# X_input = np.array(X_scaled)
# y_input = np.array(y_scaled)


# early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)

# #log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# #tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# numerical_input = keras.layers.Input(shape=(X_input.shape[1],))
# hidden1 = keras.layers.Dense(16, activation='relu')(numerical_input)
# hidden1 = keras.layers.Dropout(0.1)(hidden1)
# hidden2 = keras.layers.Dense(16, activation='relu')(hidden1)
# hidden2 = keras.layers.Dropout(0.1)(hidden2)
# #concat = keras.layers.Concatenate()([numerical_input,hidden3])
# output = keras.layers.Dense(1)(hidden2)
# model = keras.Model(inputs=numerical_input, outputs=output)

# # ---- Stage 1: Train with validation + LR scheduling ----


# model.compile(
#     optimizer=keras.optimizers.Adam(learning_rate=1e-2),  # Stage 1 LR
#     loss='mean_squared_error',
#     metrics=['mae']
# )

# lr_schedule = keras.callbacks.ReduceLROnPlateau(
#     monitor='val_loss',
#     factor=0.3,
#     patience=15,
#     min_lr=1e-5,
#     verbose=1
# )

# hist1 = model.fit(
#     X_input, y_input,
#     epochs=2000,
#     batch_size=8,
#     validation_split=0.2,
#     callbacks=[early_stopping, lr_schedule, LrChangePrinter()],
#     verbose=1
# )

# print("Best Stage 1 val_loss:", min(hist1.history['val_loss']))


# # ---- Stage 2: Fine-tune using all data (no validation) ----

# model.optimizer.learning_rate = 1e-4  # Lower LR for fine-tuning

# hist2 = model.fit(
#     X_input, y_input,
#     epochs=200,
#     batch_size=8,
#     verbose=1
# )

# print("Stage 2 final loss:", hist2.history['loss'][-1])


# # ---- Save model and scalers ----

# os.makedirs("./models", exist_ok=True)
# os.makedirs("./logs", exist_ok=True)

# model.save('./models/final_NN.keras')
# joblib.dump(scaler_X, './models/final_scaler_X.pkl')
# joblib.dump(scaler_y, './models/final_scaler_y.pkl')

# # ---- Save combined training history ----

# stage1_len = len(hist1.history['loss'])
# stage2_len = len(hist2.history['loss'])

# history_df = pd.DataFrame({
#     "stage1_loss": hist1.history['loss'] + [None]*stage2_len,
#     "stage1_val_loss": hist1.history['val_loss'] + [None]*stage2_len,
#     "stage2_loss": [None]*stage1_len + hist2.history['loss']
# })

# history_df.to_csv("./logs/final_NN_training_history.csv", index=False)




# print(results_df)

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