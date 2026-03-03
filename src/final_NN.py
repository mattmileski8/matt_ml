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

class LrChangePrinter(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        if not hasattr(self, "prev_lr"):
            self.prev_lr = lr
        if lr != self.prev_lr:
            print(f"\n Learning rate changed from {self.prev_lr:.5f} → {lr:.5f} at epoch {epoch+1}")
            self.prev_lr = lr


SEED = 42

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



# Separate features (X) and target (y)
X = df.drop(columns=["Breakdown Voltage"])  # all columns except target
y = df["Breakdown Voltage"]                 # target column

# scaler_X = StandardScaler()
# X_scaled = scaler_X.fit_transform(X)

# scaler_y = StandardScaler()
# y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))

# #joblib.dump(scaler_X, "./models/final_NN_scaler_X.pkl")
# #joblib.dump(scaler_y, "./models/final_NN_scaler_y.pkl")

# X_input = np.array(X_scaled)
# y_input = np.array(y_scaled)


# early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)

# log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
# # tensorboard --logdir logs/fit

# numerical_input = keras.layers.Input(shape=(X_input.shape[1],))
# hidden1 = keras.layers.Dense(16, activation='swish')(numerical_input)
# hidden1 = keras.layers.Dropout(0.1)(hidden1)
# hidden2 = keras.layers.Dense(16, activation='swish')(hidden1)
# hidden2 = keras.layers.Dropout(0.1)(hidden2)
# concat = keras.layers.Concatenate()([numerical_input,hidden2])
# output = keras.layers.Dense(1)(concat)
# model = keras.Model(inputs=numerical_input, outputs=output)


# model.compile(
#     optimizer=keras.optimizers.Adam(learning_rate=0.006807936096668156), 
#     loss='mean_squared_error',
#     metrics=['mae']
# )
# #learning_rate=0.006807936096668156

# hist1 = model.fit(
#     X_input, y_input,
#     epochs=5600,
#     batch_size=4,
#     validation_split=0.15,
#     callbacks=[tensorboard_callback], #early_stopping, lr_schedule, LrChangePrinter()],
#     verbose=1
#     #shuffle=False
# )

# model.save("models/final_NN_model_concat_layer.keras")

# print("Best Stage 1 val_loss:", min(hist1.history['val_loss']))


# -------------------------- K-fold validation ------------------------------------------------------

model = keras.models.load_model("./models/final_NN_model.keras")




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