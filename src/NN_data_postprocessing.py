import os
import re
import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import tensorflow as tf
from tensorflow.keras.models import load_model


# -------Plot validation and training mae-----------------------------------
# history_df_5400 = pd.read_csv("./data/training_histories/5400_training_42_history.csv")
# history_df_10000 = pd.read_csv("./data/training_histories/10000_training_history.csv")

# window = 200

# val_5400_smooth = history_df_5400["val_mae"].rolling(window=window).mean()
# #val_10000_smooth = history_df_10000["val_mae"].rolling(window=window).mean()

# plt.figure()
# #plt.plot(history_df_10000['epoch'], history_df_10000['val_mae'], label='Validation MAE')
# #plt.plot(history_df_10000["epoch"], val_10000_smooth, linewidth=2, label="Validation (smoothed)")
# #plt.plot(history_df_5600['epoch'], history_df_5600['mae'], label='Training MAE')
# plt.plot(history_df_5400['epoch'], history_df_5400['val_mae'], label='Validation MAE')
# plt.plot(history_df_5400["epoch"], val_5400_smooth, linewidth=2, label="Validation (smoothed)")
# plt.xlabel('Epoch')
# plt.ylabel('MAE')
# plt.legend()
# plt.title('Training vs Validation MAE')
# plt.savefig("./results/smoothed_val_mae.png")

# best_epoch = history_df_5600.loc[history_df_5600["val_mae"].idxmin(), "epoch"]
# print("Best epoch (val_mae):", best_epoch)

# min_val_mae = history_df_5600["val_mae"].min()
# print("Minimum val_mae:", min_val_mae)

# print(val_5400_smooth.min())


#----------------------------------------------------------------------------

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

MODEL_PATH = "./models/eight_descriptors/nn_avg_model.keras"
SCALER_X_PATH = "./models/eight_descriptors/nn_avg_scaler_X.pkl"
SCALER_Y_PATH = "./models/eight_descriptors/nn_avg_scaler_y.pkl"
OUTPUT_DIR = "./results/shap_nn_8_descriptors_all"
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv("./data/molecular_data_sorted.txt", sep="\t")
df_names = pd.read_csv("./data/molecular_names_sorted.txt", sep="\t")

df_test = pd.read_csv("./data/test_seven_sorted.txt", sep="\t")
df_test_names = pd.read_csv("./data/test_seven_names_sorted.txt", sep="\t")

df_pred = pd.read_csv("./data/molecular_tm_data_sorted.txt", sep="\t")
df_pred_names = pd.read_csv("./data/molecular_tm_names_sorted.txt", sep="\t")

# ------------ Make predictions and calculate test R² and RMSE -------------
nn_model = load_model(MODEL_PATH)
X_scaler = joblib.load(SCALER_X_PATH)
y_scaler = joblib.load(SCALER_Y_PATH)

feature_names = columns[1:6] + columns[7:]  # 8 input features (excluding DS)
#feature_names = columns[2:6] + columns[7:]  # 7 input features (excluding DS, Vibrational ZPE)
#feature_names = columns[2:6] + columns[7:8] + columns[9:]  # 6 input features (excluding DS, Vibrational ZPE, and # e-)
target_col = ["Breakdown Voltage"]

X_test = df_test[feature_names].values                              # Build X from test set
X_test_scaled = X_scaler.transform(X_test)                          # Scale X using training scaler
y_true_test = df_test[target_col].values.flatten()                  # True y values from test set
preds_scaled_test = nn_model.predict(X_test_scaled)                      # Predict scaled target
preds_scaled_test = np.asarray(preds_scaled_test).reshape(-1, 1)              # Ensure preds_scaled is 2D for inverse transform
y_pred_test = y_scaler.inverse_transform(preds_scaled_test).flatten()    # Inverse transform to real units (rel DS)

test_RMSE = np.sqrt(np.mean((y_pred_test - y_true_test)**2))
test_r2 = r2_score(y_true_test, y_pred_test)

print(f"NN Test RMSE: {test_RMSE:.3f}")
print(f"NN Test R²: {test_r2:.3f}")

stdev_residuals_test = np.std(y_true_test - y_pred_test)

# --------------- Make predictions on training data and plot ------------------------------------
X_train = df[feature_names].values
X_train_scaled = X_scaler.transform(X_train)
y_true_train = df[target_col].values.flatten()
preds_scaled_train = nn_model.predict(X_train_scaled)
preds_scaled_train = np.asarray(preds_scaled_train).reshape(-1, 1)
y_pred_train = y_scaler.inverse_transform(preds_scaled_train).flatten()

train_RMSE = np.sqrt(np.mean((y_pred_train - y_true_train)**2))
train_r2 = r2_score(y_true_train, y_pred_train)

print(f"NN Train RMSE: {train_RMSE:.3f}")
print(f"NN Train R²: {train_r2:.3f}")

stdev_residuals_train = np.std(y_true_train - y_pred_train)

# ----------------------- Parity Plot ----------------------------------------
fig, ax = plt.subplots(figsize=(4, 3.2))

ax.scatter(y_true_train, y_pred_train,  color='steelblue', edgecolors='k', alpha=0.7, label=f'Train (R² = {train_r2:.3f}), $\\sigma$={stdev_residuals_train:.3f}')
ax.scatter(y_true_test, y_pred_test, marker='s', color='orange', edgecolors='k', alpha=0.7, label=f'Test (R² = {test_r2:.3f}), $\\sigma$={stdev_residuals_test:.3f}')
# Plot y=x parity line
min_val = min(y_true_train.min(), y_pred_train.min())
max_val = max(y_true_train.max(), y_pred_train.max())
ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1)#, label='Parity line')

ax.set_xlabel('True Relative DS', fontweight='bold')
ax.set_ylabel('Predicted Relative DS', fontweight='bold')
ax.tick_params(axis='both', labelsize=9)
#ax.set_title(f'Parity Plot')
ax.legend(fontsize=8.6)
plt.tight_layout()
plt.savefig("./images/nn_parity_plot_8_descriptors.png", dpi=300, bbox_inches="tight")



# ----------------------------Make predictions on predict data----------------------------------------------
X_tm = df_pred[feature_names].values
X_tm_scaled = X_scaler.transform(X_tm)
y_pred_tm = nn_model.predict(X_tm_scaled)
y_pred_tm = np.asarray(y_pred_tm).reshape(-1, 1)
y_pred_tm = y_scaler.inverse_transform(y_pred_tm).flatten()

y_pred_tm_series = pd.Series(y_pred_tm, name='Predicted Dielectric Strength')
tm_prediction_dataset = pd.DataFrame(X_tm, columns=feature_names)
tm_prediction_dataset.insert(0, 'Predicted Dielectric Strength', y_pred_tm_series)
tm_prediction_dataset.insert(0, 'Molecule', df_pred_names['Molecule'])

tm_prediction_dataset.to_csv('./results/nn_tm_prediction_dataset.csv', index=False)



# Save training predictions with molecule names and features to .csv for later analysis
y_pred_train_series = pd.Series(y_pred_train, name='y_pred_train')
train_prediction_dataset = pd.DataFrame(X_train, columns=feature_names)
train_prediction_dataset.insert(0, 'Predicted Dielectric Strength', y_pred_train_series)
train_prediction_dataset.insert(0, 'Molecule', df_names['Molecule'])

train_prediction_dataset.to_csv('./results/nn_train_prediction_dataset.csv', index=False)

# Save test predictions with molecule names and features to .csv for later analysis
y_pred_test_series = pd.Series(y_pred_test, name='y_pred_test')
test_prediction_dataset = pd.DataFrame(X_test, columns=feature_names)
test_prediction_dataset.insert(0, 'Predicted Dielectric Strength', y_pred_test_series)
test_prediction_dataset.insert(0, 'Molecule', df_test_names['Molecule'])

test_prediction_dataset.to_csv('./results/nn_test_prediction_dataset.csv', index=False)









# #----------------------------------------------------------------------------
# # feature importance
# #----------------------------------------------------------------------------
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

# MODEL_PATH = "./models/six_descriptors/nn_avg_model.keras"
# SCALER_X_PATH = "./models/six_descriptors/nn_avg_scaler_X.pkl"
# SCALER_Y_PATH = "./models/six_descriptors/nn_avg_scaler_y.pkl"
# OUTPUT_DIR = "./results/shap_nn_6_descriptors_all"
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# # -----------------------------
# # Column names (Molecule + 5 features)
# # -----------------------------
# symbolic_feature_names = [
#     #r"$\varepsilon_{V}$",   # Vibrational ZPE 
#     r"$\alpha$",                   # Polarizability
#     #r"$\mu$",                      # Dipole Moment
#     r"$\varepsilon_{I}$",              # Adiabatic IE
#     r"$\varepsilon_{c}$",            # Cohesive Energy
#     r"$m$",                 # Molecular Mass
#     r"$n_{e}$",             # Number of electrons
#     r"$V$"                  # Molecular Volume
# ]


# feature_names = [
#     #"Vibrational ZPE",
#     "Polarizability",
#     #"Dipole Moment",
#     "Adiabatic IE",
#     "Cohesive Energy",
#     "Molecular Mass",
#     "Number e-",
#     "Molecular Volume"
# ]
# target_col = ["Breakdown Voltage"]



# # -----------------------------
# # Load model, scalers, dataset
# # -----------------------------

# model = load_model(MODEL_PATH)
# X_scaler = joblib.load(SCALER_X_PATH)
# y_scaler = joblib.load(SCALER_Y_PATH)

# df = pd.read_csv("./data/test_seven_sorted.txt", sep="\t")
# df_names = pd.read_csv("./data/test_seven_names_sorted.txt", sep="\t")

# df_training = pd.read_csv("./data/molecular_data_sorted.txt", sep="\t")
# df_training_names = pd.read_csv("./data/molecular_names_sorted.txt", sep="\t")

# # ------------------------------------------------
# # Build X and predict y
# # ------------------------------------------------
# X = df[feature_names].values
# X_scaled = X_scaler.transform(X)

# # Predict scaled target
# preds_scaled = model.predict(X_scaled)
# preds_scaled = np.asarray(preds_scaled).reshape(-1, 1)

# r2_score_value = r2_score(y_scaler.inverse_transform(preds_scaled), df[target_col].values)
# print(f"R² Score on Test Set: {r2_score_value:.4f}")

# # Inverse transform to real units
# preds = y_scaler.inverse_transform(preds_scaled).flatten()

# test_rmse = np.sqrt(np.mean((preds - df[target_col].values.flatten())**2))
# print(f"Test RMSE: {test_rmse:.4f}")



# # Save predictions
# preds_df = pd.DataFrame({
#     "Molecule": df_names["Molecule"].values,
#     "Predicted_Breakdown_Strength": preds
# })
# # preds_df.to_csv(os.path.join(OUTPUT_DIR, "nn_predictions.csv"), index=False)
# # print(f"Saved predictions → {os.path.join(OUTPUT_DIR, 'nn_predictions.csv')}")





# # ------------------------------------------------
# # SHAP DeepExplainer
# # ------------------------------------------------

# X_training_scaled = X_scaler.transform(df_training[feature_names].values)


# background = X_training_scaled 
# explainer = shap.DeepExplainer(model, background)

# print("Computing SHAP values...")
# shap_raw = explainer.shap_values(X_training_scaled, check_additivity=False)

# # shap_raw is usually a list of length 1 for single-output models
# if isinstance(shap_raw, list):
#     shap_values = np.array(shap_raw[0])
# else:
#     shap_values = np.array(shap_raw)

# # Force to (n_samples, n_features)
# shap_values = shap_values.reshape(X_training_scaled.shape[0], X_training_scaled.shape[1])
# print("Final SHAP shape:", shap_values.shape)

# # ------------------------------------------------
# # Save SHAP results
# # ------------------------------------------------
# shap_df = pd.DataFrame(shap_values, columns=[f"SHAP_{f}" for f in feature_names])
# shap_df.insert(0, "Molecule", df_training_names["Molecule"].values)
# shap_df.to_csv(os.path.join(OUTPUT_DIR, "nn_shap_values_per_molecule.csv"), index=False)
# print("Saved per-sample SHAP values CSV.")

# # Combine predictions + SHAP
# combined = pd.concat([
#     preds_df.reset_index(drop=True),
#     shap_df.drop(columns=["Molecule"]).reset_index(drop=True)
# ], axis=1)

# combined.to_csv(os.path.join(OUTPUT_DIR, "nn_predictions_and_shap.csv"), index=False)
# print("Saved combined predictions + SHAP CSV.")

# # ------------------------------------------------
# # SHAP PLOTS
# # ------------------------------------------------
# plt.figure()
# shap.summary_plot(shap_values, X_training_scaled, feature_names=symbolic_feature_names, show=False)
# plt.xticks(fontsize=18)
# plt.yticks(fontsize=18)

# # Fix the color bar font
# cbar = plt.gcf().axes[-1]  # last axis is the color bar in a SHAP summary plot
# cbar.tick_params(labelsize=18)

# if len(plt.gcf().axes) > 1:
#     right_ax = plt.gcf().axes[1]
#     right_ax.set_ylabel("Feature value", fontsize=18)

# plt.tight_layout()
# plt.xlabel("SHAP Value (Impact on model)", fontsize=18, fontweight='bold')
# plt.savefig(os.path.join(OUTPUT_DIR, "shap_summary.png"), dpi=300, bbox_inches="tight")
# plt.close()

# plt.figure()
# shap.summary_plot(shap_values, X_training_scaled, feature_names=symbolic_feature_names, plot_type="bar", show=False)
# #plt.title("SHAP mean(|value|) (NN)")
# plt.xticks(fontsize=18)
# plt.yticks(fontsize=18)
# plt.tight_layout()
# plt.xlabel("mean(|SHAP Value|)", fontsize=18, fontweight='bold')
# plt.savefig(os.path.join(OUTPUT_DIR, "shap_bar.png"), dpi=300, bbox_inches="tight")
# plt.close()

# # Dependence plots
# for feat in feature_names:
#     plt.figure()
#     shap.dependence_plot(feat, shap_values, X_training_scaled, feature_names=feature_names, show=False)
#     plt.title(f"SHAP Dependence: {feat}")
#     fname = feat.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "")
#     plt.savefig(os.path.join(OUTPUT_DIR, f"shap_dependence_{fname}.png"), dpi=300, bbox_inches="tight")
#     plt.close()

# # -----------------------------
# # Print SHAP Feature Importance Ranking
# # -----------------------------
# mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
# importance_df = pd.DataFrame({
#     "Feature": feature_names,
#     "Mean |SHAP|": mean_abs_shap
# }).sort_values(by="Mean |SHAP|", ascending=False)

# print("\n SHAP Feature Importance (NN):")
# print(importance_df.to_string(index=False))