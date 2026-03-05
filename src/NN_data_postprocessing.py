import os
import re
import joblib
import numpy as np
import pandas as pd
#import shap
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import tensorflow as tf
from tensorflow.keras.models import load_model


# Plot validation and training mae
history_df_5600 = pd.read_csv("./data/5600_training_history.csv")
history_df_10000 = pd.read_csv("./data/10000_training_history.csv")

window = 200

val_5600_smooth = history_df_5600["val_mae"].rolling(window=window).mean()
val_10000_smooth = history_df_10000["val_mae"].rolling(window=window).mean()

plt.figure()
plt.plot(history_df_10000['epoch'], history_df_10000['val_mae'], label='Validation MAE')
plt.plot(history_df_10000["epoch"], val_10000_smooth, linewidth=2, label="Validation (smoothed)")
#plt.plot(history_df_5600['epoch'], history_df_5600['mae'], label='Training MAE')
#plt.plot(history_df_5600['epoch'], history_df_5600['val_mae'], label='Validation MAE')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend()
plt.title('Training vs Validation MAE')
plt.savefig("./results/smoothed_val_mae.png")

# best_epoch = history_df_5600.loc[history_df_5600["val_mae"].idxmin(), "epoch"]
# print("Best epoch (val_mae):", best_epoch)

# min_val_mae = history_df_5600["val_mae"].min()
# print("Minimum val_mae:", min_val_mae)

print(val_10000_smooth.min())
#----------------------------------------------------------------------------
# feature importance
#----------------------------------------------------------------------------
# columns = [
#     "Molecule",
#     "Vibrational ZPE (cm^-1)",
#     "Polarizability (Å^3)",
#     "Dipole Moment (Debye)",
#     "Adiabatic IE (eV)",
#     "Cohesive Energy (kJ/mol)", 
#     "Breakdown Strength (MV/m)"
# ]

# MODEL_PATH = "./models/final_NN.keras"
# SCALER_X_PATH = "./models/final_scaler_X.pkl"
# SCALER_Y_PATH = "./models/final_scaler_y.pkl"
# OUTPUT_DIR = "./results/shap_nn_all"
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# # -----------------------------
# # Column names (Molecule + 5 features)
# # -----------------------------
# symbolic_feature_names = [
#     r"$\varepsilon_{V}$",   # Vibrational ZPE 
#     r"$\alpha$",                   # Polarizability
#     r"$\mu$",                      # Dipole Moment
#     r"$\varepsilon_{I}$",              # Adiabatic IE
#     r"$\varepsilon_{c}$"            # Cohesive Energy
# ]


# feature_names = columns[1:-1]   # Only the 5 input features
# target_col = columns[-1]

# # -----------------------------
# # Load & Parse .txt data
# # -----------------------------
# data = []
# with open('./data/molecular_data.txt', "r") as file:
#     for line in file:
#         line = line.strip()
#         if not line or line.startswith("Molecule"):
#             continue

#         # Molecule name (first token)
#         match = re.match(r'^(\S+)', line)
#         molecule = match.group(1)

#         # Extract numbers
#         values = re.findall(
#             r'[-+]?\d*\.\d+e[+-]?\d+|[-+]?\d+\.\d+|[-+]?\d+',
#             line[len(molecule):]
#         )

#         # Ensure 6 numeric values (5 features + breakdown strength)
#         while len(values) < 6:
#             values.append(None)
#         values = values[:6]

#         data.append([molecule] + values)

# df = pd.DataFrame(data, columns=columns)

# # Convert numeric columns
# for col in columns[1:]:
#     df[col] = pd.to_numeric(df[col], errors='coerce')

# df = df.dropna()  # drop rows with missing numbers

# # pd.set_option('display.max_rows', None)
# # pd.set_option('display.max_columns', None)
# # print(len(df))

# # -----------------------------
# # Load model & scalers
# # -----------------------------

# model = load_model(MODEL_PATH)
# X_scaler = joblib.load(SCALER_X_PATH)
# y_scaler = joblib.load(SCALER_Y_PATH)

# # ------------------------------------------------
# # Build X and predict y
# # ------------------------------------------------
# X = df[feature_names].values
# X_scaled = X_scaler.transform(X)

# # Predict scaled target
# preds_scaled = model.predict(X_scaled)
# preds_scaled = np.asarray(preds_scaled).reshape(-1, 1)

# # Inverse transform to real MV/m
# preds = y_scaler.inverse_transform(preds_scaled).flatten()

# # Save predictions
# preds_df = pd.DataFrame({
#     "Molecule": df["Molecule"].values,
#     "Predicted_Breakdown_Strength_MVpm": preds
# })
# preds_df.to_csv(os.path.join(OUTPUT_DIR, "nn_predictions.csv"), index=False)
# print(f"Saved predictions → {os.path.join(OUTPUT_DIR, 'nn_predictions.csv')}")

# # ------------------------------------------------
# # SHAP DeepExplainer
# # ------------------------------------------------
# print("Building DeepExplainer (this may take a moment)...")
# background = X_scaled     # uses all data — ok for ≤ few hundred rows
# explainer = shap.DeepExplainer(model, background)

# print("Computing SHAP values...")
# shap_raw = explainer.shap_values(X_scaled)

# # shap_raw is usually a list of length 1 for single-output models
# if isinstance(shap_raw, list):
#     shap_values = np.array(shap_raw[0])
# else:
#     shap_values = np.array(shap_raw)

# # Force to (n_samples, n_features)
# shap_values = shap_values.reshape(X_scaled.shape[0], X_scaled.shape[1])
# print("Final SHAP shape:", shap_values.shape)

# # ------------------------------------------------
# # Save SHAP results
# # ------------------------------------------------
# shap_df = pd.DataFrame(shap_values, columns=[f"SHAP_{f}" for f in feature_names])
# shap_df.insert(0, "Molecule", df["Molecule"].values)
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
# shap.summary_plot(shap_values, X_scaled, feature_names=symbolic_feature_names, show=False)
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
# shap.summary_plot(shap_values, X_scaled, feature_names=symbolic_feature_names, plot_type="bar", show=False)
# #plt.title("SHAP mean(|value|) (NN)")
# plt.xticks(fontsize=18)
# plt.yticks(fontsize=18)
# plt.tight_layout()
# plt.xlabel("mean(|SHAP Value|)", fontsize=18, fontweight='bold')
# plt.savefig(os.path.join(OUTPUT_DIR, "shap_bar.png"), dpi=300, bbox_inches="tight")
# plt.close()

# Dependence plots
# for feat in feature_names:
#     plt.figure()
#     shap.dependence_plot(feat, shap_values, X_scaled, feature_names=feature_names, show=False)
#     plt.title(f"SHAP Dependence: {feat}")
#     fname = feat.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "")
#     plt.savefig(os.path.join(OUTPUT_DIR, f"shap_dependence_{fname}.png"), dpi=300, bbox_inches="tight")
#     plt.close()

# -----------------------------
# Print SHAP Feature Importance Ranking
# -----------------------------
# mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
# importance_df = pd.DataFrame({
#     "Feature": feature_names,
#     "Mean |SHAP|": mean_abs_shap
# }).sort_values(by="Mean |SHAP|", ascending=False)

# print("\n SHAP Feature Importance (NN):")
# print(importance_df.to_string(index=False))