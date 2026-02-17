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


# # Define the column names based on the header
# columns = [
#     "Molecule",
#     "Vibrational ZPE (cm^-1)",
#     "Polarizability (Å^3)",
#     "Dipole Moment (Debye)",
#     "Adiabatic IE (eV)",
#     "Cohesive Energy (kJ/mol)"
# ]

# columns_training = [
#     "Molecule",
#     "Vibrational ZPE (cm^-1)",
#     "Polarizability (Å^3)",
#     "Dipole Moment (Debye)",
#     "Adiabatic IE (eV)",
#     "Cohesive Energy (kJ/mol)",
#     "Breakdown Voltage (MV/m)"
# ]


# # Load the file
# with open('./data/molecular_data_tm.txt', "r") as file:
#     lines = file.readlines()

# data = []

# for line in lines:
#     # Remove leading/trailing whitespace
#     line = line.strip()
#     if not line or line.startswith("Molecule"):
#         continue  # Skip empty and header lines

#     # Match molecule name (non-numeric part at the start)
#     match = re.match(r'^(\S+)', line)
#     if match:
#         molecule = match.group(1)
#         # Extract all numbers (scientific notation or float)
#         values = re.findall(r'(?<![A-Za-z0-9])[-+]?\d*\.\d+(?:e[+-]?\d+)?|[-+]?\d+(?:\.\d+)?', line[len(molecule):])
#         # Fill missing values with None (so all rows have 6 columns)
#         while len(values) < 5:
#             values.append(None)
#         data.append([molecule] + values)

# # Convert to DataFrame
# df = pd.DataFrame(data, columns=columns)
# df = df.replace("", pd.NA).dropna()



# #---------------------------------------------------------------------
# # Load the training data
# with open('./data/molecular_data.txt', "r") as file:
#     lines = file.readlines()

# data = []

# for line in lines:
#     # Remove leading/trailing whitespace
#     line = line.strip()
#     if not line or line.startswith("Molecule"):
#         continue  # Skip empty and header lines

#     # Match molecule name (non-numeric part at the start)
#     match = re.match(r'^(\S+)', line)
#     if match:
#         molecule = match.group(1)
#         # Extract all numbers (scientific notation or float)
#         values = re.findall(r'(?<![A-Za-z0-9])[-+]?\d*\.\d+(?:e[+-]?\d+)?|[-+]?\d+(?:\.\d+)?', line[len(molecule):])
#         # Fill missing values with None (so all rows have 6 columns)
#         while len(values) < 5:
#             values.append(None)
#         data.append([molecule] + values)

# # Convert to DataFrame
# df_training = pd.DataFrame(data, columns=columns_training)
# df_training = df_training.replace("", pd.NA).dropna()
# df_training = df_training.drop(columns=["Breakdown Voltage (MV/m)"])
# df_training["Molecule"] = df_training["Molecule"] + "*"
# df_all = pd.concat([df, df_training], ignore_index=True)

# df = df_all
# #-----------------------------------------------------------------------


# # Convert numeric columns to float
# for col in columns[1:]:
#     df[col] = pd.to_numeric(df[col], errors='coerce')


# # Remove unrealistic values
# df = df[
#     (df["Adiabatic IE (eV)"].abs() <= 1.3e2) &
#     (df["Cohesive Energy (kJ/mol)"].abs() <= 1.3e4)
# ].reset_index(drop=True)



# # Load saved NN model & scalers
# model_path = "./models/final_NN.keras"
# scaler_X_path = "./models/final_scaler_X.pkl"
# scaler_y_path = "./models/final_scaler_y.pkl"

# nn_model = load_model(model_path)
# X_scaler = joblib.load(scaler_X_path)
# y_scaler = joblib.load(scaler_y_path)


# # Prepare input features and scale using same scaler used during training
# molecule_names = df["Molecule"].values
# X_new = df.iloc[:, 1:].values
# X_new_scaled = X_scaler.transform(X_new)

# # Predict (model outputs scaled values, so inverse transform is needed)
# preds_scaled = nn_model.predict(X_new_scaled)
# preds = y_scaler.inverse_transform(preds_scaled).flatten()   # convert back to MV/m

# # Create prediction DataFrame
# df_pred = pd.DataFrame({
#     "Molecule": molecule_names,
#     "Predicted Breakdown Voltage (MV/m)": preds
# })

# # Sort descending by predicted value
# df_pred = df_pred.sort_values(by="Predicted Breakdown Voltage (MV/m)", ascending=False).reset_index(drop=True)

# # Insert index after sorting
# df_pred.insert(0, "Index", range(len(df_pred)))

# print("\n NN Predictions DataFrame (Top 5):")
# print(df_pred.head(10))

# feature_names = df.columns[1:]
# mins = np.min(X_new_scaled, axis=0)
# maxs = np.max(X_new_scaled, axis=0)

# for name, mn, mx in zip(feature_names, mins, maxs):
#     print(f"{name:30s} min={mn:.2f}, max={mx:.2f}")

# print(df_pred)

# # Save predictions to CSV
# os.makedirs("./results", exist_ok=True)
# output_path = "./results/NN_predicted_breakdown_voltages_sorted.csv"
# df_pred.to_csv(output_path, index=False)
# print(f"\n Saved predictions to: {output_path}")

# # Plot predictions vs index
# plt.figure(figsize=(8,5))
# plt.scatter(df_pred["Index"], df_pred["Predicted Breakdown Voltage (MV/m)"])
# plt.xlabel("Molecule Index")
# plt.ylabel("Predicted Breakdown Field (MV/m)")
# plt.title("NN Predicted Breakdown Strength for Molecules (Sorted)")
# plt.grid(True)

# os.makedirs("./images", exist_ok=True)
# plt.savefig("./images/NN_predicted_breakdowns_sorted.png", dpi=300, bbox_inches="tight")
# plt.close()


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