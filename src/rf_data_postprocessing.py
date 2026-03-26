import os
import re
import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score


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

MODEL_PATH = "./models/eight_descriptors/rf_avg_model.pkl"
OUTPUT_DIR = "./results/shap_rf_8_descriptors_all"
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv("./data/molecular_data_sorted.txt", sep="\t")
df_names = pd.read_csv("./data/molecular_names_sorted.txt", sep="\t")

df_test = pd.read_csv("./data/test_seven_sorted.txt", sep="\t")
df_test_names = pd.read_csv("./data/test_seven_names_sorted.txt", sep="\t")

df_pred = pd.read_csv("./data/molecular_tm_data_sorted.txt", sep="\t")
df_pred_names = pd.read_csv("./data/molecular_tm_names_sorted.txt", sep="\t")

# Attach molecule names to the feature data (molecule names live in a separate file)
df = pd.concat([df_names, df], axis=1)
df_test = pd.concat([df_test_names, df_test], axis=1)
df_pred = pd.concat([df_pred_names, df_pred], axis=1)


# ------------ Make predictions and calculate test R² and RMSE -------------
rf_model = joblib.load(MODEL_PATH)
feature_names = columns[1:6] + columns[7:]  # 8 input features (excluding DS)
#feature_names = columns[2:6] + columns[7:]  # 7 input features (excluding DS, Vibrational ZPE)
#feature_names = columns[2:6] + columns[7:8] + columns[9:]  # 6 input features (excluding DS, Vibrational ZPE, and # e-)
X = df_test[feature_names]
y_true_test = df_test["Breakdown Voltage"].values

y_pred_test = rf_model.predict(X)
rf_RMSE = np.sqrt(np.mean((y_true_test - y_pred_test) ** 2))
print(f"RF RMSE on test data: {rf_RMSE:.3f}")
r2 = r2_score(y_true_test, y_pred_test)
print(f"R² on test data: {r2:.3f}")

stdev_residuals_test = np.std(y_true_test - y_pred_test)
print(f"Standard deviation of residuals (test): {stdev_residuals_test:.3f}")

# --------------- Make predictions on training data and plot ------------------------------------
X_train = df[feature_names]
y_true_train = df["Breakdown Voltage"].values

y_pred_train = rf_model.predict(X_train)
rf_RMSE_train = np.sqrt(np.mean((y_true_train - y_pred_train) ** 2))
print(f"RF RMSE on training data: {rf_RMSE_train:.3f}")
r2_train = r2_score(y_true_train, y_pred_train)
print(f"R² on training data: {r2_train:.3f}")

stdev_residuals_train = np.std(y_true_train - y_pred_train)
print(f"Standard deviation of residuals (train): {stdev_residuals_train:.3f}")

# ----------------------- Parity Plot ----------------------------------------
fig, ax = plt.subplots(figsize=(4, 3.2))

ax.scatter(y_true_train, y_pred_train,  color='steelblue', edgecolors='k', alpha=0.7, label=f'Train (R² = {r2_train:.3f}), $\\sigma$={stdev_residuals_train:.3f}')
ax.scatter(y_true_test, y_pred_test, marker='s', color='orange', edgecolors='k', alpha=0.7, label=f'Test (R² = {r2:.3f}), $\\sigma$={stdev_residuals_test:.3f}')
# Plot y=x parity line
min_val = min(y_true_train.min(), y_pred_train.min())
max_val = max(y_true_train.max(), y_pred_train.max())
ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1)#, label='Parity line')

ax.set_xlabel('True Relative DS', fontweight='bold')
ax.set_ylabel('Predicted Relative DS', fontweight='bold')
ax.tick_params(axis='both', labelsize=9)
#ax.set_title(f'Parity Plot')
ax.legend(fontsize=8.6, loc='upper left')
plt.tight_layout()
plt.savefig("./images/rf_parity_plot_8_descriptors.png", dpi=300, bbox_inches="tight")

# ----------------------------Make predictions on predict data----------------------------------------------

X_tm = df_pred[feature_names]
y_pred_tm = rf_model.predict(X_tm)

y_pred_tm_series = pd.Series(y_pred_tm, name='Predicted Dielectric Strength')
tm_prediction_dataset = X_tm.copy()
tm_prediction_dataset.insert(0, 'Predicted Dielectric Strength', y_pred_tm_series)
tm_prediction_dataset.insert(0, 'Molecule', df_pred['Molecule'])

tm_prediction_dataset.to_csv('./results/rf_tm_prediction_dataset.csv', index=False)



# Save training predictions with molecule names and features to .csv for later analysis
y_pred_train_series = pd.Series(y_pred_train, name='y_pred_train')
train_prediction_dataset = X_train.copy()
train_prediction_dataset.insert(0, 'Predicted Dielectric Strength', y_pred_train_series)
train_prediction_dataset.insert(0, 'Molecule', df['Molecule'])

train_prediction_dataset.to_csv('./results/rf_train_prediction_dataset.csv', index=False)

# Save test predictions with molecule names and features to .csv for later analysis
y_pred_test_series = pd.Series(y_pred_test, name='y_pred_test')
test_prediction_dataset = X.copy()
test_prediction_dataset.insert(0, 'Predicted Dielectric Strength', y_pred_test_series)
test_prediction_dataset.insert(0, 'Molecule', df_test['Molecule'])

test_prediction_dataset.to_csv('./results/rf_test_prediction_dataset.csv', index=False)





# -----------------------------------------------------------------------------------------------

# # Sort descending by predicted value
# df_pred = df_pred.sort_values(by="Predicted Breakdown Voltage (MV/m)", ascending=False).reset_index(drop=True)

# df_pred.insert(0, "Index", range(len(df_pred)))

# print("\n Predictions DataFrame:")
# print(df_pred.head(10))

# # Save predictions
# os.makedirs("./results", exist_ok=True)
# output_path = "./results/rf_predicted_breakdown_voltages_sorted.csv"
# #df_pred.to_csv(output_path, index=False)



# # Plot predictions vs index
# plt.figure(figsize=(8,5))
# plt.scatter(df_pred["Index"], df_pred["Predicted Breakdown Voltage (MV/m)"], marker='o')
# plt.xlabel("Molecule Index")
# plt.ylabel("Predicted Breakdown Voltage (MV/m)")
# plt.title("RF Predicted Breakdown Strength for Molecules (sorted)")
# plt.grid(True)
# #plt.savefig("./images/RF_predicted_breakdowns.png", dpi=300, bbox_inches="tight")
# plt.close()



# ---------------------------------------------------------------
# Merge predictions back into the full dataframe (df)
# ---------------------------------------------------------------
# df_merged = df.merge(df_pred[["Molecule", "Predicted Breakdown Voltage (MV/m)"]],
#                      on="Molecule",
#                      how="left")

# # Sort by predicted breakdown strength (highest → lowest)
# df_merged = df_merged.sort_values(
#     by="Predicted Breakdown Voltage (MV/m)",
#     ascending=False
# ).reset_index(drop=True)

# # # Create ranking index for plotting
# # df_merged["Rank"] = df_merged.index

# # Rounds specified columns to 2 decimal places
# cols_to_round = ["Dipole Moment (Debye)", "Predicted Breakdown Voltage (MV/m)"]  # Replace with your column names
# df_merged[cols_to_round] = df_merged[cols_to_round].round(2)

# # Save predictions with DFT features included
# output_path_full = "./results/rf_predicted_breakdown_strength_with_features.csv"
# df_merged.to_csv(output_path_full, index=False)



# # converts csv to a text file in a format that can be copied as a latex table
# df_to_conv = pd.read_csv("./results/rf_predicted_breakdown_strength_with_features.csv")

# output_path = "./results/rf_predicted_breakdown_strength_with_features_for_latex.txt"

# with open(output_path, "w") as f:
#     # Write header row
#     f.write(" & ".join(df_to_conv.columns) + " \\\\ \\hline\n")

#     # Write data rows
#     for _, row in df_to_conv.iterrows():
#         f.write(" & ".join(map(str, row.values)) + " \\\\\n")


#--------Finds repeat molecules in final merged dataset----------------------------
# for i in range(len(df_merged) - 1):
#     if (df_merged.iloc[i]["Vibrational ZPE (cm^-1)"] - df_merged.iloc[i + 1]["Vibrational ZPE (cm^-1)"]) and (df_merged.iloc[i]["Polarizability (Å^3)"] == df_merged.iloc[i + 1]["Polarizability (Å^3)"]) and (df_merged.iloc[i]["Dipole Moment (Debye)"] == df_merged.iloc[i + 1]["Dipole Moment (Debye)"]) and (df_merged.iloc[i]["Adiabatic IE (eV)"] == df_merged.iloc[i + 1]["Adiabatic IE (eV)"]) and (df_merged.iloc[i]["Cohesive Energy (kJ/mol)"] == df_merged.iloc[i + 1]["Cohesive Energy (kJ/mol)"]) and (df_merged.iloc[i]["Predicted Breakdown Voltage (MV/m)"] == df_merged.iloc[i + 1]["Predicted Breakdown Voltage (MV/m)"]):
#         print("Match found:", df_merged.iloc[i]["Molecule"], df_merged.iloc[i + 1]["Molecule"])

# ---------------------------------------------------------------
# Create scatterplots for each feature vs. predicted breakdown
# ---------------------------------------------------------------

# features = [
#     "Vibrational ZPE (cm^-1)",
#     "Polarizability (Å^3)",
#     "Dipole Moment (Debye)",
#     "Adiabatic IE (eV)",
#     "Cohesive Energy (kJ/mol)"
# ]

# os.makedirs("./images/feature_vs_pred", exist_ok=True)

# for feat in features:

#     plt.figure(figsize=(3.5,2.8))
#     plt.scatter(df_merged["Rank"], df_merged[feat], alpha=0.7, s=7)

#     r = np.corrcoef(df_merged["Predicted Breakdown Voltage (MV/m)"], df_merged[feat])[0,1]
#     plt.text(0.05, 0.95, f"r = {r:.3f}", transform=plt.gca().transAxes,
#          fontsize=9, verticalalignment='top')

#     plt.xlabel("Molecule Index", fontsize=8.5, fontweight='bold')
#     plt.ylabel(feat, fontsize=8.5, fontweight='bold')
#     plt.title(f"{feat} vs. Predicted Breakdown Strength", fontsize=10)
#     plt.grid(True)

#     # Save figure
#     safe_name = feat.replace(" ", "_").replace("/", "_")
#     plt.savefig(f"./images/feature_vs_pred/{safe_name}.png",
#                 dpi=300, bbox_inches="tight")
#     plt.close()


#---------- This can be combined with the scatterplot section below in order for it to function
# df_plot = df.merge(
#     df_pred[["Molecule", "Predicted Breakdown Voltage (MV/m)"]],
#     on="Molecule",
#     how="left"
# )
# # Extract columns
# x = df_plot["Cohesive Energy (kJ/mol)"]
# y = df_plot["Polarizability (Å^3)"]
# c = df_plot["Predicted Breakdown Voltage (MV/m)"]

# # --------------------------------------------
# # Scatterplot
# # --------------------------------------------
# plt.figure(figsize=(7, 5))

# scatter = plt.scatter(
#     x, y,
#     c=c,
#     cmap="viridis",           # good perceptual colormap
#     s=50,
#     edgecolor="black",
#     linewidth=0.5
# )

# plt.xlabel("Cohesive Energy (kJ/mol)", fontsize=8.5, fontweight='bold')
# plt.ylabel("Polarizability (Å³)", fontsize=8.5, fontweight='bold')
# plt.title("Cohesive Energy vs Polarizability\nColored by Predicted Breakdown Strength", fontsize=11)

# cbar = plt.colorbar(scatter)
# cbar.set_label("Predicted Breakdown Strength (MV/m)", fontsize=10)

# plt.grid(True, alpha=0.3)
# plt.tight_layout()

# # Save and/or show
# plt.savefig("./images/cohesive_vs_polarizability_colored_by_pred.png",
#             dpi=300, bbox_inches="tight")
# plt.show()



#-----------------------------------------------------------------------------------
# feature importance
#-----------------------------------------------------------------------------------


# feature_names = columns[1:-1]   # Only the 5 input features
# target_col = columns[-1]

# symbolic_feature_names = [
#     r"$\varepsilon_{V}$",   # Vibrational ZPE 
#     r"$\alpha$",                   # Polarizability
#     r"$\mu$",                      # Dipole Moment
#     r"$\varepsilon_{I}$",              # Adiabatic IE
#     r"$\varepsilon_{c}$"            # Cohesive Energy
# ]

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

# df = df.dropna().reset_index(drop=True)  # drop rows with missing numbers


# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# print(df)



# # -----------------------------
# # Load Model + Scaler
# # -----------------------------

# rf_model = joblib.load(MODEL_PATH)







#------------------------------ SHAP Analysis -----------------------------------



# symbolic_feature_names = [
#     r"$\varepsilon_{V}$",   # Vibrational ZPE 
#     r"$\alpha$",                   # Polarizability
#     r"$\mu$",                      # Dipole Moment
#     r"$\varepsilon_{I}$",              # Adiabatic IE
#     r"$\varepsilon_{c}$",            # Cohesive Energy
#     r"$m$",           # Molecular Mass
#     r"$n_{e}$",           # Number of electrons
#     r"$V$"#,           # Molecular Volume
#     #"DS"
# ]

# # Build the feature matrix (raw, because RF was trained without scaling)
# X = df[feature_names].values

# # -----------------------------
# # 1. Feature Importance
# # -----------------------------
# importances = rf_model.feature_importances_
# importance_df = pd.DataFrame({
#     "Feature": feature_names,
#     "Importance": importances
# }).sort_values(by="Importance", ascending=False)

# print("\n Random Forest Feature Importances:")
# print(importance_df.to_string(index=False))

# # -----------------------------
# # 2. SHAP Analysis
# # -----------------------------
# explainer = shap.TreeExplainer(rf_model)
# shap_values = explainer.shap_values(X)

# # Convert to array if list
# if isinstance(shap_values, list):
#     shap_values = shap_values[0]

# print("\n SHAP values computed successfully!\n")

# # -----------------------------
# # Create Output Directory and save to .csv
# # -----------------------------
# output_dir = OUTPUT_DIR
# os.makedirs(output_dir, exist_ok=True)

# shap_df = pd.DataFrame(shap_values, columns=[f"SHAP_{f}" for f in feature_names])
# shap_df.insert(0, "Molecule", df_names)
# shap_df.to_csv(f"{output_dir}/rf_shap_values.csv", index=False)

# # -----------------------------
# # SHAP Summary Plot (Beeswarm)
# # -----------------------------
# plt.figure()
# shap.summary_plot(shap_values, X, feature_names=symbolic_feature_names, show=False)
# plt.xticks(fontsize=18)
# plt.yticks(fontsize=18)

# # Fix the color bar font
# cbar = plt.gcf().axes[-1]  # last axis is the color bar in a SHAP summary plot
# cbar.tick_params(labelsize=18)

# if len(plt.gcf().axes) > 1:
#     right_ax = plt.gcf().axes[1]
#     right_ax.set_ylabel("Feature value", fontsize=18)

# #plt.title("SHAP Summary: RF Feature Impact on Breakdown Strength")
# plt.tight_layout()
# plt.xlabel("SHAP Value (Impact on model)", fontsize=18, fontweight='bold')
# plt.savefig(f"{output_dir}/shap_summary.png", dpi=300, bbox_inches="tight")
# plt.close()


# # -----------------------------
# # SHAP Bar Plot (Mean |SHAP|)
# # -----------------------------
# plt.figure()
# shap.summary_plot(shap_values, X, feature_names=symbolic_feature_names, plot_type="bar", show=False)
# #plt.title("SHAP Feature Importance (Mean |SHAP|)")
# plt.xticks(fontsize=18)
# plt.yticks(fontsize=18)
# plt.tight_layout()
# plt.xlabel("mean(|SHAP Value|)", fontsize=18, fontweight='bold')
# plt.savefig(f"{output_dir}/shap_bar.png", dpi=300, bbox_inches="tight")
# plt.close()

# # -----------------------------
# # Dependence Plots
# # -----------------------------
# for feat in feature_names:
#     plt.figure()
#     shap.dependence_plot(feat, shap_values, X, feature_names=feature_names, show=False)
#     plt.title(f"SHAP Dependence: {feat}")
#     plt.tight_layout()
#     fname = feat.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "")
#     plt.savefig(f"{output_dir}/shap_dependence_{fname}.png", dpi=300, bbox_inches="tight")
#     plt.close()

