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

names = df["Molecule"]
df = df.drop(columns=["Molecule"])

# Convert numeric columns to float
for col in columns[1:]:
    df[col] = pd.to_numeric(df[col], errors='coerce')

#----------Train initial model
data_train, data_test = train_test_split(df, test_size=0.1, random_state=42)

X_train = data_train.drop(columns=['Breakdown Voltage (MV/m)'])
X_test = data_test.drop(columns=['Breakdown Voltage (MV/m)'])

y_train = data_train[['Breakdown Voltage (MV/m)']]
y_test = data_test[['Breakdown Voltage (MV/m)']]

# Scale only features
scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

#scaler_label = StandardScaler()
#y_train = scaler_label.fit_transform(y_train)
#y_test = scaler_label.transform(y_test)

#print(np.sqrt(.1737)*scaler_label.scale_[0])

# Convert to np arrays
X_train_input = np.array(X_train)
X_test_input = np.array(X_test)
y_train_input = np.array(y_train)
y_test_input = np.array(y_test)

# Flatten input for BaggingRegressor
y_train_input = y_train_input.ravel()
y_test_input = y_test_input.ravel()

n_estimators = 100
max_depth = None
min_split = 2
min_leaf = 1

# rf = RandomForestRegressor(
#     n_estimators=n_estimators,
#     max_depth=max_depth,
#     min_samples_split=min_split,
#     min_samples_leaf=min_leaf,
#     random_state=42,
#     n_jobs=-1,
#     oob_score=True, 
#     bootstrap=True 
# )

# rf.fit(X_train_input, y_train_input)

# oob_score = rf.oob_score_

# # Compute OOB RMSE (standardized)
# oob_rmse_std = np.sqrt(mean_squared_error(y_train_input, rf.oob_prediction_))

# # Convert to original MV/m
# #oob_rmse = oob_rmse_std * scaler_label.scale_[0]


#---------CV Score----------------------

rf_cv = RandomForestRegressor(
    n_estimators=n_estimators,
    max_depth=max_depth,
    min_samples_split=min_split,
    min_samples_leaf=min_leaf,
    random_state=42,
    n_jobs=-1,
    oob_score=True, 
    bootstrap=True 
)

neg_mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)

# 10-fold cross-validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)
cv_scores = cross_val_score(
    rf_cv,
    X_train_input,
    y_train_input,
    scoring=neg_mse_scorer,
    cv=kf,
    n_jobs=-1
)

# Convert to RMSE
cv_rmse = np.sqrt(-cv_scores)

print("Cross-validation RMSE per fold:", cv_rmse)
print("Mean CV RMSE:", np.mean(cv_rmse))
print("Std CV RMSE:", np.std(cv_rmse))


# # Predictions
# y_train_pred = rf.predict(X_train_input)
# y_test_pred = rf.predict(X_test_input)

# # Compute RMSE (standardized)
# train_rmse_std = np.sqrt(mean_squared_error(y_train_input, y_train_pred))
# test_rmse_std = np.sqrt(mean_squared_error(y_test_input, y_test_pred))

# # Convert back to original MV/m units
# #train_rmse = train_rmse_std * scaler_label.scale_[0]
# #test_rmse = test_rmse_std * scaler_label.scale_[0]

# metadata = {
#     "n_estimators": n_estimators,
#     "max_depth": max_depth,
#     "min_samples_split": min_split,
#     "min_samples_leaf": min_leaf,
#     "oob_r2": oob_score,
#     "oob_rmse": oob_rmse_std
# }
# joblib.dump(metadata, "./models/rf_metadata.pkl")



# # ----------------------------
# # ✅ Retrain Final Model on 100% of Data
# # ----------------------------

# X_full = df.drop(columns=['Breakdown Voltage (MV/m)'])
# y_full = df[['Breakdown Voltage (MV/m)']]

# # Scale only features
# final_scaler = StandardScaler()
# X_full = pd.DataFrame(final_scaler.fit_transform(X_full), columns=X_full.columns)

# #final_scaler_label = StandardScaler()
# #y_full = final_scaler_label.fit_transform(y_full)

# #print(np.sqrt(.1737)*scaler_label.scale_[0])

# # Convert to np arrays
# X_full_input = np.array(X_full)
# y_full_input = np.array(y_full)

# # Flatten input for BaggingRegressor
# y_full_input = y_full_input.ravel()


# final_model = RandomForestRegressor(
#     n_estimators=n_estimators,
#     max_depth=max_depth,
#     min_samples_split=min_split,
#     min_samples_leaf=min_leaf,
#     random_state=42,
#     n_jobs=-1,
#     oob_score=True,
#     bootstrap=True
# )

# final_model.fit(X_full_input, y_full_input)

# # ----------------------------
# # Save Final Model + Scalers
# # ----------------------------


# os.makedirs("./models", exist_ok=True)
# joblib.dump(final_model, "./models/final_rf_model.pkl")
# joblib.dump(final_scaler, "./models/rf_X_scaler.pkl")
# #joblib.dump(final_scaler_label, "./models/y_scaler.pkl")

# print("\n✅ Final model retrained on ALL data and saved successfully!")



# ----------------------------
# Load Model & Scaler
# ----------------------------
rf_model = joblib.load("./models/final_rf_model.pkl")
X_scaler = joblib.load("./models/rf_X_scaler.pkl")

# Prepare full dataset again
X = df.drop(columns=['Breakdown Voltage (MV/m)'])
y = df['Breakdown Voltage (MV/m)']

# Scale X
X_scaled = X_scaler.transform(X)

# Predict
preds = rf_model.predict(X_scaled)
y_actual = y.values

# ----------------------------
# Regression Fit Line & R²
# ----------------------------
#slope, intercept, r_value, _, _ = linregress(y_actual, preds)
#y_fit = slope * y_actual + intercept
r2 = r2_score(y_actual, preds)

# ----------------------------
# Plot
# ----------------------------
plt.figure(figsize=(3.5, 3.5))
plt.scatter(y_actual, preds, alpha=0.7)#, label="Predictions")
plt.plot(y_actual, y_actual, linestyle='-', color='red')#,
         #label=f"$R^2 = {r2:.3f}$")
#plt.plot([], [], ' ', label=f"$R^2 = {r2:.3f}$")           # dummy plot for blank label

plt.xlabel("Actual Breakdown Field (MV/m)", fontsize=8.5, fontweight='bold')
plt.ylabel("Predicted Breakdown Field (MV/m)", fontsize=8.5, fontweight='bold')
plt.xticks(fontsize=8.5)
plt.yticks(fontsize=8.5)
#plt.title("RF: Actual vs Predicted Electric Field at Breakdown")
#plt.legend(fontsize=8.5, labelspacing=0.2)
plt.grid(True)

plt.text(
    0.05, 0.95,                   # (x, y) position in axes coordinates
    f"$R^2 = {r2:.3f}$",
    transform=plt.gca().transAxes,
    fontsize=8.5,
    fontweight='bold',
    verticalalignment='top',
)


# Save
os.makedirs("./images", exist_ok=True)
plt.savefig("./images/final_RF_actual_vs_predicted_fitline.png", dpi=300, bbox_inches="tight")
plt.close()

residuals = y_actual - preds
z = np.abs((residuals - np.mean(residuals)) / np.std(residuals))

outliers = z > 3
for i in range(len(outliers)):
    if outliers[i] == True: 
        print(names[i])
        print(y_actual[i])
        print(preds[i])