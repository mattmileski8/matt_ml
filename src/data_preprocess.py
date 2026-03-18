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

# # -------------Training Dataset ---------------------------------------------------------------------------------
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

# # Load the file
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
#         values = re.findall(r'[-+]?\d*\.\d+e[+-]?\d+|[-+]?\d+\.\d+|[-+]?\d+', line[len(molecule):])
#         # Fill missing values with None (so all rows have 6 columns)
#         while len(values) < 6:
#             values.append(None)
#         data.append([molecule] + values)

# # Convert to DataFrame
# df = pd.DataFrame(data, columns=columns)
# df = df.replace("", pd.NA).dropna()

# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)

# df = df.reset_index(drop=True)
# names = df["Molecule"]

# df = df.drop(columns=["Molecule"])


# # Convert numeric columns to float
# for col in columns[1:]:
#     df[col] = pd.to_numeric(df[col], errors='coerce')

# # Preview the result
# #print(df)
# #print(len(df))

# df.to_csv("./data/molecular_data_sorted.txt", sep="\t", index=False)
# names.to_csv("./data/molecular_names_sorted.txt", sep="\t", index=False)


#-------------Prediction Dataset----------------------------------------------------------------------------------
# Define the column names based on the header
columns_tm = [
    "Molecule",
    "Vibrational ZPE",
    "Polarizability",
    "Dipole Moment",
    "Adiabatic IE",
    "Cohesive Energy",
    "Molecular Mass",
    "Number e-",
    "Molecular Volume"
]

# Load the file
with open('./data/molecular_data_tm.txt', "r") as file:
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
df_tm = pd.DataFrame(data, columns=columns_tm)
df_tm = df_tm.replace("", pd.NA).dropna()

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

df_tm = df_tm.reset_index(drop=True)
names_tm = df_tm["Molecule"]

df_tm = df_tm.drop(columns=["Molecule"])


# Convert numeric columns to float
for col in columns_tm[1:]:
    df_tm[col] = pd.to_numeric(df_tm[col], errors='coerce')

# Preview the result
# print(df)
# print(len(df_tm))

df_tm.to_csv("./data/molecular_tm_data_sorted.txt", sep="\t", index=False)
names_tm.to_csv("./data/molecular_tm_names_sorted.txt", sep="\t", index=False)



# #----------------seven test mols------------------------------------------------------------

# # Load the file
# with open('./data/seven_test_mol.txt', "r") as file:
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
#         values = re.findall(r'[-+]?\d*\.\d+e[+-]?\d+|[-+]?\d+\.\d+|[-+]?\d+', line[len(molecule):])
#         # Fill missing values with None (so all rows have 6 columns)
#         while len(values) < 6:
#             values.append(None)
#         data.append([molecule] + values)

# # Convert to DataFrame
# df_test = pd.DataFrame(data, columns=columns)
# df_test = df_test.replace("", pd.NA).dropna()

# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)

# df_test = df_test.reset_index(drop=True)
# names_test = df_test["Molecule"]

# df_test = df_test.drop(columns=["Molecule"])


# # Convert numeric columns to float
# for col in columns[1:]:
#     df_test[col] = pd.to_numeric(df_test[col], errors='coerce')

# # Preview the result
# #print(df)
# print(len(df))

# df_test.to_csv("./data/test_seven_sorted.txt", sep="\t", index=False)
# names_test.to_csv("./data/test_seven_names_sorted.txt", sep="\t", index=False)



#-----------------------------------------------------------------------------
# Compute Pearson correlations between each feature and target and full matrix
#-----------------------------------------------------------------------------

# clean_columns = [
#     r"$\varepsilon_{V}$",   # Vibrational ZPE 
#     r"$\alpha$",                   # Polarizability
#     r"$\mu$",                      # Dipole Moment
#     r"$\varepsilon_{I}$",              # Adiabatic IE
#     r"$\varepsilon_{c}$",            # Cohesive Energy
#     r"$m$",           # Molecular Mass
#     r"$n_{e}$",           # Number of electrons
#     r"$V$",           # Molecular Volume
#     "DS"
# ]

# df.columns = clean_columns

# feature_names = clean_columns[0:-1]   # all input features
# target_col = clean_columns[-1]        # breakdown strength, not used right now

# # correlations = df.corr(method='pearson')[[target_col]].loc[feature_names]

# # print("\n Pearson Correlation (r) with Breakdown Strength:")
# # print(correlations)

# # corr_matrix = df.corr(method='pearson')


# # select only the five feature columns
# feature_df = df[feature_names]

# # Pearson correlation matrix of features only
# corr_matrix = feature_df.corr(method='pearson')

# plt.figure(figsize=(8, 6))
# sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", square=True, annot_kws={"size": 17})
# plt.xticks(rotation=0, ha='right', fontsize=17)
# plt.yticks(rotation=0, ha='right', fontsize=17)
# # plt.xlabel(fontsize=8.5)
# # plt.ylabel(fontsize=8.5)
# #plt.title("Pearson Correlation Matrix")

# cbar = plt.gcf().axes[-1]
# cbar.tick_params(labelsize=18)

# plt.tight_layout()
# plt.savefig(f"./results/pearson_correlation_heatmap.png", dpi=300)
# plt.close()
