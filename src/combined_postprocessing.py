import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Creates 

# Load both prediction files
df_rf = pd.read_csv("./results/rf_test_rmse_per_loop.csv")
df_nn = pd.read_csv("./results/nn_test_rmse_per_loop.csv")

# Merge them on the "Molecule" column
min_rf_test_rmse = np.min(df_rf["Test_RMSE"])
min_nn_test_rmse = np.min(df_nn["Test_RMSE"])
max_rf_test_rmse = np.max(df_rf["Test_RMSE"])
max_nn_test_rmse = np.max(df_nn["Test_RMSE"])

min_rf_rmse_row = df_rf.loc[df_rf["Test_RMSE"].idxmin()]
min_nn_rmse_row = df_nn.loc[df_nn["Test_RMSE"].idxmin()]
max_rf_rmse_row = df_rf.loc[df_rf["Test_RMSE"].idxmax()]
max_nn_rmse_row = df_nn.loc[df_nn["Test_RMSE"].idxmax()]




print("\nRF min:\n", min_rf_rmse_row)
print("\nRF max:\n", max_rf_rmse_row)
print("\nNN min:\n", min_nn_rmse_row)
print("\nNN max:\n", max_nn_rmse_row)


print(np.mean(df_rf["Test_RMSE"]))
print(np.mean(df_nn["Test_RMSE"]))