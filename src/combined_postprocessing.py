import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Creates 

# Load both prediction files
df_rf = pd.read_csv("./results/rf_test_rmse_per_loop_7.csv")
df_nn = pd.read_csv("./results/nn_test_rmse_per_loop_7.csv")

# Find the minimum and maximum Test_RMSE for both models and the corresponding rows
min_rf_test_rmse = np.min(df_rf["Test_RMSE"])
min_nn_test_rmse = np.min(df_nn["Test_RMSE"])
max_rf_test_rmse = np.max(df_rf["Test_RMSE"])
max_nn_test_rmse = np.max(df_nn["Test_RMSE"])

min_rf_rmse_row = df_rf.loc[df_rf["Test_RMSE"].idxmin()]
min_nn_rmse_row = df_nn.loc[df_nn["Test_RMSE"].idxmin()]
max_rf_rmse_row = df_rf.loc[df_rf["Test_RMSE"].idxmax()]
max_nn_rmse_row = df_nn.loc[df_nn["Test_RMSE"].idxmax()]

# Find the average Test_RMSE for both models
avg_rf_test_rmse = np.mean(df_rf["Test_RMSE"])
avg_nn_test_rmse = np.mean(df_nn["Test_RMSE"])


print("\nRF min:\n", min_rf_rmse_row)
print("\nRF max:\n", max_rf_rmse_row)
print("\nNN min:\n", min_nn_rmse_row)
print("\nNN max:\n", max_nn_rmse_row)


# Find the model run whose test RMSE is closest to the mean (closest to the average performance)
closest_rf_row = df_rf.loc[(df_rf["Test_RMSE"] - avg_rf_test_rmse).abs().idxmin()]
closest_nn_row = df_nn.loc[(df_nn["Test_RMSE"] - avg_nn_test_rmse).abs().idxmin()]

print("\nRF closest to average:\n", closest_rf_row)
print("\nNN closest to average:\n", closest_nn_row)

print("\nRF average Test_RMSE:", avg_rf_test_rmse)
print("NN average Test_RMSE:", avg_nn_test_rmse)
