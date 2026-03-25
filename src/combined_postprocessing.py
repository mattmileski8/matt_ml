import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Creates 

# Load both prediction files
df_rf = pd.read_csv("./results/rf_test_rmse_per_loop_6.csv")
df_nn = pd.read_csv("./results/nn_test_rmse_per_loop_6.csv")

# Find the minimum and maximum Test_RMSE for both models and the corresponding rows
min_rf_test_rmse = np.min(df_rf["Test_RMSE"])
min_nn_test_rmse = np.min(df_nn["Test_RMSE"])
max_rf_test_rmse = np.max(df_rf["Test_RMSE"])
max_nn_test_rmse = np.max(df_nn["Test_RMSE"])

min_rf_rmse_row = df_rf.loc[df_rf["Test_RMSE"].idxmin()]
min_nn_rmse_row = df_nn.loc[df_nn["Test_RMSE"].idxmin()]
max_rf_rmse_row = df_rf.loc[df_rf["Test_RMSE"].idxmax()]
max_nn_rmse_row = df_nn.loc[df_nn["Test_RMSE"].idxmax()]

# Find the average Test_RMSE and R² for both models
avg_rf_test_rmse = np.mean(df_rf["Test_RMSE"])
avg_nn_test_rmse = np.mean(df_nn["Test_RMSE"])
avg_rf_r2 = np.mean(df_rf["R2_Test"])
avg_nn_r2 = np.mean(df_nn["R2_Train"])

median_rf_test_rmse = np.median(df_rf["Test_RMSE"])
median_nn_test_rmse = np.median(df_nn["Test_RMSE"])
median_rf_r2 = np.median(df_rf["R2_Test"])
median_nn_r2 = np.median(df_nn["R2_Train"])

avg_rf_test_rmse_std = np.std(df_rf["Test_RMSE"])
avg_nn_test_rmse_std = np.std(df_nn["Test_RMSE"])
avg_rf_r2_std = np.std(df_rf["R2_Test"])
avg_nn_r2_std = np.std(df_nn["R2_Train"])

print("\nRF min:\n", min_rf_rmse_row)
print("\nRF max:\n", max_rf_rmse_row)
print("\nNN min:\n", min_nn_rmse_row)
print("\nNN max:\n", max_nn_rmse_row)


# Find the model run whose test RMSE is closest to the mean (closest to the average performance)
closest_rf_row = df_rf.loc[(df_rf["Test_RMSE"] - avg_rf_test_rmse).abs().idxmin()]
closest_nn_row = df_nn.loc[(df_nn["Test_RMSE"] - avg_nn_test_rmse).abs().idxmin()]

print("\nRF closest to average:\n", closest_rf_row)
print("\nNN closest to average:\n", closest_nn_row)

print("\nRF average Test_RMSE:", avg_rf_test_rmse, "±", avg_rf_test_rmse_std)
print("NN average Test_RMSE:", avg_nn_test_rmse, "±", avg_nn_test_rmse_std)

print("RF average R²:", avg_rf_r2, "±", avg_rf_r2_std)
print("NN average R²:", avg_nn_r2, "±", avg_nn_r2_std)

print("\nRF median Test_RMSE:", median_rf_test_rmse)
print("NN median Test_RMSE:", median_nn_test_rmse)
print("RF median R²:", median_rf_r2)
print("NN median R²:", median_nn_r2)