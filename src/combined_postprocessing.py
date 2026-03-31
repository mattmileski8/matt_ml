import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# -----------------Evaluates the results of the RF and NN models -------------------------------------
# Load both prediction files
df_rf = pd.read_csv("./results/rf_test_rmse_per_loop_6_2.csv")
df_nn = pd.read_csv("./results/nn_test_rmse_per_loop.csv")

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
#avg_rf_r2_std = np.std(df_rf["R2_Test"])
#avg_nn_r2_std = np.std(df_nn["R2_Train"])
avg_rf_r2_IQR = np.percentile(df_rf["R2_Test"], 75) - np.percentile(df_rf["R2_Test"], 25)
avg_nn_r2_IQR = np.percentile(df_nn["R2_Train"], 75) - np.percentile(df_nn["R2_Train"], 25)


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

print("RF average R²:", avg_rf_r2)
print("NN average R²:", avg_nn_r2)

print("\nRF median Test_RMSE:", median_rf_test_rmse)
print("NN median Test_RMSE:", median_nn_test_rmse)
print("RF median R²:", median_rf_r2, "±", avg_rf_r2_IQR)
print("NN median R²:", median_nn_r2, "±", avg_nn_r2_IQR)



# # ------------------- Import, merge, and plot the predictions from both models -------------------------------------
# # Load the prediction datasets
# rf_test_pred = pd.read_csv("./results/rf_test_prediction_dataset.csv")
# rf_tm_pred = pd.read_csv("./results/rf_tm_prediction_dataset.csv")
# rf_train_pred = pd.read_csv("./results/rf_train_prediction_dataset.csv")

# nn_test_pred = pd.read_csv("./results/nn_test_prediction_dataset.csv")
# nn_tm_pred = pd.read_csv("./results/nn_tm_prediction_dataset.csv")
# nn_train_pred = pd.read_csv("./results/nn_train_prediction_dataset.csv")

# rf_total_pred = pd.concat([rf_train_pred, rf_test_pred, rf_tm_pred], ignore_index=True)
# nn_total_pred = pd.concat([nn_train_pred, nn_test_pred, nn_tm_pred], ignore_index=True)

# rf_total_pred = rf_total_pred[(rf_total_pred['Adiabatic IE'] <= 100) & (rf_total_pred['Cohesive Energy'].abs() <= 30000)] #Remove any molecules with Adiabatic IE > 100 and Cohesive Energy > 30000, which are likely outliers
# nn_total_pred = nn_total_pred[(nn_total_pred['Adiabatic IE'] <= 100) & (nn_total_pred['Cohesive Energy'].abs() <= 30000)] #Remove any molecules with Adiabatic IE > 100 and Cohesive Energy > 30000, which are likely outliers

# rf_total_pred = rf_total_pred.sort_values('Predicted Dielectric Strength', ascending=False).reset_index(drop=True)
# rf_total_pred = rf_total_pred.reset_index(drop=True)
# nn_total_pred = nn_total_pred.reset_index(drop=True)
# rf_total_pred.to_csv("./results/rf_total_prediction_dataset.csv", index=True)
# nn_total_pred.to_csv("./results/nn_total_prediction_dataset.csv", index=True)


# #print(rf_total_pred[rf_total_pred['Adiabatic IE'] > 100][['Molecule', 'Adiabatic IE']]) #Check for any molecules with very high Adiabatic IE values that might be outliers


# # rf_top5 = rf_total_pred.nlargest(5, 'Predicted Dielectric Strength')[['Molecule', 'Predicted Dielectric Strength']]
# nn_top10 = nn_total_pred.nlargest(10, 'Predicted Dielectric Strength')[['Molecule', 'Predicted Dielectric Strength']]

# # print("RF Top 5:")
# # print(rf_top5.to_string(index=False))
# print("\nNN Top 10:")
# print(nn_top10.to_string(index=False))

# # print("\n total predictions:", len(rf_total_pred))



# molecule_order = rf_total_pred[['Molecule']].copy()
# molecule_order['sort_order'] = range(len(molecule_order))

# nn_total_pred = molecule_order.merge(nn_total_pred, on='Molecule').sort_values('sort_order').drop(columns='sort_order').reset_index(drop=True)

# #print(rf_total_pred[['Molecule', 'Predicted Dielectric Strength']].head(10).to_string(index=False))
# #print(nn_total_pred[['Molecule', 'Predicted Dielectric Strength']].head(10).to_string(index=False))

# print(rf_total_pred)

# fig, ax = plt.subplots(figsize=(4, 3.5))

# ax.scatter(rf_total_pred.index, rf_total_pred['Predicted Dielectric Strength'], color='steelblue', s=10, label='RF-Predicted Values')
# ax.scatter(nn_total_pred.index, nn_total_pred['Predicted Dielectric Strength'], marker='d', color='orange', s=10, label='NN-Predicted Values')

# ax.set_xlabel('Molecule Index', fontweight='bold')
# ax.set_ylabel('Predicted Relative DS', fontweight='bold')
# ax.grid(True, linestyle='--', alpha=0.5)
# ax.legend()
# plt.tight_layout()
# plt.savefig('./images/rf_nn_combined_predictions.png', dpi=300)



# ---------------------------- Save rf predictions in a new text file in latex table format -------------------------------------


# with open('./results/prediction_latex_table.txt', 'w') as f:
#     for idx, row in rf_total_pred.iterrows():
#         values = [idx + 1] + list(row.values)
#         formatted = []
#         for i, val in enumerate(values):
#             if i in [3, 7]:                # Vibrational ZPE, Cohesive Energy — scientific notation
#                 formatted.append(f'{float(val):.2e}')
#             elif i in [2, 4, 5, 6, 10]:   # 2 decimal places
#                 formatted.append(f'{float(val):.2f}')
#             elif i in [8, 9]:              # rounded to whole number
#                 formatted.append(f'{round(float(val))}')
#             elif isinstance(val, float):
#                 formatted.append(str(round(val, 3)))
#             else:
#                 formatted.append(str(val))
#         line = ' & '.join(formatted)
#         f.write(line + ' \\\\\n')