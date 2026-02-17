import pandas as pd
import matplotlib.pyplot as plt

# Creates 

# Load both prediction files
df_rf = pd.read_csv("./results/rf_predicted_breakdown_voltages_sorted.csv")
df_nn = pd.read_csv("./results/NN_predicted_breakdown_voltages_sorted.csv")

# Merge them on the "Molecule" column
df_merged = df_rf.merge(df_nn[['Molecule', 'Predicted Breakdown Voltage (MV/m)']],
                        on='Molecule',
                        how='left')
#print(df_merged)

for index, row in df_merged.iterrows():
    print(index, row["Molecule"], row["Predicted Breakdown Voltage (MV/m)"])


# # Plot predictions vs index
# plt.figure(figsize=(3.5,2.8))
# plt.scatter(df_merged["Index"], df_merged["Predicted Breakdown Voltage (MV/m)_x"], marker='o', s=7, label='RF-Predicted Values')
# plt.scatter(df_merged["Index"], df_merged["Predicted Breakdown Voltage (MV/m)_y"], marker='d', s=4, label='NN-Predicted Value')
# plt.xlabel("Molecule Index", fontsize=8.5, fontweight='bold')
# plt.ylabel("Predicted Breakdown Field (MV/m)", fontsize=8.1, fontweight='bold')
# plt.xticks(fontsize=8.5)
# plt.yticks(fontsize=8.5)
# plt.legend(fontsize=8.5)
# #plt.title("RF Predicted Breakdown Strength for Molecules (sorted)")
# plt.grid(True)
# plt.savefig("./images/combined_predicted_breakdowns.png", dpi=300, bbox_inches="tight")
# plt.close()