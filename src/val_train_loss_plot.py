import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV files
df_train = pd.read_csv("./data/20260227-204402_train.csv")
df_val = pd.read_csv("./data/20260227-204402_validation.csv")
#print(df_train)

# Plot 2nd column (x) vs 3rd column (y)
plt.plot(df_train["Step"], df_train["Value"], color="red", label="Training Loss")
plt.plot(df_val["Step"], df_val["Value"], label="Validation Loss")

plt.legend()
plt.xlabel("Epoch")
plt.savefig("./results/val_train_loss_plot.png")
# plt.xlabel("Column 2")
# plt.ylabel("Column 3")
# plt.title("Column 2 vs Column 3")

# plt.show()

