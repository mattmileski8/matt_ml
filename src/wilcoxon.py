import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

# --- File paths (adjust as needed) ---
file_A = "./results/rf_model_prediction_errors.txt"
file_B = "./results/nn_model_prediction_errors.txt"

label_A = "RF"
label_B = "NN"

# --- Load both error files ---
df_A = pd.read_csv(file_A, sep="\t")
df_B = pd.read_csv(file_B, sep="\t")

# --- Align by Molecule to guarantee correct pairing ---
# (merge ensures row order/order mismatches don't corrupt the pairing)
merged = pd.merge(
    df_A[["Molecule", "y_true", "abs_error", "squared_error"]],
    df_B[["Molecule", "y_true", "abs_error", "squared_error"]],
    on="Molecule",
    suffixes=(f"_{label_A}", f"_{label_B}")
)

n_A, n_B, n_merged = len(df_A), len(df_B), len(merged)
print(f"{label_A} file: {n_A} rows | {label_B} file: {n_B} rows | matched after merge: {n_merged}")
if n_merged < min(n_A, n_B):
    print("WARNING: some molecules did not match between the two files — check for naming mismatches.")

# --- Sanity check: y_true should match between the two files for matched molecules ---
mismatch = merged[
    ~np.isclose(merged[f"y_true_{label_A}"], merged[f"y_true_{label_B}"])
]
if len(mismatch) > 0:
    print(f"WARNING: {len(mismatch)} molecules have different y_true values between files. "
          f"Check that both files are evaluated on the same dataset.")

# --- Compute paired differences ---
d_abs = merged[f"abs_error_{label_A}"] - merged[f"abs_error_{label_B}"]
d_sq = merged[f"squared_error_{label_A}"] - merged[f"squared_error_{label_B}"]

# --- Run Wilcoxon signed-rank test ---
def run_wilcoxon(d, error_type):
    d = d[d != 0]  # scipy drops zeros by default with zero_method='wilcox', but be explicit
    if len(d) == 0:
        print(f"{error_type}: no non-zero differences, cannot run test.")
        return
    stat, p = wilcoxon(d)
    median_diff = np.median(d)
    direction = label_A if median_diff > 0 else label_B
    print(f"\n--- Wilcoxon signed-rank test ({error_type}) ---")
    print(f"n pairs (non-zero): {len(d)}")
    print(f"statistic: {stat:.4f}")
    print(f"p-value:   {p:.6f}")
    print(f"median difference ({label_A} - {label_B}): {median_diff:.6f}")
    print(f"  -> {'Significant' if p < 0.05 else 'Not significant'} at alpha=0.05")
    print(f"  -> Higher errors in: {direction if p < 0.05 else 'N/A (no significant difference)'}")

run_wilcoxon(d_abs, "Absolute Error")
run_wilcoxon(d_sq, "Squared Error")

# --- Also report RMSE/MAE for context ---
rmse_A = np.sqrt(merged[f"squared_error_{label_A}"].mean())
rmse_B = np.sqrt(merged[f"squared_error_{label_B}"].mean())
mae_A = merged[f"abs_error_{label_A}"].mean()
mae_B = merged[f"abs_error_{label_B}"].mean()

print(f"\n--- Summary ---")
print(f"{label_A}: RMSE = {rmse_A:.4f}, MAE = {mae_A:.4f}")
print(f"{label_B}: RMSE = {rmse_B:.4f}, MAE = {mae_B:.4f}")

# --- Save merged comparison table + results ---
merged.to_csv("./wilcoxon_comparison_data.txt", sep="\t", index=False)
print("\nSaved merged per-molecule comparison to ./wilcoxon_comparison_data.txt")