"""
Computes 95% t-confidence intervals for Test_RMSE and R2_Test
from a comma-separated file (.csv or .txt).

Set FILE_PATH below to your file, then run:
    python t_interval.py
"""

import csv
import math
from pathlib import Path

# ── Set your file path here ──────────────────────────────────────────────────
FILE_PATH = "./results/nn_test_rmse_per_loop.csv"
# ─────────────────────────────────────────────────────────────────────────────


def mean(values):
    return sum(values) / len(values)


def std(values, xbar):
    n = len(values)
    variance = sum((x - xbar) ** 2 for x in values) / (n - 1)
    return math.sqrt(variance)


def t_critical(df, confidence=0.95):
    """
    Approximates the two-tailed t critical value using the
    scipy.stats module if available, otherwise falls back to a
    Wilson-Hilferty normal approximation (accurate for large df).
    """
    try:
        from scipy.stats import t
        alpha = 1 - confidence
        return t.ppf(1 - alpha / 2, df)
    except ImportError:
        z = 1.959963985
        tc = z + (z**3 + z) / (4 * df) + (5 * z**5 + 16 * z**3 + 3 * z) / (96 * df**2)
        return tc


def compute_ci(values, confidence=0.95):
    n = len(values)
    xbar = mean(values)
    s = std(values, xbar)
    df = n - 1
    t_star = t_critical(df, confidence)
    margin = t_star * (s / math.sqrt(n))
    return xbar, s, n, df, t_star, margin, (xbar - margin, xbar + margin)


def load_column(filepath, column_name):
    values = []
    with open(filepath, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            val = row.get(column_name)
            if val is not None and val.strip() != "":
                values.append(float(val.strip()))
    return values


def report(label, values, confidence=0.95):
    xbar, s, n, df, t_star, margin, (lo, hi) = compute_ci(values, confidence)
    pct = int(confidence * 100)
    print(f"\n{'='*50}")
    print(f"  {label}")
    print(f"{'='*50}")
    print(f"  n                 : {n}")
    print(f"  Mean              : {xbar:.6f}")
    print(f"  Std dev (s)       : {s:.6f}")
    print(f"  Degrees of freedom: {df}")
    print(f"  t* ({pct}%, df={df}): {t_star:.6f}")
    print(f"  Margin of error   : ±{margin:.6f}")
    print(f"  {pct}% CI         : ({lo:.6f},  {hi:.6f})")


def main():
    filepath = Path(FILE_PATH)
    if not filepath.exists():
        print(f"Error: file not found — {filepath}")
        return

    print(f"\nFile: {filepath.name}")

    for col in ("Test_RMSE", "R2_Test"):
        values = load_column(filepath, col)
        if not values:
            print(f"\nWarning: no data found for column '{col}'")
            continue
        report(col, values)

    print()


if __name__ == "__main__":
    main()







