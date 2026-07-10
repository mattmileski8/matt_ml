import re
import os

# ---- Configuration ----
table1_path = "./results/existing_latex_table.txt"  # 12-column table (formula/CAS/properties)
table2_path = "./results/max_voltage_latex_table.txt"  # 8-column table (extra properties)
output_path = "./results/combined_latex_table.txt"

# Which columns to keep from each table (0-indexed)
table1_cols = [0, 1, 2, 3]      # first four columns
table2_cols = [-4, -3, -2, -1]  # last four columns


def parse_latex_row(line):
    """Split a LaTeX table row like 'a & b & c \\\\' into a list of
    trimmed field strings, dropping the trailing '\\\\' line-break marker."""
    line = line.strip()
    # Remove a trailing LaTeX line-break marker ('\\', with optional spaces)
    line = re.sub(r"\\\\\s*$", "", line)
    return [field.strip() for field in line.split("&")]


def read_table(path):
    with open(path, "r") as f:
        lines = [ln for ln in f.read().splitlines() if ln.strip()]
    return [parse_latex_row(ln) for ln in lines]


# ---- Read both tables ----
table1 = read_table(table1_path)
table2 = read_table(table2_path)
table2 = table2[1:]

if len(table1) != len(table2):
    raise ValueError(
        f"Row count mismatch: table1 has {len(table1)} rows, "
        f"table2 has {len(table2)} rows."
    )

# ---- Combine selected columns row by row ----
combined_lines = []
for row1, row2 in zip(table1, table2):
    selected = [row1[i] for i in table1_cols] + [row2[i] for i in table2_cols]
    combined_lines.append(" & ".join(selected) + " \\\\")

with open(output_path, "w") as f:
    f.write("\n".join(combined_lines) + "\n")

print(f"Saved combined table to: {output_path}")