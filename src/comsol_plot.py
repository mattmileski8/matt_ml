import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df_point = pd.read_csv("./comsol_data/0.0625_2d_2e4.txt", sep='\s+', skiprows=9, header=None)
df_flat = pd.read_csv("./comsol_data/flat_2d_2_2e4.txt", sep='\s+', skiprows=9, header=None)
df_sphere = pd.read_csv("./comsol_data/sphere_2d_2e4.txt", sep='\s+', skiprows=9, header=None)
df_rog = pd.read_csv("./comsol_data/rog_2d_2e4.txt", sep='\s+', skiprows=9, header=None)


print(df_point.shape)

v_array = np.linspace(0, 20000, 41)
point_E = []
flat_E = []
sphere_E = []
rog_E = []

#print(v_array)


for i in range(len(v_array)):
    point_E.append(df_point.iloc[:,i+3].max() / 1000)
    flat_E.append(df_flat.iloc[:,i+3].max() / 1000)
    sphere_E.append(df_sphere.iloc[:,i+3].max() / 1000)
    rog_E.append(df_rog.iloc[:,i+3].max() / 1000)


# print(rog_E[-1])
# print(rog_E[1])

point_slope = point_E[-1] / 20000
flat_slope = flat_E[-1] / 20000
sphere_slope = sphere_E[-1] / 20000
rog_slope = rog_E[-1] / 20000


# print(point_slope)
# print(flat_slope)
# print(sphere_slope)
# print(rog_slope)



fig, ax = plt.subplots(figsize=(4, 3.2))

ax.plot(v_array, point_E, label='Pointed Electrode', linestyle='--')
ax.plot(v_array, sphere_E, label='Spherical Electrode', linestyle=':')
ax.plot(v_array, flat_E, label='Flat Electrode', linestyle='-.')
ax.plot(v_array, rog_E, label='Rogowski Electrode', linestyle='-')
ax.grid(True, alpha=0.5, linestyle='--', linewidth=0.5)
ax.tick_params(axis='both', labelsize=9)
ax.axhline(y=2.16*8.9, color='grey', linestyle='-', linewidth=1, label=r'Max E-field for $C_2F_3Cl_3$')
ax.axhline(y=1.01*8.9, color='k', linestyle='-', linewidth=1, label=r'Max E-field for $SF_6$')
ax.axhline(y=0.3793530079454656*8.9, color='gold', linestyle='-', linewidth=1, label=r'Max E-field for $CO_2$')
ax.set_xlabel('Applied Voltage (V)', fontweight='bold')
ax.set_ylabel('Max Electric Field (MV/m)', fontweight='bold')
ax.set_ylim(0, 45)
ax.set_xlim(0, 500)
#ax.set_yscale('log')
ax.legend(fontsize=8.6, loc='upper right')
plt.tight_layout()
#plt.savefig('./images/comsol_electric_field.png', dpi=300)


# ------------------------ Create a table for the max applied voltage each gas can withstand for each electrode ------------------------

# Path to the input CSV
input_path = "./results/rf_total_prediction_dataset.csv"  
 
# Read the CSV
df = pd.read_csv(input_path)
 
# Select the first three columns
first_three = df.iloc[:, :3]
 
# Build output path in the same folder
folder = os.path.dirname(input_path)
output_path = os.path.join(folder, "max_voltage_predictions.csv")
 
# Adds a new column to for the max voltage each gas can withstand for each electrode
first_three["Max V Pointed"] = df.iloc[:, 2] / point_slope * 8.9
first_three["Max V Spherical"] = df.iloc[:, 2] / sphere_slope * 8.9
first_three["Max V Flat"] = df.iloc[:, 2] / flat_slope * 8.9
first_three["Max V Rogowski"] = df.iloc[:, 2] / rog_slope * 8.9



# Save to new CSV
first_three.to_csv(output_path, index=False)

# ----------------------------------- Save values to latex table for the paper -----------------------------------

molecules_to_drop = ['C3F6O', 'C3F7NO', 'C2F4N2H2', 'CF3S_O_F', 'CH2_CHCH2F', 'ONBr', 'CH3Cl']
first_three = first_three[~first_three['Molecule'].isin(molecules_to_drop)].reset_index(drop=True)

import re

def format_molecule_name(name):
    return re.sub(r'(\d+)', r'$_{\1}$', str(name))

with open('./results/max_voltage_latex_table.txt', 'w') as f:
    for idx, row in first_three.iterrows():
        values = [idx + 1] + list(row.values)
        #values.insert(2, '')                       # Insert blank column at index 2
        formatted = []
        for i, val in enumerate(values):
            if i == 1:                             # Molecule name — subscript numbers
                formatted.append(format_molecule_name(val))
            elif i in [3, 4, 5, 6, 7]:            # 2 decimal places 
                formatted.append(f'{float(val):.2f}')
            elif isinstance(val, float):
                formatted.append(str(round(val, 3)))
            else:
                formatted.append(str(val))
        line = ' & '.join(formatted)
        f.write(line + ' \\\\\n')