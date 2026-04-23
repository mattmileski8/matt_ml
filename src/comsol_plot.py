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


print(point_E)





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
plt.savefig('./images/comsol_electric_field.png', dpi=300)