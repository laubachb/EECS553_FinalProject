import matplotlib.pyplot as plt
import numpy as np

# Data
x_values = np.arange(0, 1.1, 0.1)  # x-values from 0 to 1.0 in increments of 0.1

# First dataset
saheart_ours = [
    0.656003289,
    0.698190789,
    0.697505482,
    0.703042763,
    0.695805921,
    0.696628289,
    0.718722588,
    0.693037281,
    0.707730263,
    0.691694079,
    0.685827851
]
sa_heart_std_devs_1 = [
    0.01437993,
    0.018291182,
    0.012003293,
    0.01257202,
    0.015216483,
    0.013447421,
    0.012613731,
    0.021235797,
    0.003317113,
    0.007271802,
    0.019043172
]

# Second dataset
heart_ours = [
    0.775657895,
    0.82872807,
    0.822222222,
    0.827046784,
    0.822368421,
    0.830555556,
    0.813450292,
    0.813377193,
    0.793055556,
    0.804093567,
    0.79751462
]
heart_std_devs_2 = [
    0.017224338,
    0.004478197,
    0.02993749,
    0.036353494,
    0.019068862,
    0.05356686,
    0.009783586,
    0.042207934,
    0.031014489,
    0.023822224,
    0.052438119
]

x_values_2 = [
    0.0,
    0.2,
    0.4,
    0.6,
    0.8,
    1.0
]
# Dataset 3 (solid red line)
saheart_truth = [
    0.668885884,
    0.688691225,
    0.713906012,
    0.705031487,
    0.69994672,
    0.677000977
]

heart_truth = [
    0.705680843,
    0.783928175,
    0.79940633,
    0.78457938,
    0.769209451,
    0.73219619
]

# Plot
plt.figure(figsize=(10, 6))
ms = 10

# Dataset 1: Original data
plt.errorbar(
    x_values, saheart_ours, yerr=sa_heart_std_devs_1, fmt='^', capsize=5, capthick=1,
    color='blue', label='South African (Ours)', markersize=ms
)

# Dataset 2: New data
plt.errorbar(
    x_values, heart_ours, yerr=heart_std_devs_2, fmt='s', capsize=5, capthick=1,
    color='black', label='Heart Disease (Ours)', markersize=ms
)

# Dataset 3: Line with dots
plt.plot(
    x_values_2, saheart_truth, linestyle='-', color='green', linewidth=2, marker='o', label='South African (Literature)', markersize=ms
)

# Dataset 4: Line with dots
plt.plot(
    x_values_2, heart_truth, linestyle='-', color='red', linewidth=2, marker='d', label='Heart Disease (Literature)', markersize=ms
)

# Labels and title
# plt.title(r'Sensitivity Analysis of $p_m$', fontsize=20)
plt.xlabel(r'$p_m$', fontsize=16)
plt.ylabel('AUCROC', fontsize=16)

# Grid and legend
plt.legend(fontsize=14, loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.tick_params(axis='both', which='major', labelsize=14)

# Show plot
plt.savefig("hp_sensitivity.png")
