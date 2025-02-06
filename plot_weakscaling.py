import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ----------------------------------------------------
# 1. Input data
# ----------------------------------------------------
# Number of processes
num_procs = np.array([2, 4, 8, 16, 32])

# Serial runtimes (for each dataset size)
serial_times = np.array([11.3399, 24.4559, 55.2375, 88.3941, 157.8033])
serial_errors = np.array([0.0591, 0.0346, 0.2445, 0.1326, 0.2549])

# Parallel runtimes (for each dataset size with respective processes)
parallel_times = np.array([6.1215, 6.2492, 7.2059, 6.8138, 6.4952])
parallel_errors = np.array([0.1400, 0.0331, 0.0509, 0.3182, 0.1476])

# ----------------------------------------------------
# 2. Calculate efficiency
# ----------------------------------------------------
# Efficiency formula: Efficiency = T_serial / (P * T_parallel)
efficiency = serial_times / (num_procs * parallel_times)

# ----------------------------------------------------
# 3. Plot efficiency
# ----------------------------------------------------
plt.figure(figsize=(8, 6))

# Plot the efficiency
plt.plot(num_procs, efficiency, 'o-', label='Weak Scaling', color='green')

# Add error bars for better visualization
plt.errorbar(num_procs, efficiency, yerr=parallel_errors / (num_procs * parallel_times), fmt='o', color='green')

# Customize the plot
plt.title('Weak Scaling')
plt.xlabel('Number of Processes')
plt.ylabel('Efficiency (T_serial / (P * T_parallel))')
plt.ylim(0, 1)
plt.xticks(num_procs)
plt.grid(True)
plt.legend()

# ----------------------------------------------------
# 4. Show or save the plot
# ----------------------------------------------------
plt.tight_layout()
plt.show()

# If you prefer to save the figure, uncomment the line below:
# plt.savefig('weak_scaling_efficiency.png', dpi=300)