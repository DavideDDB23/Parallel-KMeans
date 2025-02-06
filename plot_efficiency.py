import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------
# 1. Example data (replace with your actual values)
# ----------------------------------------------------
# Number of processes
num_procs = np.array([2, 4, 8, 16, 32])

# Serial runtimes
T1_100D  = 0.5630  # Serial runtime for the "100D" dataset
T1_100D2 = 30.1454  # Serial runtime for the "100D2" dataset

# Parallel runtimes
Tn_100D  = np.array([100, 52, 28, 22, 17])
Tn_100D2 = np.array([15.1717, 7.8164, 4.1010, 2.4109, 1.8077])

# ----------------------------------------------------
# 2. Calculate efficiency
# ----------------------------------------------------
# Efficiency formula: Efficiency = T_serial / (P * T_parallel)
efficiency_100D  = T1_100D  / (num_procs * Tn_100D)
efficiency_100D2 = T1_100D2 / (num_procs * Tn_100D2)

# ----------------------------------------------------
# 3. Plot the efficiency
# ----------------------------------------------------
plt.figure(figsize=(7, 5))

# Plot efficiency for both datasets
plt.plot(num_procs, efficiency_100D, 'o-', label='Efficiency 100D')
plt.plot(num_procs, efficiency_100D2, 's-', label='Efficiency 100D2')

# Add labels and title
plt.title('MPI Efficiency')
plt.xlabel('Number of Processes')
plt.ylabel('Efficiency (T_serial / (P * T_parallel))')

# Set x-axis ticks to display all process numbers
plt.xticks(num_procs)

# Grid, legend, and layout
plt.grid(True)
plt.legend()

# Show (or save) the figure
plt.tight_layout()
plt.show()

# If you prefer to save directly, use:
# plt.savefig('efficiency_plot.png', dpi=300)