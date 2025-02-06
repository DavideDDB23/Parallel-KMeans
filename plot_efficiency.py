import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------
# 1. Example data (replace with your actual values)
# ----------------------------------------------------
# Number of processes
num_procs = np.array([2, 4, 8, 16, 32])

# Serial runtimes
T1_100D2 = 30.1454  # Serial runtime for the "100D2" dataset
T1_200k100 = 24.4559
T1_400k100 = 55.2375
T1_800k100 = 88.3941


# Parallel runtimes
Tn_100D2 = np.array([15.1717, 7.8164, 4.1010, 2.4109, 1.8077])
Tn_200k100 = np.array([12.3627, 6.2773, 3.6020, 1.9920, 1.2010])
Tn_400k100 = np.array([28.1105, 14.1946, 7.2059, 3.9621, 2.6150])
Tn_800k100 = np.array([44.6017, 22.6370, 11.9558, 6.0821, 3.3551])

# ----------------------------------------------------
# 2. Calculate efficiency
# ----------------------------------------------------
# Efficiency formula: Efficiency = T_serial / (P * T_parallel)
efficiency_100D2 = T1_100D2 / (num_procs * Tn_100D2)
efficiency_200k100 = T1_200k100 / (num_procs * Tn_200k100)
efficiency_400k100 = T1_400k100 / (num_procs * Tn_400k100)
efficiency_800k100 = T1_800k100 / (num_procs * Tn_800k100)

# ----------------------------------------------------
# 3. Plot the efficiency
# ----------------------------------------------------
plt.figure(figsize=(7, 5))

# Plot efficiency for both datasets
plt.plot(num_procs, efficiency_100D2, 's-', label='100D2')
plt.plot(num_procs, efficiency_200k100, 'o-', label='200k100')
plt.plot(num_procs, efficiency_400k100, 'p-', label='400k100')
plt.plot(num_procs, efficiency_800k100, 'x-', label='800k100')

# Add labels and title
plt.title('MPI Efficiency')
plt.xlabel('Number of Processes')
plt.ylabel('Efficiency (T_serial / (P * T_parallel))')

# Set x-axis ticks to display all process numbers
plt.xticks(num_procs)

# Grid, legend, and layout
plt.grid(True)
plt.legend()
plt.ylim(0, 1)

# Show (or save) the figure
plt.tight_layout()
plt.show()

# If you prefer to save directly, use:
# plt.savefig('efficiency_plot.png', dpi=300)