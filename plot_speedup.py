import numpy as np
import matplotlib.pyplot as plt

# Number of processes (excluding serial, which is T1)
num_procs = np.array([2, 4, 8, 16, 32])

# Suppose we ran a serial version (T1):
T1_100D  = 0.5630  # Serial runtime for the "100D" dataset
T1_100D2 = 30.1454  # Serial runtime for the "100D2" dataset

# Parallel runtimes for each number of processes (example data):
Tn_100D  = np.array([100, 52, 28, 22, 17])
Tn_100D2 = np.array([15.1717, 7.8164, 4.1010, 2.4109, 1.8077])

err_100D  = np.array([ 0,  2,  2,  3,  2])
err_100D2 = np.array([ 0.0293,  0.565,  0.400,  0.0714,  0.0700])

# ----------------------------------------------------
# 2. Compute speedup and speedup error
# ----------------------------------------------------
# Speedup is defined as T1 / Tn
speedup_100D  = T1_100D  / Tn_100D
speedup_100D2 = T1_100D2 / Tn_100D2

# To propagate error bars for speedup from Tn error,
#   dSpeedup = (T1 / Tn^2) * dTn
# Here, we assume T1 is exact and the only uncertainty is in Tn
speedup_err_100D  = (T1_100D  / (Tn_100D**2))  * err_100D
speedup_err_100D2 = (T1_100D2 / (Tn_100D2**2)) * err_100D2

# ----------------------------------------------------
# 3. Plot the speedup
# ----------------------------------------------------
plt.figure(figsize=(7, 5))

# Plot speedups with error bars
plt.errorbar(
    num_procs, speedup_100D, yerr=speedup_err_100D,
    fmt='o-', capsize=4, label='Speedup 100D'
)
plt.errorbar(
    num_procs, speedup_100D2, yerr=speedup_err_100D2,
    fmt='s-', capsize=4, label='Speedup 100D2'
)

# Labels, legend, grid
plt.title('Speedup MPI')
plt.xlabel('Number of Processes')
plt.ylabel('Speedup (T1 / Tn)')

# Ensure all process numbers are shown on the x-axis
plt.xticks(num_procs)
plt.yticks(speedup_100D2+speedup_100D)

# Grid, legend, and layout
plt.grid(True)
plt.legend()

# Show (or save) the figure
plt.tight_layout()
plt.show()

# If you prefer to save directly, use:
# plt.savefig('speedup_plot.png', dpi=300)