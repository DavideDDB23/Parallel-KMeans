import numpy as np
import matplotlib.pyplot as plt

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

err_100D2 = np.array([0.0293,  0.565, 0.400, 0.0714, 0.0700])
err_200k100 = np.array([0.0316, 0.0173, 0.1337, 0.0300, 0.0469])
err_400k100 = np.array([0.4377, 0.2256, 0.0509, 0.1407, 0.4280])
err_800k100 = np.array([0.0932, 0.0141, 0.3969, 0.2969, 0.1846])

# ----------------------------------------------------
# 2. Compute speedup and speedup error
# ----------------------------------------------------
# Speedup is defined as T1 / Tn
speedup_100D2 = T1_100D2 / Tn_100D2
speedup_200k100 = T1_200k100 / Tn_200k100
speedup_400k100 = T1_400k100 / Tn_400k100
speedup_800k100 = T1_800k100 / Tn_800k100

# To propagate error bars for speedup from Tn error,
#   dSpeedup = (T1 / Tn^2) * dTn
# Here, we assume T1 is exact and the only uncertainty is in Tn
speedup_err_100D2 = (T1_100D2 / (Tn_100D2**2)) * err_100D2
speedup_err_200k100 = (T1_200k100 / (Tn_200k100**2)) * err_200k100
speedup_err_400k100 = (T1_400k100 / (Tn_400k100**2)) * err_400k100
speedup_err_800k100 = (T1_800k100 / (Tn_800k100**2)) * err_800k100

# ----------------------------------------------------
# 3. Plot the speedup
# ----------------------------------------------------
plt.figure(figsize=(7, 5))

# Plot speedups with error bars
plt.errorbar(
    num_procs, speedup_100D2, yerr=speedup_err_100D2,
    fmt='s-', capsize=4, label='100D2'
)

plt.errorbar(
    num_procs, speedup_200k100, yerr=speedup_err_200k100,
    fmt='o-', capsize=4, label='200k100'
)

plt.errorbar(
    num_procs, speedup_400k100, yerr=speedup_err_400k100,
    fmt='p-', capsize=4, label='400k100'
)

plt.errorbar(
    num_procs, speedup_800k100, yerr=speedup_err_800k100,
    fmt='x-', capsize=4, label='800k100'
)

# Labels, legend, grid
plt.title('Speedup MPI')
plt.xlabel('Number of Processes')
plt.ylabel('Speedup (T1 / Tn)')

# Ensure all process numbers are shown on the x-axis
plt.xticks(num_procs)

# Grid, legend, and layout
plt.grid(True)
plt.legend()

# Show (or save) the figure
plt.tight_layout()
plt.show()

# If you prefer to save directly, use:
# plt.savefig('speedup_plot.png', dpi=300)