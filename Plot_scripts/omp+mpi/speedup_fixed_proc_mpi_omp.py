import pandas as pd
import matplotlib.pyplot as plt

# Load data
mpi_data = pd.read_csv('/Users/davide/Desktop/summary_statistics_omp_mpi.csv', delimiter=';')
sequential_data = pd.read_csv('/Users/davide/Desktop/summary_statistics_seq.csv', delimiter=',')

selected_files = ['input100D.inp', 'input10D.inp', 'input20D.inp', 'input2D.inp']
mpi_data = mpi_data[~mpi_data['Test File'].isin(selected_files)]

# Select relevant columns and merge based on 'Test File'
sequential_data = sequential_data[['Test File', 'Average']].rename(columns={'Average': 'Average_seq'})
merged_data = mpi_data.merge(sequential_data, on='Test File')

processes = 8

# Filter data for MPI processes = 8 and Threads = {2, 4, 8, 16, 32}
threads_set = {2, 4, 8, 16, 32}
filtered_data = merged_data[(merged_data['Processes'] == processes) & (merged_data['Threads'].isin(threads_set))]

# Calculate speedup as T_serial / T_parallel
filtered_data['Speedup'] = filtered_data['Average_seq'] / filtered_data['Average']
filtered_data['Scaled Std Dev'] = filtered_data['Std Dev'] / 1

# Plot setup
plt.figure(figsize=(8, 6), dpi=150)

# Define markers and styles
markers = ['o', 's', 'p', '^', 'x', 'd', 'h', '*', '>', '<']
plot_styles = zip(filtered_data['Test File'].unique(), markers)

# Plot speedup data for each test file
for test_file, marker in plot_styles:
    test_data = filtered_data[filtered_data['Test File'] == test_file]
    if not test_data.empty:
        plt.errorbar(
            test_data['Threads'],
            test_data['Speedup'],
            yerr=test_data['Scaled Std Dev'],
            fmt=f'{marker}-',
            capsize=5,
            label=test_file.replace('.inp', '')
        )

# Customize the plot
plt.title(f'Speedup MPI+OMP, Fixed MPI Processes = {processes}')
plt.xlabel('Number of Threads')
plt.ylabel('Speedup (T1 / Tp)')
plt.xticks(sorted(threads_set))
plt.legend(loc='upper left')
plt.grid(True)
plt.tight_layout()

plt.savefig(f'/Users/davide/Desktop/Parallel-KMeans/Plots/mpi_omp/speedup_fixed_proc_mpi_omp_{processes}.png', dpi=150)

# Show the plot
plt.show()