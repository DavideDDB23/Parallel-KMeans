import pandas as pd
import matplotlib.pyplot as plt

# Load data
mpi_data = pd.read_csv('/Users/davide/Desktop/summary_statistics_omp_mpi.csv', delimiter=';')

# Boolean to select test files
use_selected_files = False  # Change to False to select other files

# Define file sets
selected_files = ['input100D.inp', 'input10D.inp', 'input20D.inp', 'input2D.inp']
if use_selected_files:
    filtered_data = mpi_data[mpi_data['Test File'].isin(selected_files)]
else:
    filtered_data = mpi_data[~mpi_data['Test File'].isin(selected_files)]

filtered_data['Scaled Std Dev'] = filtered_data['Std Dev'] / 2

threads = 32

# Filter data for MPI processes = 2 and Threads = {2, 4, 8, 16, 32}
processes_set = {2, 4, 8}
filtered_plot_data = filtered_data[(filtered_data['Threads'] == threads) & (filtered_data['Processes'].isin(processes_set))]

# Plot setup
plt.figure(figsize=(8, 6), dpi=150)

# Define markers and styles
markers = ['o', 's', 'p', '^', 'x', '>']
plot_styles = zip(filtered_plot_data['Test File'].unique(), markers)

# Plot data for each test file
for test_file, marker in plot_styles:
    test_data = filtered_plot_data[filtered_plot_data['Test File'] == test_file]
    if not test_data.empty:
        plt.errorbar(
            test_data['Processes'],
            test_data['Average'],
            yerr=test_data['Scaled Std Dev'],
            fmt=f'{marker}-',
            capsize=5,
            label=test_file.replace('.inp', '')
        )

# Customize the plot
plt.title(f'Runtime, OpenMP Threads = {threads}')
plt.xlabel('Number of Processes')
plt.ylabel('Execution Time (s)')
plt.xticks(sorted(processes_set))
plt.legend(loc='upper right')
plt.grid(True)
plt.tight_layout()

plt.savefig(f'/Users/davide/Desktop/Parallel-KMeans/Plots/mpi_omp/exectime_fixed_threads_mpi_omp_{threads}.png', dpi=150)

# Show the plot
plt.show()