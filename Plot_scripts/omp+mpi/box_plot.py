import pandas as pd
import matplotlib.pyplot as plt

# Load data
omp_mpi_data = pd.read_csv("/Users/davide/Desktop/summary_statistics_omp_mpi.csv", delimiter=';')

# Boolean to select test files
use_selected_files = True  # Change to False to select other files

# Define file sets
selected_files = ['input100D.inp', 'input10D.inp', 'input20D.inp', 'input2D.inp']
if use_selected_files:
    filtered_data = omp_mpi_data[omp_mpi_data['Test File'].isin(selected_files)]
else:
    filtered_data = omp_mpi_data[~omp_mpi_data['Test File'].isin(selected_files)]

filtered_data['Scaled Std Dev'] = filtered_data['Std Dev'] / 2

# Define the specific combination of processes and threads
selected_processes = 4
selected_threads = 4

# Filter data for the specified combination of processes and threads
filtered_data = filtered_data[
    (filtered_data['Processes'] == selected_processes) &
    (filtered_data['Threads'] == selected_threads)
]

# Prepare data for the box plot
boxplot_data = []
test_files = filtered_data['Test File'].unique()

# Collect relevant statistics for each test file
for test_file in test_files:
    file_data = filtered_data[filtered_data['Test File'] == test_file]
    boxplot_data.append([
        file_data['Min'].values[0],
        file_data['Q1'].values[0],
        file_data['Median'].values[0],
        file_data['Q3'].values[0],
        file_data['Max'].values[0]
    ])

# Plot the box plot
plt.figure(figsize=(10, 6), dpi=150)
plt.boxplot(boxplot_data, vert=True, patch_artist=True, labels=[tf.replace('.inp', '') for tf in test_files])

# Customize the plot
plt.title(f'Runtime, {selected_processes} MPI Processes and {selected_threads} OpenMP Threads')
plt.ylabel('Execution Time (s)')
plt.xticks(rotation=45)
plt.grid(True, axis='y', linestyle='--', alpha=0.7)

# Save and show the plot
plt.tight_layout()
plt.savefig(f'/Users/davide/Desktop/Parallel-KMeans/Plots/mpi_omp/omp_mpi_boxplot_{selected_processes}_{selected_threads}_{use_selected_files}.png', dpi=150)
plt.show()