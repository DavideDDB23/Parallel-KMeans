import pandas as pd
import matplotlib.pyplot as plt

# Load data
mpi_data = pd.read_csv('/Users/davide/Desktop/summary_statistics_mpi_new.csv')

# Filter MPI data for test files with 8 processes
mpi_data_8_processes = mpi_data[mpi_data['Processes'] == 16]

# Prepare data for box plot
boxplot_data = []
test_files = mpi_data_8_processes['Test File'].unique()

# Collect relevant statistics for each test file
for test_file in test_files:
    file_data = mpi_data_8_processes[mpi_data_8_processes['Test File'] == test_file]
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
plt.title('Runtime 16 Processes MPI')
plt.ylabel('Execution Time (s)')
plt.xticks(rotation=45)
plt.grid(True, axis='y', linestyle='--', alpha=0.7)

# Save and show the plot
plt.tight_layout()
plt.savefig('/Users/davide/Desktop/Parallel-KMeans/Plots/mpi_boxplot_16_processes_new.png', dpi=150)
plt.show()