import pandas as pd
import matplotlib.pyplot as plt

# Load data
omp_data = pd.read_csv('/Users/davide/Desktop/summary_statistics_omp.csv')

# Filter OpenMP data for test files with 8 threads
omp_data_8_threads = omp_data[omp_data['Threads'] == 16]

# Prepare data for box plot
boxplot_data = []
test_files = omp_data_8_threads['Test File'].unique()

# Collect relevant statistics for each test file
for test_file in test_files:
    file_data = omp_data_8_threads[omp_data_8_threads['Test File'] == test_file]
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
plt.title('Runtime 16 Threads OpenMP')
plt.ylabel('Execution Time (s)')
plt.xticks(rotation=45)
plt.grid(True, axis='y', linestyle='--', alpha=0.7)

# Save and show the plot
plt.tight_layout()
plt.savefig('/Users/davide/Desktop/Parallel-KMeans/Plots/omp_boxplot_16_threads.png', dpi=150)
plt.show()