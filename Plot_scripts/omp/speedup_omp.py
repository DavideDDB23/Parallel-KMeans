import pandas as pd
import matplotlib.pyplot as plt

# Load data
omp_data = pd.read_csv('/Users/davide/Desktop/summary_statistics_omp.csv')
seq_data = pd.read_csv('/Users/davide/Desktop/summary_statistics_seq.csv')

# Merge data and calculate speedup
merged_omp_data = omp_data.merge(seq_data[['Test File', 'Average']], on='Test File', suffixes=('_omp', '_seq'))
merged_omp_data['Speedup'] = merged_omp_data['Average_seq'] / merged_omp_data['Average_omp']

# Filter relevant test files for plotting
filtered_test_files = ['input100D2.inp', '200k_100.inp', '400k_100.inp', '800k_100.inp', '1600k_100.inp']
filtered_omp_data = merged_omp_data[merged_omp_data['Test File'].isin(filtered_test_files)]

# Plot setup for OpenMP data
plt.figure(figsize=(8, 6), dpi=150)
markers = ['o', 's', 'p', '^', 'x']
plot_styles = zip(filtered_test_files, markers)

# Plot OpenMP data with error bars
for test_file, marker in plot_styles:
    test_data = filtered_omp_data[filtered_omp_data['Test File'] == test_file]
    plt.errorbar(
        test_data['Threads'],
        test_data['Speedup'],
        yerr=test_data['Std Dev'],
        fmt=f'{marker}-',
        capsize=5,
        label=test_file.replace('.inp', '')
    )

# Customize the plot
plt.title('Speedup OpenMP')
plt.xlabel('Number of Threads')
plt.ylabel('Speedup (T1 / Tn)')
plt.yticks(range(0, int(filtered_omp_data['Speedup'].max()) + 5, 5))  # Fine-grained y-axis ticks
plt.xticks(test_data['Threads'].unique())  # Show all thread counts on the x-axis
plt.legend(title='', loc='upper left')
plt.grid(True)
plt.tight_layout()

# Save the plot
plt.savefig('/Users/davide/Desktop/Parallel-KMeans/Plots/speedup_openmp.png', dpi=150)

# Show the plot
plt.show()
