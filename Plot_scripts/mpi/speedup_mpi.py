import pandas as pd
import matplotlib.pyplot as plt

# Load data
mpi_data = pd.read_csv('/Users/davide/Desktop/summary_statistics_mpi_new.csv')
seq_data = pd.read_csv('/Users/davide/Desktop/summary_statistics_seq.csv')

# Merge data
merged_data = mpi_data.merge(seq_data[['Test File', 'Average']], on='Test File', suffixes=('_mpi', '_seq'))
merged_data['Speedup'] = merged_data['Average_seq'] / merged_data['Average_mpi']

# Filter relevant test files
filtered_test_files = ['input100D2.inp', '200k_100.inp', '400k_100.inp', '800k_100.inp', '1600k_100.inp']
filtered_data = merged_data[merged_data['Test File'].isin(filtered_test_files)]

# Plot setup
plt.figure(figsize=(8, 6), dpi=150)
markers = ['o', 's', 'p', '^', 'x']
plot_styles = zip(filtered_test_files, markers)

# Plot data with error bars
for test_file, marker in plot_styles:
    test_data = filtered_data[filtered_data['Test File'] == test_file]
    plt.errorbar(
        test_data['Processes'],
        test_data['Speedup'],
        yerr=test_data['Std Dev'],
        fmt=f'{marker}-',
        capsize=5,
        label=test_file.replace('.inp', '')
    )

# Customize the plot
plt.title('Speedup MPI')
plt.xlabel('Number of Processes')
plt.ylabel('Speedup (T1 / Tn)')
plt.yticks(range(0, int(filtered_data['Speedup'].max()) + 5, 5))  # Fine-grained y-axis ticks
plt.xticks(test_data['Processes'].unique())  # Show all process numbers on the x-axis
plt.legend(title='', loc='upper left')
plt.grid(True)
plt.tight_layout()

# Save the plot
plt.savefig('/Users/davide/Desktop/Parallel-KMeans/Plots/speedup_mpi_new.png', dpi=150)

# Show the plot
plt.show()
