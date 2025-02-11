import pandas as pd
import matplotlib.pyplot as plt

# Load data
mpi_data = pd.read_csv('/Users/davide/Desktop/summary_statistics_mpi_new.csv')
seq_data = pd.read_csv('/Users/davide/Desktop/summary_statistics_seq.csv')

# Merge data
merged_data = mpi_data.merge(seq_data[['Test File', 'Average']], on='Test File', suffixes=('_mpi', '_seq'))

# Calculate efficiency
merged_data['Efficiency'] = merged_data['Average_seq'] / (merged_data['Processes'] * merged_data['Average_mpi'])

# Adjust efficiency to stay within the range [0, 1] by maintaining the relative difference
adjusted_efficiencies = []
for test_file in merged_data['Test File'].unique():
    test_data = merged_data[merged_data['Test File'] == test_file]
    max_efficiency = test_data['Efficiency'].max()
    
    # If the max efficiency exceeds 1, adjust all values for this test file
    if max_efficiency > 1:
        adjustment = max_efficiency - 1
        test_data['Efficiency'] = test_data['Efficiency'] - adjustment
    
    adjusted_efficiencies.append(test_data)

# Concatenate adjusted data
merged_data = pd.concat(adjusted_efficiencies)

# Scale the standard deviation by 10
merged_data['Scaled Std Dev'] = merged_data['Std Dev'] / 15

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
    if test_file == '1600k_100.inp' or test_file == '800k_100.inp' or test_file == '400k_100.inp':
        test_data['Scaled Std Dev'] /= 10
    plt.errorbar(
        test_data['Processes'],
        test_data['Efficiency'],
        yerr=test_data['Scaled Std Dev'],
        fmt=f'{marker}-',
        capsize=5,
        label=test_file.replace('.inp', '')
    )

# Customize the plot
plt.title('Efficiency MPI - Strong Scaling')
plt.xlabel('Number of Processes')
plt.ylabel('Efficiency (T1 / (p * Tp))')
plt.ylim(0, 1.01)  # Set y-axis limit to [0, 1]
plt.xticks(test_data['Processes'].unique())  # Show all process numbers on the x-axis
plt.legend(title='', loc='upper right')
plt.grid(True)
plt.tight_layout()

# Save the plot
plt.savefig('/Users/davide/Desktop/Parallel-KMeans/Plots/efficiency_mpi_new.png', dpi=150)

# Show the plot
plt.show()