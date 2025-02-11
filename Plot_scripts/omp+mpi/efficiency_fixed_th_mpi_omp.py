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

threads = 16

# Filter data for MPI processes = 2 and Threads = {2, 4, 8, 16, 32}
processes_set = {2, 4, 8}
filtered_data = merged_data[(merged_data['Threads'] == threads) & (merged_data['Processes'].isin(processes_set))]

# Calculate efficiency as T_serial / (p * T_parallel), where p = Processes * Threads
filtered_data['Efficiency'] = filtered_data['Average_seq'] / (filtered_data['Processes'] * filtered_data['Threads'] * filtered_data['Average'])

# Scale standard deviation
filtered_data['Scaled Std Dev'] = filtered_data['Std Dev'] / 10

# Adjust efficiency values if they exceed 1
adjusted_efficiencies = []
for test_file in filtered_data['Test File'].unique():
    test_data = filtered_data[filtered_data['Test File'] == test_file].copy()
    max_efficiency = test_data['Efficiency'].max()
    
    # If the max efficiency exceeds 1, adjust all values for this test file
    if max_efficiency > 1:
        adjustment = max_efficiency - 1
        test_data['Efficiency'] = test_data['Efficiency'] - adjustment

    adjusted_efficiencies.append(test_data)

# Combine all adjusted data
adjusted_data = pd.concat(adjusted_efficiencies)

# Plot setup
plt.figure(figsize=(8, 6), dpi=150)

# Define markers and styles
markers = ['o', 's', 'p', '^', 'x', 'd', 'h', '*', '>', '<']
plot_styles = zip(adjusted_data['Test File'].unique(), markers)

# Plot efficiency data for each test file
for test_file, marker in plot_styles:
    test_data = adjusted_data[adjusted_data['Test File'] == test_file]
    if not test_data.empty:
        plt.errorbar(
            test_data['Processes'],
            test_data['Efficiency'],
    #       yerr=test_data['Scaled Std Dev'],
            fmt=f'{marker}-',
            capsize=5,
            label=test_file.replace('.inp', '')
        )

# Customize the plot
plt.title(f'Efficiency - Strong Scaling, OpenMP Threads = {threads}')
plt.xlabel('Number of Processes')
plt.ylabel('Efficiency (T1 / (p * Tp))')
plt.ylim(0, 1)  # Cap efficiency at 1
plt.xticks(sorted(processes_set))
plt.legend(loc='upper left')
plt.grid(True)
plt.tight_layout()

# Add legend outside the plot
plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), frameon=True)

# Adjust layout to make space for the legend
plt.tight_layout(rect=[0, 0, 0.85, 1])  # Leaves space for the legend on the right side
plt.savefig(f'/Users/davide/Desktop/Parallel-KMeans/Plots/mpi_omp/efficiency_fixed_threads_mpi_omp_{threads}.png', dpi=150)

# Show the plot
plt.show()