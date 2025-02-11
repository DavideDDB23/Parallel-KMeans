import pandas as pd
import matplotlib.pyplot as plt

# Load data
omp_mpi_data = pd.read_csv('/Users/davide/Desktop/summary_statistics_omp_mpi.csv', delimiter=';')
seq_data = pd.read_csv('/Users/davide/Desktop/summary_statistics_seq.csv')

# Define thread and test file combinations for the experiment with MPI processes = 2
scaling_processes_pairs = {
    2: '400k_100.inp',
    4: '800k_100.inp',
    8: '1600k_100.inp',
}

num_threads = 8

# Calculate efficiency for each combination
efficiency_data = []
for processes, test_file in scaling_processes_pairs.items():
    # Get parallel execution time for the current test file and thread count with MPI processes = 2
    parallel_time = omp_mpi_data[(omp_mpi_data['Test File'] == test_file) & 
                             (omp_mpi_data['Threads'] == num_threads) & 
                             (omp_mpi_data['Processes'] == processes)]['Average'].values[0]
    
    # Get serial execution time for the corresponding test file
    serial_time = seq_data[seq_data['Test File'] == test_file]['Average'].values[0]
    
    # Get standard deviation
    std_dev = omp_mpi_data[(omp_mpi_data['Test File'] == test_file) & 
                       (omp_mpi_data['Threads'] == num_threads) & 
                       (omp_mpi_data['Processes'] == processes)]['Std Dev'].values[0]
    
    # Calculate efficiency
    efficiency = serial_time / (num_threads * processes * parallel_time)
    
    # Append data for plotting
    efficiency_data.append((processes, efficiency, test_file, std_dev))

# Convert to DataFrame
efficiency_df = pd.DataFrame(efficiency_data, columns=['Processes', 'Efficiency', 'Test File', 'Std Dev'])

# Scale the standard deviation by 10 for all values
efficiency_df['Scaled Std Dev'] = efficiency_df['Std Dev'] / 50

# Ensure that efficiency is capped at 1
efficiency_df['Efficiency'] = efficiency_df['Efficiency'].apply(lambda e: min(e, 1))

# Define markers
markers = ['o', 's', 'p', '^', 'x']

# Plot setup
plt.figure(figsize=(8, 6), dpi=150)

# Plot data using `efficiency_df`
for i, row in efficiency_df.iterrows():
    plt.errorbar(
        row['Processes'],
        row['Efficiency'],
        yerr=row['Scaled Std Dev'],
        fmt=markers[i],
        capsize=5,
        label=row['Test File'].replace('.inp', '')
    )

# Plot the line connecting the points
plt.plot(
    efficiency_df['Processes'],
    efficiency_df['Efficiency'],
    color='green',
    linestyle='-',
    alpha=0.7
)

# Customize the plot
plt.title(f'Weak Scaling, Fixed OpenMP Threads = {num_threads}')
plt.xlabel('Number of Processes')
plt.ylabel('Efficiency (T1 / (p * Tp))')
plt.ylim(0, 1.01)  # Set y-axis limit to [0, 1.01]
plt.xticks(efficiency_df['Processes'])
plt.grid(True)
plt.tight_layout()

# Add legend and save the plot
plt.legend(loc='lower right')

plt.savefig('/Users/davide/Desktop/Parallel-KMeans/Plots/mpi_omp/weak_scaling_threads_omp_mpi_8.png', dpi=150)

plt.show()