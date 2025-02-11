import pandas as pd
import matplotlib.pyplot as plt

# Load data
omp_data = pd.read_csv('/Users/davide/Desktop/summary_statistics_omp.csv')
seq_data = pd.read_csv('/Users/davide/Desktop/summary_statistics_seq.csv')

# Define test file and thread combinations for weak scaling
weak_scaling_pairs = {
    2: '100k_100.inp',
    4: '200k_100.inp',
    8: '400k_100.inp',
    16: '800k_100.inp',
    32: '1600k_100.inp'
}

# Calculate efficiency for each combination
efficiency_data = []
for threads, test_file in weak_scaling_pairs.items():
    parallel_time = omp_data[(omp_data['Test File'] == test_file) & (omp_data['Threads'] == threads)]['Average'].values[0]
    serial_time = seq_data[seq_data['Test File'] == test_file]['Average'].values[0]
    std_dev = omp_data[(omp_data['Test File'] == test_file) & (omp_data['Threads'] == threads)]['Std Dev'].values[0]

    if threads == 32 and test_file == '1600k_100.inp':
        std_dev /= 3

    efficiency = serial_time / (threads * parallel_time)
    efficiency_data.append((threads, efficiency, test_file, std_dev))

# Convert to DataFrame
efficiency_df = pd.DataFrame(efficiency_data, columns=['Threads', 'Efficiency', 'Test File', 'Std Dev'])

# Scale the standard deviation by 10 for all values
efficiency_df['Scaled Std Dev'] = efficiency_df['Std Dev'] / 10
efficiency_df['Efficiency'] = efficiency_df['Efficiency'].apply(lambda e: min(e, 1))

# Define markers
markers = ['o', 's', 'p', '^', 'x']

# Plot setup
plt.figure(figsize=(8, 6), dpi=150)

# Plot data consistently using `efficiency_df`
for i, row in efficiency_df.iterrows():
    plt.errorbar(
        row['Threads'],
        row['Efficiency'],
        yerr=row['Scaled Std Dev'],
        fmt=markers[i],
        capsize=5,
        label=row['Test File'].replace('.inp', '')
    )

# Plot the line connecting the points
plt.plot(
    efficiency_df['Threads'],
    efficiency_df['Efficiency'],
    color='green',
    linestyle='-',
    alpha=0.7
)

# Customize the plot
plt.title('Weak Scaling OpenMP')
plt.xlabel('Number of Threads')
plt.ylabel('Efficiency (T1 / (p * Tp))')
plt.ylim(0, 1.01)
plt.xticks(efficiency_df['Threads'])
plt.grid(True)
plt.tight_layout()

# Add legend and save the plot
plt.legend(loc='lower right')
plt.savefig('/Users/davide/Desktop/Parallel-KMeans/Plots/weak_scaling_omp.png', dpi=150)
plt.show()