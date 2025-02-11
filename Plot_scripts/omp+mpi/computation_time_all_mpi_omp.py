import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV file (adjust the path if needed)
df = pd.read_csv('/Users/davide/Desktop/summary_statistics_omp_mpi_all.csv', delimiter=';')

# Define the order of MPI and OMP combinations
combination_order = ['MPI: 2\nOMP: 32', 'MPI: 4\nOMP: 16', 'MPI: 8\nOMP: 8', 'MPI: 16\nOMP: 4', 'MPI: 32\nOMP: 2']

# Create a new column with labels in the format "MPI: X\nOMP: Y"
df['Combination_Label'] = df.apply(lambda row: f'MPI: {row["Processes"]}\nOMP: {row["Threads"]}', axis=1)

# Filter and sort combinations to match the desired order
df = df[df['Combination_Label'].isin(combination_order)]
df['Combination_Label'] = pd.Categorical(df['Combination_Label'], categories=combination_order, ordered=True)

# Get unique test files and markers
test_files = df['Test File'].unique()
markers = ['o', 's', 'D', 'X', '^', 'v', 'p', '*', 'P']

# Set plot style
sns.set(style="darkgrid")

# Create the scatter plot
plt.figure(figsize=(10, 6), dpi=150)
for i, test_file in enumerate(test_files):
    subset = df[df['Test File'] == test_file]
    plt.scatter(
        subset['Combination_Label'],
        subset['Average'],
        label=test_file.split('.')[0],  # Shortened label without file extension
        marker=markers[i % len(markers)],
        s=100
    )

# Add labels and title
plt.ylabel('Execution Time (s)')
plt.title('Average Runtime doubling MPI processes and halving OpenMP threads')
plt.xticks(rotation=45, ha='right')
plt.grid(True)

# Add legend outside the plot
plt.legend(title='Test Files', loc='upper left', bbox_to_anchor=(1.05, 1), frameon=True)

# Adjust layout to make space for the legend
plt.tight_layout(rect=[0, 0, 0.85, 1])

# Save and show the plot
plt.savefig('/Users/davide/Desktop/Parallel-KMeans/Plots/mpi_omp_runtime_all.png', dpi=150, bbox_inches='tight')
plt.show()