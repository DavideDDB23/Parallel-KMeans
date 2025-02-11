import pandas as pd
import matplotlib.pyplot as plt

# Load data
cuda_data = pd.read_csv("/Users/davide/Desktop/summary_statistics_cuda.csv")

# Define whether to use specific test files or all
use_selected_files = False  # Change to False to include other files
selected_files = ['input100D.inp', 'input10D.inp', 'input20D.inp', 'input2D.inp']

# Filter the data based on the selection
if use_selected_files:
    filtered_data = cuda_data[cuda_data['Test File'].isin(selected_files)]
else:
    filtered_data = cuda_data[~cuda_data['Test File'].isin(selected_files)]

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
plt.title('Runtime CUDA')
plt.ylabel('Execution Time (s)')
plt.xticks(rotation=45)
plt.grid(True, axis='y', linestyle='--', alpha=0.7)

# Save and show the plot
plt.tight_layout()
plt.savefig(f'/Users/davide/Desktop/Parallel-KMeans/Plots/cuda/cuda_boxplot{use_selected_files}.png', dpi=150)
plt.show()
