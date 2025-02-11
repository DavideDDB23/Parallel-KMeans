import pandas as pd
import matplotlib.pyplot as plt

# Load sequential data
seq_data = pd.read_csv('/Users/davide/Desktop/summary_statistics_seq.csv')

# Set the condition for filtering files
plot_numeric_files = True  # Change this to False to plot the other set of files

# Apply filtering based on the condition
if plot_numeric_files:
    # Plot files starting with a number and 'input100D2'
    filtered_data = seq_data[seq_data['Test File'].str.match(r'^\d|input100D2')]
else:
    # Plot files with specific names not starting with numbers
    filtered_data = seq_data[seq_data['Test File'].isin(['input100D.inp', 'input20D.inp', 'input10D.inp', 'input2D2.inp', 'input2D.inp'])]

# Prepare data for box plot
boxplot_data_seq = []
test_files_seq = filtered_data['Test File'].unique()

# Collect relevant statistics for each test file
for test_file in test_files_seq:
    file_data = filtered_data[filtered_data['Test File'] == test_file]
    boxplot_data_seq.append([
        file_data['Min'].values[0],
        file_data['Q1'].values[0],
        file_data['Median'].values[0],
        file_data['Q3'].values[0],
        file_data['Max'].values[0]
    ])

# Plot the box plot
plt.figure(figsize=(10, 6), dpi=150)
plt.boxplot(boxplot_data_seq, vert=True, patch_artist=True, labels=[tf.replace('.inp', '') for tf in test_files_seq])

# Customize the plot
plt.title('Runtime Sequential')
plt.ylabel('Execution Time (s)')
plt.xticks(rotation=45)
plt.grid(True, axis='y', linestyle='--', alpha=0.7)

# Save and show the plot
plt.tight_layout()
output_file = '/Users/davide/Desktop/Parallel-KMeans/Plots/sequential_boxplot_1.png'
plt.savefig(output_file, dpi=150)
plt.show()