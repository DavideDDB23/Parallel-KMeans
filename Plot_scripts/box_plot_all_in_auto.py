import pandas as pd
import matplotlib.pyplot as plt

# Load data
file_paths = {
    "Sequential": "/Users/davide/Desktop/summary_statistics_seq.csv",
    "MPI": "/Users/davide/Desktop/summary_statistics_mpi_new.csv",
    "OpenMP": "/Users/davide/Desktop/summary_statistics_omp.csv",
    "MPI+OpenMP": "/Users/davide/Desktop/summary_statistics_omp_mpi.csv",
    "CUDA": "/Users/davide/Desktop/summary_statistics_cuda.csv"
}

# Read data with appropriate delimiters
data_frames = {
    "Sequential": pd.read_csv(file_paths["Sequential"]),
    "MPI": pd.read_csv(file_paths["MPI"]),
    "OpenMP": pd.read_csv(file_paths["OpenMP"]),
    "MPI+OpenMP": pd.read_csv(file_paths["MPI+OpenMP"], delimiter=';'),
    "CUDA": pd.read_csv(file_paths["CUDA"])
}

# Selected test file
selected_file = 'input100D.inp'

# Determine optimal MPI processes based on the lowest average time
mpi_df = data_frames["MPI"]
mpi_filtered = mpi_df[mpi_df['Test File'] == selected_file]
if not mpi_filtered.empty:
    mpi_processes = mpi_filtered.loc[mpi_filtered['Average'].idxmin(), 'Processes']
else:
    mpi_processes = "N/A"

# Determine optimal OpenMP threads based on the lowest average time
omp_df = data_frames["OpenMP"]
omp_filtered = omp_df[omp_df['Test File'] == selected_file]
if not omp_filtered.empty:
    omp_threads = omp_filtered.loc[omp_filtered['Average'].idxmin(), 'Threads']
else:
    omp_threads = "N/A"

# Determine optimal MPI+OpenMP configuration based on the lowest average time
mpi_omp_df = data_frames["MPI+OpenMP"]
mpi_omp_filtered = mpi_omp_df[mpi_omp_df['Test File'] == selected_file]
if not mpi_omp_filtered.empty:
    best_mpi_omp = mpi_omp_filtered.loc[mpi_omp_filtered['Average'].idxmin()]
    mpi_omp_threads = best_mpi_omp['Threads']
    mpi_omp_processes = best_mpi_omp['Processes']
else:
    mpi_omp_threads, mpi_omp_processes = "N/A", "N/A"

# Prepare boxplot data
boxplot_data = []
labels = []

# Iterate in the specified order
for key in ["Sequential", "MPI", "OpenMP", "MPI+OpenMP", "CUDA"]:
    df = data_frames[key]

    # Filter based on test file
    if 'Test File' in df.columns:
        filtered_df = df[df['Test File'] == selected_file]
    else:
        continue  # Skip if 'Test File' column is missing

    # Additional filtering for MPI and OpenMP implementations
    config = ""
    if key == "MPI" and 'Processes' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['Processes'] == mpi_processes]
        config = f" \n{mpi_processes} Processes"
    elif key == "OpenMP" and 'Threads' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['Threads'] == omp_threads]
        config = f" \n{omp_threads} Threads"
    elif key == "MPI+OpenMP" and 'Threads' in filtered_df.columns and 'Processes' in filtered_df.columns:
        filtered_df = filtered_df[
            (filtered_df['Threads'] == mpi_omp_threads) & 
            (filtered_df['Processes'] == mpi_omp_processes)
        ]
        config = f" \n{mpi_omp_processes} Processes \n{mpi_omp_threads} Threads"

    # Ensure data exists before processing
    if not filtered_df.empty:
        boxplot_data.append([
            filtered_df['Min'].values[0],
            filtered_df['Q1'].values[0],
            filtered_df['Median'].values[0],
            filtered_df['Q3'].values[0],
            filtered_df['Max'].values[0]
        ])
        labels.append(key + config)

# Plot the box plot
plt.figure(figsize=(10, 6), dpi=150)
plt.boxplot(boxplot_data, vert=True, patch_artist=True, labels=labels)

# Customize the plot
plt.title(f'Runtime {selected_file}')
plt.ylabel('Execution Time (s)')
plt.xticks(rotation=30, ha='right')
plt.grid(True, axis='y', linestyle='--', alpha=0.7)

# Save and show the plot
plt.tight_layout()
plt.savefig(f'/Users/davide/Desktop/Parallel-KMeans/Plots/boxplot_all_in_auto_{selected_file}.png', dpi=150)
plt.show()