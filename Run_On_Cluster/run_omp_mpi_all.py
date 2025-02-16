#!/usr/bin/env python3
import os
import subprocess
import re
import math
import csv

# ---------------------------
# Helper functions for statistics
# ---------------------------
def compute_median(data):
    """Compute median from a sorted list."""
    n = len(data)
    if n == 0:
        return None
    if n % 2 == 1:
        return data[n // 2]
    else:
        return (data[n // 2 - 1] + data[n // 2]) / 2

def compute_quartiles(sorted_data):
    """
    Given a sorted list, compute Q1, median, and Q3.
    For an odd-length list the median is excluded from the halves.
    """
    n = len(sorted_data)
    median = compute_median(sorted_data)
    if n % 2 == 0:
        lower_half = sorted_data[: n // 2]
        upper_half = sorted_data[n // 2:]
    else:
        lower_half = sorted_data[: n // 2]
        upper_half = sorted_data[n // 2 + 1:]
    Q1 = compute_median(lower_half) if lower_half else median
    Q3 = compute_median(upper_half) if upper_half else median
    return Q1, median, Q3

def compute_summary_stats(data):
    """Compute and return summary statistics in a dictionary."""
    n = len(data)
    if n == 0:
        return None
    sorted_data = sorted(data)
    avg = sum(data) / n
    variance = sum((x - avg) ** 2 for x in data) / n
    std_dev = math.sqrt(variance)
    Q1, median, Q3 = compute_quartiles(sorted_data)
    return {
        "min": sorted_data[0],
        "Q1": Q1,
        "median": median,
        "Q3": Q3,
        "max": sorted_data[-1],
        "average": avg,
        "std_dev": std_dev
    }

# ---------------------------
# Main processing
# ---------------------------
def main():
    # --- Configuration ---
    test_files = [
        "100k_100.inp",
        "200k_100.inp",
        "400k_100.inp",
        "800k_100.inp",
        "1600k_100.inp"
    ]

    # Define the specific combinations of MPI processes and OpenMP threads to test
    combinations = [
        (2, 32),  # MPI: 2, OMP: 32
        (4, 16),  # MPI: 4, OMP: 16
        (8, 8),   # MPI: 8, OMP: 8
        (16, 4),  # MPI: 16, OMP: 4
        (32, 2)   # MPI: 32, OMP: 2
    ]

    compare_files = [
        "res100D2",
        "res100D",
        "res100k_100",
        "res200k_100",
        "res400k_100",
        "res800k_100",
        "res1600k_100"
    ]

    try:
        num_runs = int(input("Enter the number of runs per test file per combination: "))
    except ValueError:
        print("Invalid input. Please enter an integer.")
        return

    # Parameters passed to the executable
    cluster    = "20"
    ITERATIONS = "5000"
    CHANGES    = "1"
    THRESHOLD  = "0.0001"

    # Path to the executable (ensure it is compiled and available)
    executable = "./KMEANS_omp_mpi.out"

    # Directories for test files and logs; create directories if they do not exist.
    test_files_dir = "/home/deblasio_2082600/Parallel-KMeans/test_files"
    logs_dir       = "/home/deblasio_2082600/Parallel-KMeans/logs"
    os.makedirs(logs_dir, exist_ok=True)
    results_dir    = os.path.join("results", "omp_mpi")
    os.makedirs(results_dir, exist_ok=True)

    sequential_dir = os.path.join("results", "seq")

    results = []

    # Loop over each test file and each combination of processes and threads
    for test_file in test_files:
        input_path = os.path.join(test_files_dir, test_file)
        test_base  = os.path.splitext(test_file)[0]

        for proc, th in combinations:
            for run in range(1, num_runs + 1):
                print("========================================")
                print(
                    f"Submitting job for test file '{test_file}', processes = {proc}, threads = {th} "
                    f"(run {run}/{num_runs})..."
                )

                # Define job-specific file names
                condor_file = f"job_omp_mpi_{test_base}_p{proc}_t{th}_{run}.sub"
                out_file    = os.path.join(logs_dir, f"out_omp_mpi_{test_base}_p{proc}_t{th}_{run}.txt")
                res_file    = os.path.join(logs_dir, f"res_omp_mpi_{test_base}_p{proc}_t{th}_{run}.txt")
                log_file    = os.path.join(logs_dir, f"log_omp_mpi_{test_base}_p{proc}_t{th}_{run}.txt")
                err_file    = os.path.join(logs_dir, f"err_omp_mpi_{test_base}_p{proc}_t{th}_{run}.txt")

                # Build the Condor submission file content
                condor_content = f"""universe = parallel
log = {log_file}
output = {out_file}
error = {err_file}
executable = openmpiscript.sh
arguments = {executable} {input_path} {cluster} {ITERATIONS} {CHANGES} {THRESHOLD} {res_file} {th}
should_transfer_files = YES
transfer_input_files = {executable}
when_to_transfer_output = on_exit_or_evict
machine_count = {proc}
request_cpus = 32
getenv = True
queue
"""
                with open(condor_file, "w") as f:
                    f.write(condor_content)

                # Submit the job
                try:
                    submit_proc = subprocess.run(
                        ["condor_submit", condor_file],
                        capture_output=True, text=True, check=True
                    )
                except subprocess.CalledProcessError as e:
                    print(
                        f"Error submitting job (run {run} for {test_file} with "
                        f"{proc} processes, {th} threads): {e}"
                    )
                    continue

                submit_output = submit_proc.stdout
                cluster_id_match = re.search(r"submitted to cluster\s+(\d+)", submit_output)
                if cluster_id_match:
                    cluster_id = cluster_id_match.group(1)
                    print(f"Job submitted to cluster {cluster_id}.")
                else:
                    print("Warning: Could not extract ClusterId from condor_submit output.")
                    continue

                # Wait for the job to complete
                print(f"Waiting for job (ClusterId {cluster_id}) to complete...")
                try:
                    subprocess.run(["condor_wait", log_file], check=True)
                except subprocess.CalledProcessError as e:
                    print(f"Error waiting for job {cluster_id}: {e}")
                    continue

                # Parse the output file for computation time
                if not os.path.isfile(res_file):
                    print(f"Output file '{res_file}' not found; cannot parse computation time.")
                    continue

                with open(res_file, "r") as f:
                    lines = f.readlines()

                time_value = None
                computation_line_index = None
                for idx in range(len(lines) - 1, -1, -1):
                    if "Computation:" in lines[idx]:
                        match = re.search(r"Computation:\s*([\d\.]+)\s*seconds", lines[idx])
                        if match:
                            try:
                                time_value = float(match.group(1))
                                print("Computation time:", time_value)
                            except ValueError:
                                pass
                        computation_line_index = idx
                        break

                if time_value is not None:
                    print(
                        f"Run {run} for '{test_file}' with {proc} processes and {th} threads: "
                        f"Computation time = {time_value} seconds"
                    )
                else:
                    print(f"Could not parse a valid computation time from '{res_file}'.")

                if computation_line_index is not None:
                    del lines[computation_line_index]
                    with open(res_file, "w") as f:
                        f.writelines(lines)

                # Save the run result
                results.append({
                    "test_file": test_file,
                    "processes": proc,
                    "threads": th,
                    "run": run,
                    "time": time_value,
                    "match": False  # Default to False for now
                })

                os.remove(condor_file)

    # Write raw data to CSV
    raw_csv_file = f"computation_times_omp_mpi_combinations_2.csv"
    try:
        with open(raw_csv_file, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Test File", "Processes", "Threads", "Run", "Computation Time (s)", "Matches Sequential"])
            for row in results:
                writer.writerow([
                    row["test_file"],
                    row["processes"],
                    row["threads"],
                    row["run"],
                    row["time"],
                    "Yes" if row["match"] else "No"
                ])
        print(f"\nRaw data table written to '{raw_csv_file}'.")
    except Exception as e:
        print(f"Error writing raw data CSV file: {e}")

    # ---------------------------
    # Compute and display summary statistics for each (test_file, processes, threads).
    # ---------------------------
    summary = {}  # key = (test_file, processes, threads), value = list of run dictionaries
    for row in results:
        key = (row["test_file"], row["processes"], row["threads"])
        summary.setdefault(key, []).append(row)

    summary_csv_file = f"summary_statistics_omp_mpi_2.csv"
    try:
        with open(summary_csv_file, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            # Added "Threads" column in the summary
            writer.writerow([
                "Test File", "Processes", "Threads",
                "Min", "Q1", "Median", "Q3", "Max",
                "Average", "Std Dev", "Matches Sequential"
            ])

            print("\nSummary statistics per test file, process count, and thread count:")
            for (test_file, proc, th), run_list in summary.items():
                times = [r["time"] for r in run_list if r["time"] is not None]
                if not times:
                    continue
                stats = compute_summary_stats(times)
                # Mark "Yes" if all runs in the group matched the sequential file.
                all_match = all(r["match"] for r in run_list)
                match_str = "Yes" if all_match else "No"
                print(
                    f"  {test_file} (Processes: {proc}, Threads: {th}): "
                    f"{len(times)} runs, "
                    f"min = {stats['min']:.4f}, Q1 = {stats['Q1']:.4f}, "
                    f"median = {stats['median']:.4f}, Q3 = {stats['Q3']:.4f}, "
                    f"max = {stats['max']:.4f}, average = {stats['average']:.4f}, "
                    f"std = {stats['std_dev']:.4f}, Matches Sequential: {match_str}"
                )
                writer.writerow([
                    test_file,
                    proc,
                    th,
                    f"{stats['min']:.4f}",
                    f"{stats['Q1']:.4f}",
                    f"{stats['median']:.4f}",
                    f"{stats['Q3']:.4f}",
                    f"{stats['max']:.4f}",
                    f"{stats['average']:.4f}",
                    f"{stats['std_dev']:.4f}",
                    match_str
                ])
        print(f"\nSummary statistics written to '{summary_csv_file}'.")
    except Exception as e:
        print(f"Error writing summary CSV file: {e}")

if __name__ == "__main__":
    main()