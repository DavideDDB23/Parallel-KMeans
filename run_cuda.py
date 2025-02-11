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
    """
    Compute and return summary statistics in a dictionary:
      - min, Q1, median, Q3, max, average, and standard deviation.
    """
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
        "input100D2.inp",
        "input100D.inp",
        "input20D.inp",
        "input10D.inp",
        "input2D.inp",
        "100k_100.inp",
        "200k_100.inp",
        "400k_100.inp",
        "800k_100.inp",
        "1600k_100.inp"
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
        num_runs = int(input("Enter the number of runs per test file: "))
    except ValueError:
        print("Invalid input. Please enter an integer.")
        return

    # Parameters passed to the executable (adjust these as needed)
    cluster = "20"         # e.g., number of clusters
    ITERATIONS = "5000"
    CHANGES = "1"
    THRESHOLD = "0.0001"

    # Path to the executable (ensure it is compiled and available)
    executable = "./KMEANS_cuda.out"

    # Directories for test files and logs; create directories if they do not exist.
    test_files_dir = "/home/deblasio_2082600/Parallel-KMeans/test_files"
    logs_dir = "/home/deblasio_2082600/Parallel-KMeans/logs"
    os.makedirs(logs_dir, exist_ok=True)
    results_dir = os.path.join("results", "cuda")
    os.makedirs(results_dir, exist_ok=True)

    # Directory for sequential (gold standard) files.
    sequential_dir = os.path.join("results", "seq")

    # This list will collect the raw results.
    results = []

    # Loop over each test file, each process count, and perform the desired number of runs.
    for test_file in test_files:
        input_path = os.path.join(test_files_dir, test_file)
        test_base = os.path.splitext(test_file)[0]
        for run in range(1, num_runs + 1):
            print("========================================")
            print(f"Submitting job for test file '{test_file}', (run {run}/{num_runs})...")

            # Define job-specific file names.
            condor_file = f"job_cuda_{test_base}_{run}.sub"
            out_file = os.path.join(logs_dir, f"out_cuda_{test_base}_{run}.txt")
            res_file = os.path.join(logs_dir, f"res_cuda_{test_base}_{run}.txt")
            log_file = os.path.join(logs_dir, f"log_cuda_{test_base}_{run}.txt")
            err_file = os.path.join(logs_dir, f"err_cuda_{test_base}_{run}.txt")

            # Build the Condor submission file content.
            condor_content = f"""universe = vanilla
log = {log_file}
output = {out_file}
error = {err_file}
executable = {executable}
arguments = {input_path} {cluster} {ITERATIONS} {CHANGES} {THRESHOLD} {res_file}
request_gpus = 1
getenv = True
queue
"""
            # Write the temporary Condor submission file.
            with open(condor_file, "w") as f:
                f.write(condor_content)

            # Submit the job via condor_submit.
            try:
                submit_proc = subprocess.run(
                    ["condor_submit", condor_file],
                    capture_output=True, text=True, check=True
                )
            except subprocess.CalledProcessError as e:
                print(f"Error submitting job (run {run} for {test_file}): {e}")
                continue

            submit_output = submit_proc.stdout
            cluster_id_match = re.search(r"submitted to cluster\s+(\d+)", submit_output)
            if cluster_id_match:
                cluster_id = cluster_id_match.group(1)
                print(f"Job submitted to cluster {cluster_id}.")
            else:
                print("Warning: Could not extract ClusterId from condor_submit output.")
                continue

            # Wait for the job to complete.
            print(f"Waiting for job (ClusterId {cluster_id}) to complete...")
            try:
                subprocess.run(["condor_wait", log_file], check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error waiting for job {cluster_id}: {e}")
                continue

            if not os.path.isfile(out_file):
                print(f"Output file '{out_file}' not found; cannot parse computation time.")
                continue

            time_value = None
            with open(out_file, "r") as f:
                for line in f:
                    if "Computation:" in line:
                        match = re.search(r"Computation:\s*([\d\.]+)\s*seconds", line)
                        if match:
                            try:
                                time_value = float(match.group(1))
                            except ValueError:
                                pass
                        break

            if time_value is not None:
                print(f"Run {run} for '{test_file}': Computation time = {time_value} seconds")
            else:
                print(f"Could not parse a valid computation time from '{out_file}'.")

            sequential_match = False
            if os.path.isfile(res_file):
                if test_file.startswith("input") and test_file.endswith(".inp"):
                    core = test_file[len("input"):-len(".inp")]
                    expected_file_name = "res" + core
                else:
                    core = test_file[:-len(".inp")]
                    expected_file_name = "res" + core

                if expected_file_name:
                    if expected_file_name not in compare_files:
                        print(f"Warning: Derived expected file '{expected_file_name}' is not in the compare_files list.")

                    compare_file_path = os.path.join(sequential_dir, expected_file_name + ".txt")
                    if os.path.isfile(compare_file_path):
                        diff_proc = subprocess.run(["diff", res_file, compare_file_path],
                                                   capture_output=True, text=True)
                        if diff_proc.returncode == 0:
                            print("Output matches the expected sequential file.")
                            sequential_match = True
                        else:
                            print("Output does NOT match the expected sequential file.")
                            print("Diff output:")
                            print(diff_proc.stdout)
                    else:
                        print(f"Expected sequential file '{compare_file_path}' not found.")
                else:
                    print("Could not determine expected sequential file based on test file name.")
            else:
                print(f"Result file '{res_file}' not found for comparison.")

            results.append({
                "test_file": test_file,
                "run": run,
                "time": time_value,
                "match": sequential_match
            })

            os.remove(condor_file)

    # ---------------------------
    # Write the raw data into a CSV table.
    # ---------------------------
    raw_csv_file = "computation_times_cuda.csv"
    try:
        with open(raw_csv_file, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Test File", "Run", "Computation Time (s)", "Matches Sequential"])
            for row in results:
                writer.writerow([row["test_file"], row["run"], row["time"],
                                 "Yes" if row["match"] else "No"])
        print(f"\nRaw data table written to '{raw_csv_file}'.")
    except Exception as e:
        print(f"Error writing raw data CSV file: {e}")

    # ---------------------------
    # Compute and display summary statistics for each test file and process count.
    # ---------------------------
    summary = {}
    for row in results:
        key = row["test_file"]
        summary.setdefault(key, []).append(row)

    summary_csv_file = "summary_statistics_cuda.csv"
    try:
        with open(summary_csv_file, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Test File", "Min", "Q1", "Median", "Q3", "Max",
                             "Average", "Std Dev", "Matches Sequential"])
            print("\nSummary statistics per test file:")
            for test_file, run_list in summary.items():
                times = [r["time"] for r in run_list if r["time"] is not None]
                if not times:
                    continue
                stats = compute_summary_stats(times)
                all_match = all(r["match"] for r in run_list)
                match_str = "Yes" if all_match else "No"
                print(f"  {test_file}: "
                      f"{len(times)} runs, min = {stats['min']:.4f}, Q1 = {stats['Q1']:.4f}, "
                      f"median = {stats['median']:.4f}, Q3 = {stats['Q3']:.4f}, max = {stats['max']:.4f}, "
                      f"average = {stats['average']:.4f}, std = {stats['std_dev']:.4f}, "
                      f"Matches Sequential: {match_str}")
                writer.writerow([test_file,
                                 f"{stats['min']:.4f}",
                                 f"{stats['Q1']:.4f}",
                                 f"{stats['median']:.4f}",
                                 f"{stats['Q3']:.4f}",
                                 f"{stats['max']:.4f}",
                                 f"{stats['average']:.4f}",
                                 f"{stats['std_dev']:.4f}",
                                 match_str])
        print(f"\nSummary statistics written to '{summary_csv_file}'.")
    except Exception as e:
        print(f"Error writing summary CSV file: {e}")

if __name__ == "__main__":
    main()