#!/bin/bash

# Prompt the user for the number of runs
read -p "Enter the number of runs: " num_runs

# Array to hold each run's time
declare -a times

# Loop over the desired number of runs
for i in $(seq 1 $num_runs); do
  echo "==============================="
  echo "Submitting job $i/$num_runs..."
  
  # Submit the Condor job
  submit_output=$(condor_submit job_seq.sub 2>&1)
  rc=$?
  
  if [ $rc -ne 0 ]; then
    echo "condor_submit failed (exit code $rc)."
    echo "Output from condor_submit:"
    echo "$submit_output"
    break
  fi
  
  # -------------------------------------------------------------------
  # Extract the ClusterId from the condor_submit output
  # -------------------------------------------------------------------
  cluster_id=$(echo "$submit_output" | sed -n 's/.*submitted to cluster \([0-9]\+\).*/\1/p')
  
  if [ -z "$cluster_id" ]; then
    echo "Warning: Could not parse ClusterId from condor_submit output."
    continue
  fi
  
  # -------------------------------------------------------------------
  # Wait for this specific cluster to complete
  # -------------------------------------------------------------------
  echo "Waiting for job (ClusterId $cluster_id) to complete..."
  condor_wait logs/seq_log.
  
  # -------------------------------------------------------------------
  # Parse the output file for the time.
  # -------------------------------------------------------------------
  output_file="logs/out_seq."
  
  # Ensure the output file exists
  if [ ! -f "$output_file" ]; then
    echo "Output file '$output_file' not found; cannot parse computation time."
    continue
  fi
  
  # -------------------------------------------------------------------
  # Extract the "Computation: XX.XXXXXX seconds" line
  # The line looks like:   Computation: 12.132704 seconds
  # We can parse out the second field (which is the numeric value).
  # -------------------------------------------------------------------
  time_value=$(grep "Computation:" "$output_file" | awk '{print $2}')
  
  # Check if time_value is a valid decimal number
  if [[ $time_value =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
    echo "Run $i: Computation time = $time_value seconds"
    times[$i]=$time_value
  else
    echo "Could not parse a valid computation time from '$output_file' for run $i."
  fi

done

# -------------------------------------------------------------------
# Now compute the average from the collected times
# -------------------------------------------------------------------
count=0
sum=0
for i in "${!times[@]}"; do
  sum=$(echo "$sum + ${times[$i]}" | bc -l)
  count=$((count + 1))
done

# Print out each run's time and the final average
if [ $count -gt 0 ]; then
  echo "========================================="
  echo "Computation times for each run:"
  for i in "${!times[@]}"; do
    echo "  Run $i: ${times[$i]} seconds"
  done
  avg=$(echo "scale=4; $sum / $count" | bc -l)
  echo
  echo "Average computation time over $count run(s): $avg seconds"
  echo "========================================="
else
  echo "No successful runs or no valid computation times were collected."
fi