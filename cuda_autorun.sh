#!/bin/bash

# Prompt the user for the number of runs
read -p "Enter the number of runs: " num_runs

# Array to store each run's time
declare -a times

# For loop to run the job N times
for i in $(seq 1 $num_runs); do
  echo "================================="
  echo "Submitting CUDA job $i/$num_runs..."
  
  # Submit the Condor job
  submit_output=$(condor_submit job_cuda.sub 2>&1)
  rc=$?
  
  if [ $rc -ne 0 ]; then
    echo "condor_submit failed (exit code $rc)."
    echo "Output from condor_submit:"
    echo "$submit_output"
    break
  fi
  
  #------------------------------------------------------------
  # Extract the ClusterId from the condor_submit output
  #------------------------------------------------------------
  cluster_id=$(echo "$submit_output" | sed -n 's/.*submitted to cluster \([0-9]\+\).*/\1/p')
  
  if [ -z "$cluster_id" ]; then
    echo "Warning: Could not parse ClusterId from condor_submit output."
    break
  fi
  
  #------------------------------------------------------------
  # Wait for this specific cluster to complete
  #------------------------------------------------------------
  echo "Waiting for job (ClusterId $cluster_id) to complete..."
  condor_wait logs/cuda_job.log $cluster_id
  
  #------------------------------------------------------------
  # Parse the output file for the time
  #------------------------------------------------------------
  output_file="logs/cuda_job.out"
  
  if [ ! -f "$output_file" ]; then
    echo "Output file '$output_file' not found; cannot parse computation time."
    break
  fi
  
  #------------------------------------------------------------
  # Extract the line "Computation: XX.XXXX seconds"
  #------------------------------------------------------------
  time_value=$(grep "Computation:" "$output_file" | awk '{print $2}')
  
  # Check if time_value is a valid decimal number
  if [[ $time_value =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
    echo "Run $i: Computation time = $time_value seconds"
    times[$i]=$time_value
  else
    echo "Could not parse a valid computation time from '$output_file' for run $i."
  fi
done

#------------------------------------------------------------
# Compute the average from the collected times
#------------------------------------------------------------
count=0
sum=0
for i in "${!times[@]}"; do
  sum=$(echo "$sum + ${times[$i]}" | bc -l)
  count=$((count + 1))
done

if [ $count -gt 0 ]; then
  echo "========================================="
  echo "Computation times for each of the $count run(s):"
  for i in "${!times[@]}"; do
    echo "  Run $i: ${times[$i]} seconds"
  done
  avg=$(echo "scale=4; $sum / $count" | bc -l)
  echo
  echo "Average computation time: $avg seconds"
  echo "========================================="
else
  echo "No valid computation times were collected."
fi