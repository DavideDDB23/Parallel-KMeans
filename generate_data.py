#!/usr/bin/env python3
import numpy as np
import sys

def generate_new_file(output_file, num_points, num_dims):
    new_data = np.random.randint(-100, 101, size=(num_points, num_dims))
    
    # Save the generated data to the output file with tab-separated values.
    np.savetxt(output_file, new_data, fmt='%d', delimiter='\t')
    print(f"Generated file '{output_file}' with {num_points} points and {num_dims} dimensions.")

def main():
    output_file = '3200k_100.inp'
    num_points = 3200000
    num_dims = 100

    generate_new_file(output_file, num_points, num_dims)

if __name__ == "__main__":
    main()