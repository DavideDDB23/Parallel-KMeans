#!/usr/bin/env python3
import random

def generate_new_file(output_file, num_points, num_dims):
    with open(output_file, 'w') as file:
        for _ in range(num_points):
            # Generate a row of random integers between -100 and 100.
            row = [str(random.randint(-100, 100)) for _ in range(num_dims)]
            # Write the row to the file as tab-separated values.
            file.write('\t'.join(row) + '\n')
    
    print(f"Generated file '{output_file}' with {num_points} points and {num_dims} dimensions.")

def main():
    output_file = '1600k_100.inp'
    num_points = 1600000
    num_dims = 100

    generate_new_file(output_file, num_points, num_dims)

if __name__ == "__main__":
    main()