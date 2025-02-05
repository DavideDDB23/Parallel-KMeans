/*
 * Fused k-Means clustering algorithm (assignment + update in one kernel)
 *
 * Efficient CUDA version with 1-indexed cluster assignments
 * using a single fused kernel that iterates on-device.
 *
 *
 * (c) 2022 Diego García-Álvarez, Arturo Gonzalez-Escribano, Modified by ChatGPT
 *
 * This work is licensed under a Creative Commons Attribution-ShareAlike 4.0 International License.
 */

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <float.h>
#include <cuda.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

#define MAXLINE 2000
#define MAXCAD 200

// Macros for CUDA error checking
#define CHECK_CUDA_CALL(a) { \
    cudaError_t ok = a; \
    if (ok != cudaSuccess) { \
        fprintf(stderr, "CUDA Error in line %d: %s\n", __LINE__, cudaGetErrorString(ok)); \
        exit(EXIT_FAILURE); \
    } \
}
#define CHECK_CUDA_LAST() { \
    cudaError_t ok = cudaGetLastError(); \
    if (ok != cudaSuccess) { \
        fprintf(stderr, "CUDA Error (last) in line %d: %s\n", __LINE__, cudaGetErrorString(ok)); \
        exit(EXIT_FAILURE); \
    } \
}

//-------------------------------------------------------------
// File I/O helper functions
//-------------------------------------------------------------
void showFileError(int error, char* filename) {
    printf("Error\n");
    switch (error) {
        case -1:
            fprintf(stderr, "\tFile %s has too many columns.\n", filename);
            break;
        case -2:
            fprintf(stderr, "Error reading file: %s.\n", filename);
            break;
        case -3:
            fprintf(stderr, "Error writing file: %s.\n", filename);
            break;
    }
    fflush(stderr);
}

int readInput(char* filename, int *lines, int *samples) {
    FILE *fp;
    char line[MAXLINE] = "";
    char *ptr;
    const char *delim = "\t";
    int contlines = 0, contsamples = 0;
    
    if ((fp = fopen(filename, "r")) != NULL) {
        while (fgets(line, MAXLINE, fp) != NULL) {
            if (strchr(line, '\n') == NULL) {
                fclose(fp);
                return -1;
            }
            contlines++;
            ptr = strtok(line, delim);
            contsamples = 0;
            while (ptr != NULL) {
                contsamples++;
                ptr = strtok(NULL, delim);
            }
        }
        fclose(fp);
        *lines = contlines;
        *samples = contsamples;
        return 0;
    } else {
        return -2;
    }
}

int readInput2(char* filename, float* data) {
    FILE *fp;
    char line[MAXLINE] = "";
    char *ptr;
    const char *delim = "\t";
    int i = 0;
    
    if ((fp = fopen(filename, "rt")) != NULL) {
        while (fgets(line, MAXLINE, fp) != NULL) {
            ptr = strtok(line, delim);
            while (ptr != NULL) {
                data[i] = atof(ptr);
                i++;
                ptr = strtok(NULL, delim);
            }
        }
        fclose(fp);
        return 0;
    } else {
        return -2;
    }
}

int writeResult(int *classMap, int lines, const char* filename) {    
    FILE *fp;
    if ((fp = fopen(filename, "wt")) != NULL) {
        for (int i = 0; i < lines; i++) {
            fprintf(fp, "%d\n", classMap[i]); // assignments are 1-indexed
        }
        fclose(fp);
        return 0;
    } else {
        return -3;
    }
}

void initCentroids(const float *data, float* centroids, int* centroidPos, int d, int K) {
    for (int i = 0; i < K; i++) {
        int idx = centroidPos[i];
        memcpy(&centroids[i * d], &data[idx * d], d * sizeof(float));
    }
}

//-------------------------------------------------------------
// Fused k-Means kernel (assignment + update in one kernel)
// Uses cooperative groups for grid-wide synchronization.
// The iterative loop runs on-device.
// Global arrays for accumulators must be allocated beforehand.
// Cluster assignments are stored 1-indexed.
__global__ void fused_kmeans_kernel(
    const float* data,        // [n * d]
    int n,
    int d,
    int K,
    int max_iterations,
    int min_changes,          // termination: minimum number of point changes to continue
    float threshold,          // termination: squared threshold for max centroid movement
    int* class_map,           // [n]  (1-indexed assignments)
    float* centroids,         // [K * d]
    float* globalSums,        // [K * d]
    int* globalCounts,        // [K]
    int* globalChanges,       // single int (number of changed assignments in iteration)
    float* globalMaxMovement  // single float (max centroid movement squared)
)
{
    cg::grid_group grid = cg::this_grid();
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = grid.size();
    
    // Perform iterations on-device.
    for (int iter = 0; iter < max_iterations; iter++) {
        // Reset accumulators in global memory.
        for (int i = tid; i < K * d; i += total_threads) {
            globalSums[i] = 0.0f;
        }
        for (int i = tid; i < K; i += total_threads) {
            globalCounts[i] = 0;
        }
        if (tid == 0) {
            *globalChanges = 0;
            *globalMaxMovement = 0.0f;
        }
        grid.sync();

        // Each thread processes a subset of points.
        for (int i = tid; i < n; i += total_threads) {
            const float* point = &data[i * d];
            float best_dist = FLT_MAX;
            int best_cluster = -1;
            // Compute squared Euclidean distance to each centroid.
            for (int c = 0; c < K; c++) {
                float dist = 0.0f;
                for (int j = 0; j < d; j++) {
                    float diff = point[j] - centroids[c * d + j];
                    dist += diff * diff;
                }
                if (dist < best_dist) {
                    best_dist = dist;
                    best_cluster = c;
                }
            }
            // Compare to old assignment (stored 1-indexed).
            int old_assignment = class_map[i] - 1;  // convert to 0-index
            if (old_assignment != best_cluster) {
                atomicAdd(globalChanges, 1);
            }
            // Store new assignment as 1-indexed.
            class_map[i] = best_cluster + 1;
            // Accumulate sum for the chosen cluster.
            atomicAdd(&globalCounts[best_cluster], 1);
            for (int j = 0; j < d; j++) {
                atomicAdd(&globalSums[best_cluster * d + j], point[j]);
            }
        }
        grid.sync();

        // One thread (tid==0) updates centroids.
        if (tid == 0) {
            for (int c = 0; c < K; c++) {
                if (globalCounts[c] > 0) {
                    float movement = 0.0f;
                    for (int j = 0; j < d; j++) {
                        float newVal = globalSums[c * d + j] / globalCounts[c];
                        float diff = centroids[c * d + j] - newVal;
                        movement += diff * diff;
                        centroids[c * d + j] = newVal;
                    }
                    if (movement > *globalMaxMovement)
                        *globalMaxMovement = movement;
                }
            }
        }
        grid.sync();

        // Check termination condition.
        int changes = *globalChanges;
        float max_movement = *globalMaxMovement;
        // If few changes or centroid movement is small, finish.
        if (changes <= min_changes || max_movement <= threshold) {
            break;
        }
        grid.sync();
    } // end for iterations
}

//-------------------------------------------------------------
// Host code
//-------------------------------------------------------------
int main(int argc, char* argv[]) {

    // Expected parameters:
    // argv[1]: Input data file  
    // argv[2]: Number of clusters  
    // argv[3]: Maximum number of iterations  
    // argv[4]: Percentage of points (changes) required to continue  
    // argv[5]: Threshold (centroid movement) for convergence  
    // argv[6]: Output file (each line: 1-indexed cluster assignment)
    if (argc != 7) {
        fprintf(stderr, "EXECUTION ERROR K-MEANS: Parameters are not correct.\n");
        fprintf(stderr, "./KMEANS [Input Filename] [Number of clusters] [Number of iterations] [Percentage of changes] [Threshold] [Output data file]\n");
        exit(EXIT_FAILURE);
    }
    
    int n = 0, d = 0;
    int error = readInput(argv[1], &n, &d);
    if (error != 0) {
        showFileError(error, argv[1]);
        exit(error);
    }
    
    float *data = (float*) calloc(n * d, sizeof(float));
    if (data == NULL) {
        fprintf(stderr, "Memory allocation error.\n");
        exit(EXIT_FAILURE);
    }
    error = readInput2(argv[1], data);
    if (error != 0) {
        showFileError(error, argv[1]);
        exit(error);
    }
    
    // Parameters.
    int K = atoi(argv[2]);
    int maxIterations = atoi(argv[3]);
    int min_changes = (int)(n * atof(argv[4]) / 100.0);
    float threshold = atof(argv[5]);
    float threshold_sq = threshold * threshold; // squared threshold

    int* centroid_pos = (int*) calloc(K, sizeof(int));
    float* centroids = (float*) calloc(K * d, sizeof(float));
    int* class_map = (int*) calloc(n, sizeof(int));
    if (centroid_pos == NULL || centroids == NULL || class_map == NULL) {
        fprintf(stderr, "Memory allocation error.\n");
        exit(EXIT_FAILURE);
    }
    
    // Initialize centroids by randomly choosing points.
    srand(0);
    for (int i = 0; i < K; i++) {
        centroid_pos[i] = rand() % n;
    }
    initCentroids(data, centroids, centroid_pos, d, K);
    
    // Initialize class_map to 0 (meaning unassigned).
    for (int i = 0; i < n; i++) {
        class_map[i] = 0;
    }
    
    printf("\nInput properties:\n");
    printf("\tData file: %s\n\tPoints: %d\n\tDimensions: %d\n", argv[1], n, d);
    printf("\tClusters: %d\n\tMax iterations: %d\n", K, max_iterations);
    printf("\tMin changes: %d [%g%% of %d points]\n", min_changes, atof(argv[4]), n);
    printf("\tThreshold (squared): %f\n", threshold_sq);
    
    // --- Allocate GPU memory ---
    int data_size = n * d * sizeof(float);
    int centroids_size = K * d * sizeof(float);
    int classes_size = n * sizeof(int);
    int sums_size = K * d * sizeof(float);
    int counts_size = K * sizeof(int);
    
    float *gpu_data, *gpu_centroids, *gpu_globalSums;
    int *gpu_class_map, *gpu_globalCounts, *gpu_globalChanges;
    float *gpu_globalMaxMovement;
    
    CHECK_CUDA_CALL(cudaMalloc((void**)&gpu_data, data_size));
    CHECK_CUDA_CALL(cudaMemcpy(gpu_data, data, data_size, cudaMemcpyHostToDevice));
    
    CHECK_CUDA_CALL(cudaMalloc((void**)&gpu_centroids, centroids_size));
    CHECK_CUDA_CALL(cudaMemcpy(gpu_centroids, centroids, centroids_size, cudaMemcpyHostToDevice));
    
    CHECK_CUDA_CALL(cudaMalloc((void**)&gpu_class_map, classes_size));
    CHECK_CUDA_CALL(cudaMemset(gpu_class_map, 0, classes_size));
    
    CHECK_CUDA_CALL(cudaMalloc((void**)&gpu_globalSums, sums_size));
    CHECK_CUDA_CALL(cudaMemset(gpu_globalSums, 0, sums_size));
    
    CHECK_CUDA_CALL(cudaMalloc((void**)&gpu_globalCounts, counts_size));
    CHECK_CUDA_CALL(cudaMemset(gpu_globalCounts, 0, counts_size));
    
    CHECK_CUDA_CALL(cudaMalloc((void**)&gpu_globalChanges, sizeof(int)));
    CHECK_CUDA_CALL(cudaMemset(gpu_globalChanges, 0, sizeof(int)));
    
    CHECK_CUDA_CALL(cudaMalloc((void**)&gpu_globalMaxMovement, sizeof(float)));
    CHECK_CUDA_CALL(cudaMemset(gpu_globalMaxMovement, 0, sizeof(float)));
    
    // --- Launch the fused kernel cooperatively ---
    // Choose block and grid dimensions. (For cooperative launch, the entire grid must be launched together.)
    int threadsPerBlock = 256;
    int numBlocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    // It is possible to have more threads than points; extra threads will simply loop over the data.
    dim3 blockDim(threadsPerBlock);
    dim3 gridDim(numBlocks);
    
    // Calculate the total number of threads in the grid (used in the kernel).
    // Note: For cooperative launch, the device must support grid synchronization.
    void* kernelArgs[] = {
        (void*)&gpu_data,
        (void*)&n,
        (void*)&d,
        (void*)&K,
        (void*)&max_iterations,
        (void*)&min_changes,
        (void*)&threshold_sq,
        (void*)&gpu_class_map,
        (void*)&gpu_centroids,
        (void*)&gpu_globalSums,
        (void*)&gpu_globalCounts,
        (void*)&gpu_globalChanges,
        (void*)&gpu_globalMaxMovement
    };
    
    // Launch the kernel cooperatively.
    CHECK_CUDA_CALL(cudaLaunchCooperativeKernel((void*)fused_kmeans_kernel, gridDim, blockDim, kernelArgs));
    CHECK_CUDA_CALL(cudaDeviceSynchronize());
    
    // Copy back results.
    CHECK_CUDA_CALL(cudaMemcpy(class_map, gpu_class_map, classes_size, cudaMemcpyDeviceToHost));
    CHECK_CUDA_CALL(cudaMemcpy(centroids, gpu_centroids, centroids_size, cudaMemcpyDeviceToHost));
    
    // Write the final assignments.
    error = writeResult(class_map, n, argv[6]);
    if (error != 0) {
        showFileError(error, argv[6]);
        exit(error);
    }
    
    // Free host memory.
    free(data);
    free(class_map);
    free(centroid_pos);
    free(centroids);
    
    // Free GPU memory.
    cudaFree(gpu_data);
    cudaFree(gpu_centroids);
    cudaFree(gpu_class_map);
    cudaFree(gpu_globalSums);
    cudaFree(gpu_globalCounts);
    cudaFree(gpu_globalChanges);
    cudaFree(gpu_globalMaxMovement);
    
    printf("\nFused kernel k-Means completed.\n");
    return 0;
}