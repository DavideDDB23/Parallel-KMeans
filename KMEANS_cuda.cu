/*
 * k-Means clustering algorithm
 *
 * CUDA version
 *
 * Parallel computing (Degree in Computer Engineering)
 * 2022/2023
 *
 * Version: 1.0
 *
 * (c) 2022 Diego García-Álvarez, Arturo Gonzalez-Escribano
 * Grupo Trasgo, Universidad de Valladolid (Spain)
 *
 * This work is licensed under a Creative Commons Attribution-ShareAlike 4.0 International License.
 * https://creativecommons.org/licenses/by-sa/4.0/
 */
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <float.h>
#include <cuda.h>


#define MAXLINE 2000
#define MAXCAD 200

//Macros
#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))

/*
 * Macros to show errors when calling a CUDA library function,
 * or after launching a kernel
 */
#define CHECK_CUDA_CALL(a) { \
	cudaError_t ok = a; \
	if ( ok != cudaSuccess ) \
		fprintf(stderr, "-- Error CUDA call in line %d: %s\n", __LINE__, cudaGetErrorString( ok ) ); \
	}

#define CHECK_CUDA_LAST()	{ \
	cudaError_t ok = cudaGetLastError(); \
	if ( ok != cudaSuccess ) \
		fprintf(stderr, "-- Error CUDA last in line %d: %s\n", __LINE__, cudaGetErrorString( ok ) ); \
	}

/* 
Function showFileError: It displays the corresponding error during file reading.
*/
void showFileError(int error, char* filename)
{
	printf("Error\n");
	switch (error)
	{
		case -1:
			fprintf(stderr,"\tFile %s has too many columns.\n", filename);
			fprintf(stderr,"\tThe maximum number of columns has been exceeded. MAXLINE: %d.\n", MAXLINE);
			break;
		case -2:
			fprintf(stderr,"Error reading file: %s.\n", filename);
			break;
		case -3:
			fprintf(stderr,"Error writing file: %s.\n", filename);
			break;
	}
	fflush(stderr);	
}

/* 
Function readInput: It reads the file to determine the number of rows and columns.
*/
int readInput(char* filename, int *lines, int *samples)
{
    FILE *fp;
    char line[MAXLINE] = "";
    char *ptr;
    const char *delim = "\t";
    int contlines, contsamples = 0;
    
    contlines = 0;

    if ((fp=fopen(filename,"r"))!=NULL)
    {
        while(fgets(line, MAXLINE, fp)!= NULL) 
		{
			if (strchr(line, '\n') == NULL)
			{
				return -1;
			}
            contlines++;       
            ptr = strtok(line, delim);
            contsamples = 0;
            while(ptr != NULL)
            {
            	contsamples++;
				ptr = strtok(NULL, delim);
	    	}	    
        }
        fclose(fp);
        *lines = contlines;
        *samples = contsamples;  
        return 0;
    }
    else
	{
    	return -2;
	}
}

/* 
Function readInput2: It loads data from file.
*/
int readInput2(char* filename, float* data)
{
    FILE *fp;
    char line[MAXLINE] = "";
    char *ptr;
    const char *delim = "\t";
    int i = 0;
    
    if ((fp=fopen(filename,"rt"))!=NULL)
    {
        while(fgets(line, MAXLINE, fp)!= NULL)
        {         
            ptr = strtok(line, delim);
            while(ptr != NULL)
            {
            	data[i] = atof(ptr);
            	i++;
				ptr = strtok(NULL, delim);
	   		}
	    }
        fclose(fp);
        return 0;
    }
    else
	{
    	return -2; //No file found
	}
}

/* 
Function writeResult: It writes in the output file the cluster of each sample (point).
*/
int writeResult(int *classMap, int lines, const char* filename)
{	
    FILE *fp;
    
    if ((fp=fopen(filename,"wt"))!=NULL)
    {
        for(int i=0; i<lines; i++)
        {
        	fprintf(fp,"%d\n",classMap[i]);
        }
        fclose(fp);  
   
        return 0;
    }
    else
	{
    	return -3; //No file found
	}
}

/*
Function initCentroids: This function copies the values of the initial centroids, using their 
position in the input data structure as a reference map.
*/
void initCentroids(const float *data, float* centroids, int* centroidPos, int samples, int K)
{
	int i;
	int idx;
	for(i = 0; i < K; i++) {
		idx = centroidPos[i];
		memcpy(&centroids[i * samples], &data[idx * samples], (samples * sizeof(float)));
	}
}

//-------------------------------------------------------------
// CUDA Kernels and Device Functions
//-------------------------------------------------------------

// Implementation of a custom atomicMax operation for floats.
__device__ inline float custom_atomic_max(float *value_address, float val)
{
    int *address_as_int = (int *)value_address;
    int old = *address_as_int, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_int, assumed, __float_as_int(fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

/*
 * step_1_kernel:
 *   - Each thread computes the nearest centroid for one data point.
 *   - Uses dynamic shared memory to copy the centroids and to store block-level
 *     accumulators (sums and counts) for centroid updates.
 *   - Updates the global assignment array (classMap) and counts the number of changes.
 *
 * Dynamic shared memory layout:
 *   [sharedCentroids | blockSums | blockCounts]
 *     - sharedCentroids: K * samples floats (a copy of the centroids)
 *     - blockSums: K * samples floats (partial sums for each centroid)
 *     - blockCounts: K ints (number of points assigned per centroid)
 */
__global__ void step_1_kernel( float *  data,
                               float *  centroids,
                              int *globalCounts,
                              float *globalSums,
                              int *classMap,
                              int *changes_return,
                              int lines,
                              int samples,
                              int K)
{
    // Flatten thread index (we assume grid.y == 1)
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int blockSize = blockDim.x * blockDim.y;
    int idx = blockIdx.x * blockSize + tid;

    // Allocate dynamic shared memory:
    extern __shared__ char sharedBuffer[];
    float *sharedCentroids = (float *)sharedBuffer;                  // K * samples floats.
    float *blockSums = sharedCentroids + K * samples;                  // K * samples floats.
    int   *blockCounts = (int *)(blockSums + K * samples);             // K ints.

    // A block-level shared variable to count how many assignments changed in this block.
    __shared__ int blockChanges;

    // Copy centroids into shared memory.
    for (int i = tid; i < K * samples; i += blockSize)
    {
        sharedCentroids[i] = centroids[i];
    }
    // Initialize blockSums and blockCounts to zero.
    for (int i = tid; i < K * samples; i += blockSize)
    {
        blockSums[i] = 0.0f;
    }
    for (int i = tid; i < K; i += blockSize)
    {
        blockCounts[i] = 0;
    }
    if (tid == 0)
    {
        blockChanges = 0;
    }
    __syncthreads();

    if (idx < lines)
    {
        const float *point = &data[idx * samples];
        int best_cluster = 0;
        float best_dist = FLT_MAX;
        // Loop over centroids and compute squared Euclidean distance.
        for (int c = 0; c < K; c++)
        {
            float dist = 0.0f;
            // Unroll the inner loop if samples is small.
#pragma unroll
            for (int j = 0; j < samples; j++)
            {
                float diff = point[j] - sharedCentroids[c * samples + j];
                dist += diff * diff;
            }
            if (dist < best_dist)
            {
                best_dist = dist;
                best_cluster = c;
            }
        }

        // Note: classMap stores 1-indexed assignments.
        int old_cluster = classMap[idx] - 1;
        if (old_cluster != best_cluster)
        {
            atomicAdd(&blockChanges, 1);
        }
        classMap[idx] = best_cluster + 1;

        // Update block-level accumulators.
        atomicAdd(&blockCounts[best_cluster], 1);
        for (int j = 0; j < samples; j++)
        {
            atomicAdd(&blockSums[best_cluster * samples + j], point[j]);
        }
    }
    __syncthreads();

    // One thread (tid==0) per block updates the global accumulators.
    if (tid == 0)
    {
        atomicAdd(changes_return, blockChanges);
        for (int c = 0; c < K; c++)
        {
            atomicAdd(&globalCounts[c], blockCounts[c]);
            for (int j = 0; j < samples; j++)
            {
                atomicAdd(&globalSums[c * samples + j], blockSums[c * samples + j]);
            }
        }
    }
}

/*
 * step_2_kernel:
 *   - Each thread processes one cluster (centroid).
 *   - The new centroid is computed by averaging the sums in globalSums (from step 1) divided by the count.
 *   - The squared Euclidean distance between the old and new centroid is computed.
 *   - A custom atomic max is used to update the global maximum centroid movement.
 */
__global__ void step_2_kernel(float *globalSums,
                              float *centroids,
                              int *globalCounts,
                              float *maxDistance,
                              int samples,
                              int K)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < K)
    {
        float dist = 0.0f;
        if (globalCounts[c] > 0)
        {
            for (int j = 0; j < samples; j++)
            {
                float newVal = globalSums[c * samples + j] / (float)globalCounts[c];
                float diff = centroids[c * samples + j] - newVal;
                dist += diff * diff;
                centroids[c * samples + j] = newVal;
            }
        }
        custom_atomic_max(maxDistance, dist);
    }
}

int main(int argc, char *argv[])
{
    // START CLOCK (for overall timing)
    clock_t start, end;
    start = clock();

    // PARAMETERS:
    // argv[1]: Input data file
    // argv[2]: Number of clusters
    // argv[3]: Maximum number of iterations
    // argv[4]: Percentage of points that must change to continue
    // argv[5]: Threshold (centroid movement) for convergence
    // argv[6]: Output file (each line: cluster assignment, 1-indexed)
    if (argc != 7)
    {
        fprintf(stderr, "EXECUTION ERROR K-MEANS: Parameters are not correct.\n");
        fprintf(stderr, "./KMEANS [Input Filename] [Number of clusters] [Number of iterations] [Percentage of changes] [Threshold] [Output data file]\n");
        fflush(stderr);
        exit(-1);
    }

    int lines = 0, samples = 0;
    int error = readInput(argv[1], &lines, &samples);
    if (error != 0)
    {
        showFileError(error, argv[1]);
        exit(error);
    }

    float *data = (float *)calloc(lines * samples, sizeof(float));
    if (data == NULL)
    {
        fprintf(stderr, "Memory allocation error.\n");
        exit(-4);
    }
    error = readInput2(argv[1], data);
    if (error != 0)
    {
        showFileError(error, argv[1]);
        exit(error);
    }

    // Parameters from command line.
    int K = atoi(argv[2]);
    int maxIterations = atoi(argv[3]);
    int minChanges = (int)(lines * atof(argv[4]) / 100.0);
    float maxThreshold = atof(argv[5]);

    int *centroidPos = (int *)calloc(K, sizeof(int));
    float *centroids = (float *)calloc(K * samples, sizeof(float));
    int *classMap = (int *)calloc(lines, sizeof(int));
    if (centroidPos == NULL || centroids == NULL || classMap == NULL)
    {
        fprintf(stderr, "Memory allocation error.\n");
        exit(-4);
    }

    // Initialize centroids randomly.
    srand(0);
    for (int i = 0; i < K; i++)
    {
        centroidPos[i] = rand() % lines;
    }
    initCentroids(data, centroids, centroidPos, samples, K);

    printf("\n    Input properties:");
    printf("\n\tData file: %s \n\tPoints: %d\n\tDimensions: %d\n", argv[1], lines, samples);
    printf("\tNumber of clusters: %d\n", K);
    printf("\tMaximum number of iterations: %d\n", maxIterations);
    printf("\tMinimum number of changes: %d [%g%% of %d points]\n", minChanges, atof(argv[4]), lines);
    printf("\tMaximum centroid precision: %f\n", maxThreshold);

    // Retrieve and display CUDA device properties.
    cudaDeviceProp cuda_prop;
    CHECK_CUDA_CALL(cudaGetDeviceProperties(&cuda_prop, 0));
    printf("\n    Device: %s\n", cuda_prop.name);
    printf("\tCompute Capability: %d.%d\n", cuda_prop.major, cuda_prop.minor);
    printf("\tMax threads / block: %d\n", cuda_prop.maxThreadsPerBlock);
    printf("\tMax threads / SM: %d\n", cuda_prop.maxThreadsPerMultiProcessor);
    printf("\tMax shared memory per SM: %zuB\n", cuda_prop.sharedMemPerMultiprocessor);
    printf("\tNumber of SMs: %d\n", cuda_prop.multiProcessorCount);

    CHECK_CUDA_CALL(cudaSetDevice(0));
    CHECK_CUDA_CALL(cudaDeviceSynchronize());

    // Allocate host memory for some intermediate results.
    char *output_msg = (char *)calloc(100000, sizeof(char));
    int it = 0;
    int changes = 0;
    float maxDist = 0.0f;
    // Precompute threshold squared (avoid pow() in loop).
    float thresholdSq = maxThreshold * maxThreshold;

    int *pointsPerClass = (int *)malloc(K * sizeof(int));
    float *auxCentroids = (float *)malloc(K * samples * sizeof(float));
    if (pointsPerClass == NULL || auxCentroids == NULL)
    {
        fprintf(stderr, "Memory allocation error.\n");
        exit(-4);
    }

    // Calculate dynamic shared memory needed for step_1_kernel.
    int sharedMemSize = 2 * K * samples * sizeof(float) + K * sizeof(int);

    // Determine grid dimensions for step_1_kernel.
    // We use a 2D block configuration of 32x32 threads (1024 threads per block).
    dim3 gen_block(32, 32);
    int threadsPerBlock = gen_block.x * gen_block.y;
    int numBlocks = (lines + threadsPerBlock - 1) / threadsPerBlock;
    dim3 dyn_grid_pts(numBlocks, 1);

    // Grid configuration for step_2_kernel: each thread processes one cluster.
    int threadsPerBlock2 = 256;
    int blocksForClusters = (K + threadsPerBlock2 - 1) / threadsPerBlock2;

    // ------------------------------------------------------------
    // GPU Memory Allocation and Data Transfer
    // ------------------------------------------------------------
    float *gpu_data;
    float *gpu_centroids;
    int *gpu_class_map;
    float *gpu_aux_centroids;
    int *gpu_points_per_class;
    int *gpu_changes;
    float *gpu_max_distance;

    int data_size = lines * samples * sizeof(float);
    int centroids_size = K * samples * sizeof(float);

    CHECK_CUDA_CALL(cudaMalloc((void **)&gpu_centroids, centroids_size));
    CHECK_CUDA_CALL(cudaMemcpy(gpu_centroids, centroids, centroids_size, cudaMemcpyHostToDevice));

    CHECK_CUDA_CALL(cudaMalloc((void **)&gpu_data, data_size));
    CHECK_CUDA_CALL(cudaMemcpy(gpu_data, data, data_size, cudaMemcpyHostToDevice));

    // Allocate device memory for the cluster assignments.
    CHECK_CUDA_CALL(cudaMalloc((void **)&gpu_class_map, lines * sizeof(int)));
    CHECK_CUDA_CALL(cudaMemset(gpu_class_map, 0, lines * sizeof(int)));

    // Allocate device memory for auxiliary centroids (to accumulate sums).
    CHECK_CUDA_CALL(cudaMalloc((void **)&gpu_aux_centroids, centroids_size));
    CHECK_CUDA_CALL(cudaMemset(gpu_aux_centroids, 0, centroids_size));

    // Allocate device memory for points per cluster.
    CHECK_CUDA_CALL(cudaMalloc((void **)&gpu_points_per_class, K * sizeof(int)));
    CHECK_CUDA_CALL(cudaMemset(gpu_points_per_class, 0, K * sizeof(int)));

    // Allocate device memory for the change counter.
    CHECK_CUDA_CALL(cudaMalloc((void **)&gpu_changes, sizeof(int)));
    // Allocate device memory for tracking maximum centroid movement.
    CHECK_CUDA_CALL(cudaMalloc((void **)&gpu_max_distance, sizeof(float)));

    end = clock();
    printf("\nMemory allocation and initialization: %f seconds\n", (double)(end - start) / CLOCKS_PER_SEC);
    fflush(stdout);

    // START CUDA computation clock
    start = clock();

    // ------------------------------------------------------------
    // Main k-Means Iterative Loop (on the GPU)
    // ------------------------------------------------------------
    do
    {
        it++;
        // Reset accumulators on the GPU.
        CHECK_CUDA_CALL(cudaMemset(gpu_changes, 0, sizeof(int)));
        CHECK_CUDA_CALL(cudaMemcpy(gpu_max_distance, 0, sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA_CALL(cudaMemset(gpu_aux_centroids, 0, centroids_size));
        CHECK_CUDA_CALL(cudaMemset(gpu_points_per_class, 0, K * sizeof(int)));

        // Launch step_1_kernel: each thread processes one data point.
        step_1_kernel<<<dyn_grid_pts, gen_block, sharedMemSize>>>(gpu_data,
                                                                    gpu_centroids,
                                                                    gpu_points_per_class,
                                                                    gpu_aux_centroids,
                                                                    gpu_class_map,
                                                                    gpu_changes,
                                                                    lines,
                                                                    samples,
                                                                    K);
        CHECK_CUDA_LAST();
        CHECK_CUDA_CALL(cudaDeviceSynchronize());

        // Retrieve the number of changed assignments.
        CHECK_CUDA_CALL(cudaMemcpy(&changes, gpu_changes, sizeof(int), cudaMemcpyDeviceToHost));

        // Launch step_2_kernel: update each centroid and compute its movement.
        step_2_kernel<<<blocksForClusters, threadsPerBlock2>>>(gpu_aux_centroids,
                                                               gpu_centroids,
                                                               gpu_points_per_class,
                                                               gpu_max_distance,
                                                               samples,
                                                               K);
        CHECK_CUDA_LAST();
        CHECK_CUDA_CALL(cudaDeviceSynchronize());

        // Copy the maximum centroid movement back to the host.
        CHECK_CUDA_CALL(cudaMemcpy(&maxDist, gpu_max_distance, sizeof(float), cudaMemcpyDeviceToHost));

    } while ((changes > minChanges) && (it < maxIterations) && (maxDist > thresholdSq));

    printf("%s", output_msg);
    CHECK_CUDA_CALL(cudaDeviceSynchronize());

    end = clock();
    printf("\nComputation: %f seconds", (double)(end - start) / CLOCKS_PER_SEC);
    fflush(stdout);
    start = clock();

    if (changes <= minChanges)
    {
        printf("\n\nTermination condition:\nMinimum number of changes reached: %d [%d]", changes, minChanges);
    }
    else if (it >= maxIterations)
    {
        printf("\n\nTermination condition:\nMaximum number of iterations reached: %d [%d]", it, maxIterations);
    }
    else
    {
        printf("\n\nTermination condition:\nCentroid update precision reached: %g [%g]", maxDist, maxThreshold);
    }

    // Copy final cluster assignments from the GPU.
    CHECK_CUDA_CALL(cudaMemcpy(classMap, gpu_class_map, lines * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_CALL(cudaDeviceSynchronize());

    // Write the results to the output file.
    error = writeResult(classMap, lines, argv[6]);
    if (error != 0)
    {
        showFileError(error, argv[6]);
        exit(error);
    }

    // Free host and device memory.
    free(data);
    free(classMap);
    free(centroidPos);
    free(centroids);
    free(pointsPerClass);
    free(auxCentroids);

    cudaFree(gpu_data);
    cudaFree(gpu_centroids);
    cudaFree(gpu_aux_centroids);
    cudaFree(gpu_changes);
    cudaFree(gpu_class_map);
    cudaFree(gpu_max_distance);
    cudaFree(gpu_points_per_class);

    end = clock();
    printf("\n\nMemory deallocation: %f seconds\n", (double)(end - start) / CLOCKS_PER_SEC);
    fflush(stdout);
    return 0;
}