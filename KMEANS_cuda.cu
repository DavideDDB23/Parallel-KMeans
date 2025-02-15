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

#define CHECK_CUDA_LAST() { \
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

    if ((fp = fopen(filename,"r")) != NULL)
    {
        while (fgets(line, MAXLINE, fp) != NULL) 
		{
			if (strchr(line, '\n') == NULL)
			{
				return -1;
			}
            contlines++;       
            ptr = strtok(line, delim);
            contsamples = 0;
            while (ptr != NULL)
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
    
    if ((fp = fopen(filename,"rt")) != NULL)
    {
        while (fgets(line, MAXLINE, fp) != NULL)
        {         
            ptr = strtok(line, delim);
            while (ptr != NULL)
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
    	return -2; // File not found
	}
}

/* 
Function writeResult: It writes in the output file the cluster of each sample (point).
*/
int writeResult(int *classMap, int lines, const char* filename)
{	
    FILE *fp;
    
    if ((fp = fopen(filename,"wt")) != NULL)
    {
        for (int i = 0; i < lines; i++)
        {
        	fprintf(fp, "%d\n", classMap[i]);
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

// Constant memory declarations for read-only parameters.
// These variables are copied from host once and used by all kernels without further data transfers.
__constant__ int gpu_K; // Number of clusters
__constant__ int gpu_n; // Number of data points (lines)
__constant__ int gpu_d; // Number of dimensions (samples)

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
 *     accumulators for centroid updates.
 *   - Updates the global assignment array (classMap) and counts the number of changes.
 *
 * Dynamic shared memory:
 *     - sharedCentroids: K * samples * sizeof(float)   (copy of the centroids)
 *     - blockSums: K * samples * sizeof(float)         (partial sums for each centroid)
 *     - blockCounts: K * sizeof(int)                   (number of points assigned per centroid)
 */
__global__ void step_1_kernel(float *data,
                              float *centroids,
                              int *globalCounts,
                              float *globalSums,
                              int *classMap,
                              int *changes_return)
{
    // Load constant parameters from constant memory.
    int lines   = gpu_n;
    int samples = gpu_d;
    int K       = gpu_K;

    // Flatten thread index
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int blockSize = blockDim.x * blockDim.y;
    int idx = blockIdx.x * blockSize + tid;

    // Allocate dynamic shared memory:
    extern __shared__ char sharedBuffer[];
    float *sharedCentroids = (float *)sharedBuffer;              // K * samples
    float *blockSums = sharedCentroids + K * samples;            // K * samples
    int   *blockCounts = (int *)(blockSums + K * samples);       // K

    // A block-level shared variable to count how many assignments changed in this block
    __shared__ int blockChanges;

    // Copy centroids into shared memory
    for (int i = tid; i < K * samples; i += blockSize)
    {
        sharedCentroids[i] = centroids[i];
    }

    // Initialize blockSums and blockCounts to zero for each centroid in this block.
    for (int i = tid; i < K * samples; i += blockSize)
    {
        blockSums[i] = 0.0f;
    }
    for (int i = tid; i < K; i += blockSize)
    {
        blockCounts[i] = 0;
    }

    // Initialize changes for this block
    if (tid == 0)
    {
        blockChanges = 0;
    }
    __syncthreads();

    // Each thread processes one data point.
    if (idx < lines)
    {
        const float *point = &data[idx * samples];
        int class_idx = 1;
        float min_dist = FLT_MAX;

        // For each centroid...
        for (int c = 0; c < K; c++)
        {
            float dist = 0.0f;
            
            #pragma unroll
            for (int j = 0; j < samples; j++)
            {
                // Compute l_2 (squared, without sqrt)
                float diff = point[j] - sharedCentroids[c * samples + j];
                dist += diff * diff;
            }

			// If the distance is smallest so far, update min_dist and the class of the point
            if (dist < min_dist)
            {
                min_dist = dist;
                class_idx = c + 1;
            }
        }

        // If the class changed, increment atomically blockChanges
        if (classMap[idx] != class_idx)
        {
            atomicAdd(&blockChanges, 1);
        }

        // Assign the new class to the point
        classMap[idx] = class_idx;

        // Update block-level accumulators using 0-indexed cluster index.
        int accum_idx = class_idx - 1;
        atomicAdd(&blockCounts[accum_idx], 1);
        for (int j = 0; j < samples; j++)
        {
            atomicAdd(&blockSums[accum_idx * samples + j], point[j]);
        }
    }
    __syncthreads();

    // One thread (tid==0) per block updates the global accumulators.
    if (tid == 0)
    {
        // Add blockChanges to the global changes counter
        atomicAdd(changes_return, blockChanges);

        // For each centroid... add block sums and counts to the global sums and counts
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
 *   - Each thread processes one centroid.
 *   - The new centroid is computed by averaging the sums in globalSums (from step 1) divided by the count.
 *   - Compute squared Euclidean distance between the old and new centroid.
 *   - To update the global maximum centroid movement is used a custom atomic max.
 */
__global__ void step_2_kernel(float *globalSums,
                              float *centroids,
                              int *globalCounts,
                              float *maxDistance)
{
    // Load constant parameters from constant memory.
    int samples = gpu_d;
    int K       = gpu_K;

    int c = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Each thread updates one centroid if c < K
    if (c < K)
    {
        float dist = 0.0f;

        // Only update the centroid if there is at least one point assigned to it.
        if (globalCounts[c] > 0)
        {
            // For each centroid's dimension...
            for (int j = 0; j < samples; j++)
            {
                // Compute new value of the centroid in this dimension by averaging
                float newVal = globalSums[c * samples + j] / (float)globalCounts[c];

                // Calculate the squared difference between the old and new coordinate values
                float diff = centroids[c * samples + j] - newVal;
                dist += diff * diff;

                // Update the centroid's coordinate with the newly computed value.
                centroids[c * samples + j] = newVal;
            }
        }

        // Update maximum distance using a custom atomic max for floats
        custom_atomic_max(maxDistance, dist);
    }
}

int main(int argc, char *argv[])
{
    //START CLOCK***************************************
	clock_t start, end;
	start = clock();
	//**************************************************
	
    /*
	* PARAMETERS
	*
	* argv[1]: Input data file
	* argv[2]: Number of clusters
	* argv[3]: Maximum number of iterations of the method. Algorithm termination condition.
	* argv[4]: Minimum percentage of class changes. Algorithm termination condition.
	*          If between one iteration and the next, the percentage of class changes is less than
	*          this percentage, the algorithm stops.
	* argv[5]: Precision in the centroid distance after the update.
	*          It is an algorithm termination condition. If between one iteration of the algorithm 
	*          and the next, the maximum distance between centroids is less than this precision, the
	*          algorithm stops.
	* argv[6]: Output file. Class assigned to each point of the input file.
	* */
	if(argc !=  7)
	{
		fprintf(stderr,"EXECUTION ERROR K-MEANS: Parameters are not correct.\n");
		fprintf(stderr,"./KMEANS [Input Filename] [Number of clusters] [Number of iterations] [Number of changes] [Threshold] [Output data file]\n");
		fflush(stderr);
		exit(-1);
	}

    // Reading the input data
	// lines = number of points; samples = number of dimensions per point
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

	// Parameters
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

	// Initial centrodis
    srand(0);
    for (int i = 0; i < K; i++)
    {
        centroidPos[i] = rand() % lines;
    }

    // Loading the array of initial centroids with the data from the array data
	// The centroids are points stored in the data array.
    initCentroids(data, centroids, centroidPos, samples, K);

    printf("\n    Input properties:");
    printf("\n\tData file: %s \n\tPoints: %d\n\tDimensions: %d\n", argv[1], lines, samples);
    printf("\tNumber of clusters: %d\n", K);
    printf("\tMaximum number of iterations: %d\n", maxIterations);
    printf("\tMinimum number of changes: %d [%g%% of %d points]\n", minChanges, atof(argv[4]), lines);
    printf("\tMaximum centroid precision: %f\n", maxThreshold);

	// Check CUDA device properties
    cudaDeviceProp cuda_prop;
    CHECK_CUDA_CALL(cudaGetDeviceProperties(&cuda_prop, 0));
    printf("\n    Device: %s\n", cuda_prop.name);
    printf("\tCompute Capability: %d.%d\n", cuda_prop.major, cuda_prop.minor);
    printf("\tMax threads / block: %d\n", cuda_prop.maxThreadsPerBlock);
    printf("\tMax threads / SM: %d\n", cuda_prop.maxThreadsPerMultiProcessor);
    printf("\tMax shared memory per SM: %zuB\n", cuda_prop.sharedMemPerMultiprocessor);
    printf("\tNumber of SMs: %d\n", cuda_prop.multiProcessorCount);

    // Set GPU device
    CHECK_CUDA_CALL(cudaSetDevice(0));
    CHECK_CUDA_CALL(cudaDeviceSynchronize());

	// Initialize constant vars
    CHECK_CUDA_CALL(cudaMemcpyToSymbol(gpu_K, &K, sizeof(int)));
    CHECK_CUDA_CALL(cudaMemcpyToSymbol(gpu_n, &lines, sizeof(int)));
    CHECK_CUDA_CALL(cudaMemcpyToSymbol(gpu_d, &samples, sizeof(int)));

    char *output_msg = (char *)calloc(100000, sizeof(char));
    
    int it = 0;
    int changes = 0;
    float maxDist = 0.0f;

    // pointPerClass: number of points classified in each class
	// auxCentroids: mean of the points in each class
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
    // 2D block configuration of 32x32 threads (1024 threads per block).
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

    // Allocate and copy the centroids array.
    CHECK_CUDA_CALL(cudaMalloc((void **)&gpu_centroids, centroids_size));
    CHECK_CUDA_CALL(cudaMemcpy(gpu_centroids, centroids, centroids_size, cudaMemcpyHostToDevice));

    // Allocate and copy the data array.
    CHECK_CUDA_CALL(cudaMalloc((void **)&gpu_data, data_size));
    CHECK_CUDA_CALL(cudaMemcpy(gpu_data, data, data_size, cudaMemcpyHostToDevice));

    // Allocate device memory for classMap.
    CHECK_CUDA_CALL(cudaMalloc((void **)&gpu_class_map, lines * sizeof(int)));
    CHECK_CUDA_CALL(cudaMemset(gpu_class_map, 0, lines * sizeof(int)));

    // Allocate device memory for auxCentroids.
    CHECK_CUDA_CALL(cudaMalloc((void **)&gpu_aux_centroids, centroids_size));
    CHECK_CUDA_CALL(cudaMemset(gpu_aux_centroids, 0, centroids_size));

    // Allocate device memory for pointsPerClass.
    CHECK_CUDA_CALL(cudaMalloc((void **)&gpu_points_per_class, K * sizeof(int)));
    CHECK_CUDA_CALL(cudaMemset(gpu_points_per_class, 0, K * sizeof(int)));

    // Allocate device memory for the changes.
    CHECK_CUDA_CALL(cudaMalloc((void **)&gpu_changes, sizeof(int)));
    // Allocate device memory for maxDist.
    CHECK_CUDA_CALL(cudaMalloc((void **)&gpu_max_distance, sizeof(float)));

    //END CLOCK*****************************************
	end = clock();
	printf("\nCUDA initialization and Memory Allocation: %f seconds\n", (double)(end - start) / CLOCKS_PER_SEC);
	fflush(stdout);
	//**************************************************
	//START CLOCK***************************************
	start = clock();
	//**************************************************

    float initial_max_distance = 0.0f;
    do
    {
        it++; // increment iteration counter

		// Reset changes, max distance, table of centroids and points per class
        CHECK_CUDA_CALL(cudaMemset(gpu_changes, 0, sizeof(int)));
        CHECK_CUDA_CALL(cudaMemcpy(gpu_max_distance, &initial_max_distance, sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA_CALL(cudaMemset(gpu_aux_centroids, 0, centroids_size));
        CHECK_CUDA_CALL(cudaMemset(gpu_points_per_class, 0, K * sizeof(int)));

        // Launch step_1_kernel: each thread processes one data point.
        step_1_kernel<<<dyn_grid_pts, gen_block, sharedMemSize>>>(gpu_data,
                                                                    gpu_centroids,
                                                                    gpu_points_per_class,
                                                                    gpu_aux_centroids,
                                                                    gpu_class_map,
                                                                    gpu_changes);
        CHECK_CUDA_LAST();

        // Write down to host the changes for checking convergence condition after waiting for GPU
        CHECK_CUDA_CALL(cudaDeviceSynchronize());
        CHECK_CUDA_CALL(cudaMemcpy(&changes, gpu_changes, sizeof(int), cudaMemcpyDeviceToHost));

        // Launch step_2_kernel: update each centroid and compute its movement.
        step_2_kernel<<<blocksForClusters, threadsPerBlock2>>>(gpu_aux_centroids,
                                                               gpu_centroids,
                                                               gpu_points_per_class,
                                                               gpu_max_distance);
        CHECK_CUDA_LAST();
        
        // Write down to host the max movement for checking convergence condition
        CHECK_CUDA_CALL(cudaDeviceSynchronize());
        CHECK_CUDA_CALL(cudaMemcpy(&maxDist, gpu_max_distance, sizeof(float), cudaMemcpyDeviceToHost));

    } while ((changes > minChanges) && (it < maxIterations) && (maxDist > pow(maxThreshold, 2)));

    // Output and termination conditions
    printf("%s", output_msg);
    
    CHECK_CUDA_CALL(cudaDeviceSynchronize());

    //END CLOCK*****************************************
	end = clock();
	printf("\nComputation Time: %f seconds", (double)(end - start) / CLOCKS_PER_SEC);
	fflush(stdout);
	//**************************************************
	//START CLOCK***************************************
	start = clock();
	//*************************************************

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

	// Writing the classification of each point to the output file, from the GPU.
    CHECK_CUDA_CALL(cudaMemcpy(classMap, gpu_class_map, lines * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_CALL(cudaDeviceSynchronize());

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

    //END CLOCK*****************************************
	end = clock();
	printf("\n\nMemory deallocation: %f seconds\n", (double)(end - start) / CLOCKS_PER_SEC);
	fflush(stdout);
	//***************************************************/
	return 0;
}