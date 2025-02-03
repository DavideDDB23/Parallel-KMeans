/*
 * k-Means clustering algorithm
 *
 * MPI+PThreads version
 *
 * Parallel computing (Degree in Computer Engineering)
 * 2022/2023
 *
 * Version: 1.1
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
#include <mpi.h>
#include <pthread.h>

#define MAXLINE 2000
#define MAXCAD 200

// Macros
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

/*
Function showFileError: It displays the corresponding error during file reading.
*/
void showFileError(int error, char *filename)
{
    printf("Error\n");
    switch (error)
    {
    case -1:
        fprintf(stderr, "\tFile %s has too many columns.\n", filename);
        fprintf(stderr, "\tThe maximum number of columns has been exceeded. MAXLINE: %d.\n", MAXLINE);
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

/*
Function readInput: It reads the file to determine the number of rows and columns.
*/
int readInput(char *filename, int *lines, int *samples)
{
    FILE *fp;
    char line[MAXLINE] = "";
    char *ptr;
    const char *delim = "\t";
    int contlines, contsamples = 0;
    contlines = 0;
    if ((fp = fopen(filename, "r")) != NULL)
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
int readInput2(char *filename, float *data)
{
    FILE *fp;
    char line[MAXLINE] = "";
    char *ptr;
    const char *delim = "\t";
    int i = 0;
    if ((fp = fopen(filename, "rt")) != NULL)
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
int writeResult(int *classMap, int lines, const char *filename)
{
    FILE *fp;
    if ((fp = fopen(filename, "wt")) != NULL)
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
        return -3; // File open error
    }
}

/*
Function initCentroids: This function copies the values of the initial centroids, using their
position in the input data structure as a reference map.
*/
void initCentroids(const float *data, float *centroids, int *centroidPos, int samples, int K)
{
    int i, idx;
    for (i = 0; i < K; i++)
    {
        idx = centroidPos[i];
        memcpy(&centroids[i * samples], &data[idx * samples], (samples * sizeof(float)));
    }
}

/*
Function euclideanDistance: Euclidean distance
This function could be modified
*/
float euclideanDistance(float *point, float *center, int samples)
{
    float dist = 0.0;
    for (int i = 0; i < samples; i++)
    {
        dist += (point[i] - center[i]) * (point[i] - center[i]);
    }
    return dist; // Squared distance
}

/*
Function zeroFloatMatriz: Set matrix elements to 0
This function could be modified
*/
void zeroFloatMatriz(float *matrix, int rows, int columns)
{
    memset(matrix, 0, rows * columns * sizeof(float));
}

/*
Function zeroIntArray: Set array elements to 0
This function could be modified
*/
void zeroIntArray(int *array, int size)
{
    memset(array, 0, size * sizeof(int));
}

//  #############################################
//      PThreads data structures and functions
//  #############################################

/*
 * Data structure to pass information to each Pthread used for point assignment.
 *
 * thread_id:       Identifier for the thread.
 * num_threads:     Total number of threads.
 * local_n:         Number of local data points in this MPI process.
 * samples:         Number of dimensions per point.
 * K:               Number of clusters.
 * local_points:    Pointer to the local data points.
 * local_classMap:  Array holding the current cluster assignment for each point.
 * centroids:       Global centroids (shared among threads).
 * local_changes:   Counter for the number of changes in point assignments by this thread.
 * start:           Starting index in the local_points array for this thread.
 * end:             Ending index (exclusive) for this thread.
 */
typedef struct
{
    int thread_id;
    int num_threads;
    int local_n;
    int samples;
    int K;
    float *local_points;
    int *local_classMap;
    float *centroids;
    int local_changes;
    int start;         
    int end;           
} assign_thread_data_t;

/*
 * Data structure to pass information to each Pthread used for accumulating centroid contributions.
 *
 * thread_id:         Identifier for the thread.
 * num_threads:       Total number of threads.
 * local_n:           Number of local data points in this MPI process.
 * samples:           Number of dimensions per points.
 * K:                 Number of clusters.
 * local_points:      Pointer to the local data points.
 * local_classMap:    Array with current cluster assignments.
 * partial_counts:    Thread’s partial count of points per cluster (array of length K).
 * partial_centroids: Thread’s partial sum of point coordinates per cluster (array of length K*samples).
 * start:             Starting index in the local_points array for this thread.
 * end:               Ending index (exclusive) for this thread.
 */
typedef struct
{
    int thread_id;
    int num_threads;
    int local_n;
    int samples;
    int K;
    float *local_points;
    int *local_classMap;
    int *partial_counts;      
    float *partial_centroids; 
    int start;               
    int end;                 
} centroid_thread_data_t;

/*
 * Pthread function that assigns each data point (in its assigned block) to the nearest centroid.
 * Each thread takes a part of the data, denoted from start and end.
 *
 * For each point in the assigned range:
 *    - Compute the squared Euclidean distance to each centroid.
 *    - Determine the nearest centroid.
 *    - Update the local assignment and count any change in the cluster assignment.
 */
void *assign_points_thread(void *arg)
{
    assign_thread_data_t *data = (assign_thread_data_t *)arg;
    int start = (*data).start;
    int end = (*data).end;
    int samples = (*data).samples;
    int K = (*data).K;
    (*data).local_changes = 0;

    // For each point...
    for (int i = start; i < end; i++)
    {
        int class_int = 1;
        float minDist = FLT_MAX;
        // For each cluster...
        for (int j = 0; j < K; j++)
        {
            float dist = 0.0f;

            // For each dimension...
            for (int d = 0; d < samples; d++)
            {
                float diff = data->local_points[i * samples + d] - data->centroids[j * samples + d];
                dist += diff * diff;
            }

            // Check if distance is smaller than the one found so far
            if (dist < minDist)
            {
                minDist = dist;
                class_int = j + 1;
            }
        }

        // If there has been a change, add a local change
        if ((*data).local_classMap[i] != class_int)
        {
            (*data).local_changes++;
        }

        // Change class of data point
        (*data).local_classMap[i] = class_int;
    }
    return NULL; 
}

/*
 * Pthread function that computes partial sums for updating centroids.
 *
 * Each thread processes its assigned block of points:
 *    - For each point, it adds the point's coordinate values to the appropriate centroid accumulator.
 *    - It also counts the number of points assigned to each centroid.
 */
void *centroid_accumulate_thread(void *arg)
{
    centroid_thread_data_t *data = (centroid_thread_data_t *)arg;
    int start = (*data).start;
    int end = (*data).end;
    int samples = (*data).samples;
    int K = (*data).K;

    // Initialize the thread’s partial arrays to zero
    for (int k = 0; k < K; k++)
    {
        (*data).partial_counts[k] = 0;
        for (int d = 0; d < samples; d++)
        {
            (*data).partial_centroids[k * samples + d] = 0.0f;
        }
    }

    // Accumulate contributions from the assigned points
    for (int i = start; i < end; i++)
    {
        int cls = (*data).local_classMap[i]; // classes are 1-indexed
        int idx = cls - 1;
        (*data).partial_counts[idx]++;
        for (int d = 0; d < samples; d++)
        {
            (*data).partial_centroids[idx * samples + d] += (*data).local_points[i * samples + d];
        }
    }
    return NULL;
}

int main(int argc, char *argv[])
{
    // Initialize MPI
    MPI_Init(&argc, &argv);
    int rank, size;
	// Get the rank of the current process
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // Get the total number of processes
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    // Set the error handler for MPI_COMM_WORLD to return errors instead of aborting
    MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);

    // START CLOCK***************************************
	double start, end;
	start = MPI_Wtime();
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
	 * argv[7]: (OPTIONAL) Number of threads for PThreads
     * */

    if ((argc != 7) && !(argc == 8))
    {
        if (rank == 0)
        {
            fprintf(stderr, "EXECUTION ERROR MPI+PThreads: Parameters are not correct.\n");
            fprintf(stderr, "./KMEANS [Input Filename] [Number of clusters] [Number of iterations] [Number of changes] [Threshold] [Output data file] Optional: [Number of Threads]\n");
            fflush(stderr);
        }

        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    // Set the number of default Pthreads threads
    int threads = 8;
    // Can be eventually passed as a parameter
    if (argc == 8)
    {
        threads = atoi(argv[7]);
    }

	// Reading the input data on the root process (rank 0)
	// lines = number of points; samples = number of dimensions per point
    int lines = 0, samples = 0;
    float *points = NULL;

    if (rank == 0)
    {
        int error = readInput(argv[1], &lines, &samples);
        if (error != 0)
        {
            showFileError(error, argv[1]);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        points = (float *)calloc(lines * samples, sizeof(float));
        if (points == NULL)
        {
            fprintf(stderr, "Memory allocation error.\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        error = readInput2(argv[1], points);
        if (error != 0)
        {
            showFileError(error, argv[1]);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
    }

	// Broadcast the values of lines (data points) and samples (dimensions) to all processes
    MPI_Bcast(&lines, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&samples, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Everyone gets the arguments of the program
    int K = atoi(argv[2]);
    int maxIterations = atoi(argv[3]);
    int minChanges = (int)(lines * atof(argv[4]) / 100.0);
    float maxThreshold = atof(argv[5]);

    float *centroids = (float *)calloc(K * samples, sizeof(float));
    if (centroids == NULL)
    {
        fprintf(stderr, "Memory allocation error.\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    int *classMap = NULL;

    // Rank 0 must initialize centroids and class mappings, all other processes will get the arrays from it
    if (rank == 0)
    {
        int *centroidPos = (int *)calloc(K, sizeof(int));
        classMap = (int *)calloc(lines, sizeof(int));
        if (centroidPos == NULL || classMap == NULL)
        {
            fprintf(stderr, "Memory allocation error.\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        srand(0);
        for (int i = 0; i < K; i++)
            centroidPos[i] = rand() % lines;

		// Loading the array of initial centroids with the data from the array data
		// The centroids are points stored in the data array.
        initCentroids(points, centroids, centroidPos, samples, K);
        free(centroidPos);

        printf("\n\tData file: %s \n\tPoints: %d\n\tDimensions: %d\n", argv[1], lines, samples);
        printf("\tNumber of clusters: %d\n", K);
        printf("\tMaximum number of iterations: %d\n", maxIterations);
        printf("\tMinimum number of changes: %d [%g%% of %d points]\n", minChanges, atof(argv[4]), lines);
        printf("\tMaximum centroid precision: %f\n", maxThreshold);
    }

	// Broadcast the initial centroids to all processes
    MPI_Bcast(centroids, K * samples, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // END CLOCK*****************************************
	end = MPI_Wtime();
	printf("\n%d |Memory allocation: %f seconds\n", rank, end - start);
	fflush(stdout);
	//**************************************************
	// START CLOCK***************************************
	MPI_Barrier(MPI_COMM_WORLD); // Ensure that all processes start timer at the same time
	start = MPI_Wtime();
	//**************************************************

    char *outputMsg = NULL;
    if (rank == 0)
    {
        outputMsg = (char *)calloc(10000, sizeof(char));
    }

    int it = 0;
    int changes;
    float maxDist;
    
    // pointPerClass: number of points classified in each class
	// auxCentroids: mean of the points in each class
    int *pointsPerClass = (int *)malloc(K * sizeof(int));
    float *auxCentroids = (float *)malloc(K * samples * sizeof(float));
    if (pointsPerClass == NULL || auxCentroids == NULL)
    {
        fprintf(stderr, "Memory allocation error.\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

	//  VALUES NEEDED FOR STEP 1: Distribute data points among processes
    int *sendcounts = (int *)malloc(size * sizeof(int));
    int *displs = (int *)malloc(size * sizeof(int));
    int remainder = lines % size;
    int sum = 0;

    for (int i = 0; i < size; ++i)
    {
        sendcounts[i] = (lines / size) * samples;
        if (i < remainder)
            sendcounts[i] += samples; // Distribute the remainder among the first 'remainder' processes
        displs[i] = sum;
        sum += sendcounts[i];
    }

	// Works also with odd number of processes / points
	// Calculate the number of local lines (data points) for each process
    int local_n = sendcounts[rank] / samples;
   	// Allocate memory for local data points and their class assignments
    float *local_points = (float *)calloc(local_n * samples, sizeof(float));
    int *local_classMap = (int *)calloc(local_n, sizeof(int));

    if (local_points == NULL || local_classMap == NULL)
    {
        fprintf(stderr, "Memory allocation error.\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

	// Scatter the data points from the root process to all processes
	// MPI_Scatterv allows varying counts of data to be sent to each process
    MPI_Scatterv(points, sendcounts, displs, MPI_FLOAT, local_points, sendcounts[rank], MPI_FLOAT, 0, MPI_COMM_WORLD);

	//  VALUES NEEDED FOR STEP 2: Distribute centroid updates among processes
    int *centroid_sendcounts = (int *)malloc(size * sizeof(int));
    int *centroid_displs = (int *)malloc(size * sizeof(int));
    int centroid_remainder = K % size;
    sum = 0;
    for (int i = 0; i < size; ++i)
    {
        centroid_sendcounts[i] = (K / size) * samples;
        if (i < centroid_remainder)
            centroid_sendcounts[i] += samples; // Distribute remainder centroids
        centroid_displs[i] = sum;
        sum += centroid_sendcounts[i];
    }

    int local_k = centroid_sendcounts[rank] / samples;  // Number of centroids handled by this process
    // Allocate memory for local centroid updates
    float *local_centroids = (float *)calloc(local_k * samples, sizeof(float));
    if (local_centroids == NULL)
    {
        fprintf(stderr, "Memory allocation error.\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    do
    {
        it++; // Increment iteration counter

        /* -------------------------------------------------------------------
		 *  STEP 1: Assign points to nearest centroid (using Pthreads)
		 *	Calculate the distance from each point to the centroid
		 *	Assign each point to the nearest centroid.
		 ------------------------------------------------------------------- */
        
        pthread_t assign_threads[threads];
        assign_thread_data_t assign_data[threads];
        int points_per_thread = local_n / threads;
        int extra = local_n % threads; // Distribute any remaining points among the first threads
        int current_start = 0;

        // Create threads for point assignment
        for (int t = 0; t < threads; t++)
        {
            assign_data[t].thread_id = t;
            assign_data[t].num_threads = threads;
            assign_data[t].local_n = local_n;
            assign_data[t].samples = samples;
            assign_data[t].K = K;
            assign_data[t].local_points = local_points;
            assign_data[t].local_classMap = local_classMap;
            assign_data[t].centroids = centroids;
            assign_data[t].start = current_start;
            int count = points_per_thread + (t < extra ? 1 : 0);
            assign_data[t].end = current_start + count;
            current_start += count;
            pthread_create(&assign_threads[t], NULL, assign_points_thread, &assign_data[t]);
        }

        int local_changes = 0;
        // Wait for all assignment threads to complete and sum up their changes
        for (int t = 0; t < threads; t++)
        {
            pthread_join(assign_threads[t], NULL);
            local_changes += assign_data[t].local_changes;
        }

		// Gather all the changes from each process and sum them up
        MPI_Request MPI_REQUEST; // Handle for the non-blocking reduction
		// MPI_Iallreduce initiates a non-blocking reduction operation where all processes contribute
		// their local_changes, and the sum is stored in 'changes' for all the process
        MPI_Iallreduce(&local_changes, &changes, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD, &MPI_REQUEST);

        /* -------------------------------------------------------------------
         * STEP 2: Recalculate centroids by accumulating local contributions
         * 
         * Pthreads are used to sum the contributions from each data point for the centroids.
         * Each thread computes partial sums and counts for its assigned subset.
		 ------------------------------------------------------------------- */
        
        int *partial_counts_all = malloc(threads * K * sizeof(int));
        float *partial_centroids_all = malloc(threads * K * samples * sizeof(float));
        pthread_t centroid_threads[threads];
        centroid_thread_data_t centroid_data[threads];
        int points_per_thread2 = local_n / threads;
        extra = local_n % threads;
        current_start = 0;

        // Create threads for accumulating centroid contributions
        for (int t = 0; t < threads; t++)
        {
            centroid_data[t].thread_id = t;
            centroid_data[t].num_threads = threads;
            centroid_data[t].local_n = local_n;
            centroid_data[t].samples = samples;
            centroid_data[t].K = K;
            centroid_data[t].local_points = local_points;
            centroid_data[t].local_classMap = local_classMap;
            // Each thread uses its own portion of the partial arrays
            centroid_data[t].partial_counts = partial_counts_all + t * K;
            centroid_data[t].partial_centroids = partial_centroids_all + t * K * samples;
            centroid_data[t].start = current_start;
            int count = points_per_thread2 + (t < extra ? 1 : 0);
            centroid_data[t].end = current_start + count;
            current_start += count;
            pthread_create(&centroid_threads[t], NULL, centroid_accumulate_thread, &centroid_data[t]);
        }

        // Wait for all centroid accumulation threads to finish
        for (int t = 0; t < threads; t++)
        {
            pthread_join(centroid_threads[t], NULL);
        }

        // Combine partial results from all threads into global local accumulators:
        // pointsPerClass: counts for each centroid, and auxCentroids: summed coordinate values.
        zeroIntArray(pointsPerClass, K);
        zeroFloatMatriz(auxCentroids, K, samples);
        for (int t = 0; t < threads; t++)
        {
            for (int k = 0; k < K; k++)
            {
                pointsPerClass[k] += partial_counts_all[t * K + k];
                for (int d = 0; d < samples; d++)
                {
                    auxCentroids[k * samples + d] += partial_centroids_all[t * K * samples + k * samples + d];
                }
            }
        }

        free(partial_counts_all);
        free(partial_centroids_all);

        // MPI_Allreduce: Sum the centroid accumulations from all MPI processes.
        MPI_Allreduce(MPI_IN_PLACE, pointsPerClass, K, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, auxCentroids, K * samples, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

        /* -------------------------------------------------------------------
         * STEP 3: Update centroids and check convergence
         * 
         * Each MPI process updates the centroids assigned to it by averaging the accumulated sums.
         * The maximum distance between old and new centroid positions is computed for the convergence test.
		 ------------------------------------------------------------------- */

        float local_maxDist = 0.0f;
        // Each process is assigned a portion of the centroids for update.
        for (int i = 0; i < local_k; i++)
        {
            int global_idx = centroid_displs[rank] / samples + i;
            if (global_idx >= K)
                break;

            float distance = 0.0f;
            // If no points were assigned to the centroid, skip update
            if (pointsPerClass[global_idx] == 0)
                continue;

            // For each dimension, compute the new centroid coordinate and the squared difference
            for (int j = 0; j < samples; j++)
            {
                float new_val = auxCentroids[global_idx * samples + j] / pointsPerClass[global_idx];
                float diff = centroids[global_idx * samples + j] - new_val;
                distance += diff * diff;
                local_centroids[i * samples + j] = new_val;
            }

            // Update local maximum centroid movement if needed
            if (distance > local_maxDist)
                local_maxDist = distance;
        }

		// Reduce to find the maximum distance across all processes
        MPI_Allreduce(&local_maxDist, &maxDist, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
        
        // Gather all local centroids into the global centroids array
        MPI_Allgatherv(local_centroids, local_k * samples, MPI_FLOAT,
                       centroids, centroid_sendcounts, centroid_displs, MPI_FLOAT, MPI_COMM_WORLD);
        // MPI_Allgatherv gathers variable amounts of data from all processes and distributes
		// the combined data to all processes. This updates the centroids for the next iteration.

        // Wait if the non-blocking reduction didn't complete
        MPI_Wait(&MPI_REQUEST, MPI_STATUS_IGNORE);

    } while ((changes > minChanges) && (it < maxIterations) && (maxDist > maxThreshold * maxThreshold));

	// Prepare to gather the class assignments from all processes
    int *recvcounts = (int *)malloc(size * sizeof(int));
    int *rdispls = (int *)malloc(size * sizeof(int));
    sum = 0;
    for (int i = 0; i < size; ++i)
    {
        recvcounts[i] = sendcounts[i] / samples; // Number of points per process
        rdispls[i] = sum;
        sum += recvcounts[i];
    }

	// Gather all local_classMap arrays from each process into the global classMap array on the root process
    MPI_Gatherv(local_classMap, local_n, MPI_INT,
                classMap, recvcounts, rdispls, MPI_INT, 0, MPI_COMM_WORLD);

	// 	Output and termination conditions
    if (rank == 0)
    {
        printf("%s", outputMsg);
    }

    // END CLOCK*****************************************
	end = MPI_Wtime();
	// Reduce to get the maximum time across all processes
	double computation_time = end - start;
	double max_computation_time;
	MPI_Reduce(&computation_time, &max_computation_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
	// Thread 0 print the maximum computation time
	if (rank == 0)
	{
		printf("\n Computation: %f seconds\n", max_computation_time);
		fflush(stdout);
	}
	//**************************************************
	// START CLOCK***************************************
	MPI_Barrier(MPI_COMM_WORLD); // Ensure that all processes start timer at the same time
	start = MPI_Wtime();
	//**************************************************

    if (rank == 0)
    {
        if (changes <= minChanges)
        {
            printf("\n\nTermination condition: Minimum number of changes reached: %d [%d]", changes, minChanges);
        }
        else if (it >= maxIterations)
        {
            printf("\n\nTermination condition: Maximum number of iterations reached: %d [%d]", it, maxIterations);
        }
        else
        {
            printf("\n\nTermination condition: Centroid update precision reached: %g [%g]", maxDist, maxThreshold);
        }

        int error = writeResult(classMap, lines, argv[6]);
        if (error != 0)
        {
            showFileError(error, argv[6]);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        fflush(stdout);
    }

    // Free memory
    free(local_points);
    free(local_classMap);
    free(local_centroids);
    free(sendcounts);
    free(displs);
    free(centroid_sendcounts);
    free(centroid_displs);
    free(recvcounts);
    free(rdispls);
    free(centroids);
    free(pointsPerClass);
    free(auxCentroids);

	//	Free memory on the root process
    if (rank == 0)
    {
        free(points);
        free(classMap);
        free(outputMsg);
    }

    // END CLOCK*****************************************
	end = MPI_Wtime();
	printf("\n\n%d |Memory deallocation: %f seconds\n", rank, end - start);
	fflush(stdout);
	//***************************************************/

	//	FINALIZE: Clean up the MPI environment
	MPI_Finalize();
	return 0;
}
