/*
 * k-Means clustering algorithm
 *
 * MPI+OpenMP hybrid version
 *
 * Parallel computing (Degree in Computer Engineering)
 * 2022/2023
 *
 * Version: 1.1 (Hybrid MPI+OMP with padded cluster counters)
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
#include <omp.h>

#define MAXLINE 2000
#define MAXCAD 200

// Macros
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

// Padding factor of 16 so that each element in the array occupies 16 ints (64 bytes), to prevent false sharing.
#define PADDING 16

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
        return -2; // No file found
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
        return -3; // No file found
    }
}

/*
Function initCentroids: This function copies the values of the initial centroids, using their
position in the input data structure as a reference map.
*/
void initCentroids(const float *data, float *centroids, int *centroidPos, int samples, int K)
{
    int i;
	int idx;
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
		// Fused multiply-add, computes (a * b) + c in a single, efficient step
		// Reduce rounding errors, for consistency of results across all implementations
		dist = fmaf(point[i] - center[i], point[i] - center[i], dist);
	}
	return dist; // Squared distance
}

int main(int argc, char *argv[])
{
    // Initialize MPI
    int provided;
	// MPI_THREAD_FUNNELED allows the process to call MPI functions only from the main thread.
    MPI_Init_thread(NULL, NULL, MPI_THREAD_FUNNELED, &provided);
    // Check provided thread level
    if (provided < MPI_THREAD_FUNNELED)
    {
        fprintf(stderr, "Error: MPI does not provide required thread support level\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int rank, size;
    // Get the rank of the current process
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	// Get the total number of processes
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	// Set the error handler for MPI_COMM_WORLD to return errors instead of aborting
	MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);

    // START CLOCK***************************************
	double start, end;
    MPI_Barrier(MPI_COMM_WORLD); // Synchronize all processes, so they start timer at the same time.
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
	 * argv[7]: (OPTIONAL) Number of threads for OpenMP
	 * */
    if ((argc != 7) && (argc != 8))
    {
        if (rank == 0)
        {
            fprintf(stderr, "EXECUTION ERROR K-MEANS: Parameters are not correct.\n");
		    fprintf(stderr, "./KMEANS [Input Filename] [Number of clusters] [Number of iterations] [Number of changes] [Threshold] [Output data file] Optional: [Number of Threads]\n");
            fflush(stderr);
        }
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

	// Set the number of OpenMP threads
	int nThreads = 4;
	if (argc == 8)
	{
		nThreads = atoi(argv[7]); // Set thread count from command-line argument
	}
	omp_set_num_threads(nThreads);

	// Reading the input data on the root process (rank 0)
	// lines = number of points; samples = number of dimensions per point
    int lines = 0, samples = 0;
    float *data = NULL;
    
    if (rank == 0)
    {
        int error = readInput(argv[1], &lines, &samples);
        if (error != 0)
        {
            showFileError(error, argv[1]);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        data = (float *)calloc(lines * samples, sizeof(float));
        if (data == NULL)
        {
            fprintf(stderr, "Memory allocation error.\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        error = readInput2(argv[1], data);
        if (error != 0)
        {
            showFileError(error, argv[1]);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
    }

	// Broadcast the values of lines (data points) and samples (dimensions) to all processes
	MPI_Bcast(&lines, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&samples, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Program parameters
    int K = atoi(argv[2]);
    int maxIterations = atoi(argv[3]);
    int minChanges = (int)(lines * atof(argv[4]) / 100.0);
    float maxThreshold = atof(argv[5]);

    // Allocate memory for centroids
    float *centroids = (float *)calloc(K * samples, sizeof(float));
    if (centroids == NULL)
    {
        fprintf(stderr, "Memory allocation error.\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    int *classMap = NULL;

	// Rank 0 initialize centroids and class mappings, all other processes will get the arrays from it
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
        initCentroids(data, centroids, centroidPos, samples, K);
        free(centroidPos);

        printf("\n\tData file: %s \n\tPoints: %d\n\tDimensions: %d\n", argv[1], lines, samples);
        printf("\tNumber of clusters: %d\n", K);
        printf("\tMaximum number of iterations: %d\n", maxIterations);
        printf("\tMinimum number of changes: %d [%g%% of %d points]\n", minChanges, atof(argv[4]), lines);
        printf("\tMaximum centroid precision: %f\n", maxThreshold);
    }

    // Broadcast the initial centroids to all processes so they all start with the same values.
    MPI_Bcast(centroids, K * samples, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // END CLOCK*****************************************
	end = MPI_Wtime();
	printf("\n%d |Memory allocation: %f seconds\n", rank, end - start);
	fflush(stdout);
	//**************************************************
	// START CLOCK***************************************
	MPI_Barrier(MPI_COMM_WORLD); // Synchronize all processes, so they start timer at the same time.
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

	// Arrays for computing the new centroids:
    // Allocate the pointsPerClass array with padding to avoid false sharing.
    int *pointsPerClass = NULL;
    if (posix_memalign((void **)&pointsPerClass, 64, sizeof(int) * K * PADDING) != 0)
    {
        fprintf(stderr, "posix_memalign for pointsPerClass failed.\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    float *auxCentroids = (float *)malloc(K * samples * sizeof(float));
    if (auxCentroids == NULL)
    {
        fprintf(stderr, "Memory allocation error.\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    // Allocate a contiguous array for pointsPerClass to perform an efficient MPI_Allreduce.
    // The padded pointsPerClass array is not contiguous in memory, so I copy its elements into
    // pointsPerClassContig before the MPI reduction.
    int *pointsPerClassContig = (int *)malloc(K * sizeof(int));
    if (pointsPerClassContig == NULL)
    {
        fprintf(stderr, "Memory allocation error.\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    // VALUES NEEDED FOR STEP 1: Distribute data points among processes, works also with odd number of points / processes.
	// 	Each array contains an entry for each process
    int *sendcounts = (int *)malloc(size * sizeof(int)); // Array that stores how many data points each process will receive.
	int *displs = (int *)malloc(size * sizeof(int)); // Array that store the starting index (offset) of each process’s portion in the data array.

	int remainder = lines % size;
	int sum = 0; // To calculate the starting position for each process’s data.
	for (int i = 0; i < size; ++i)
	{
		sendcounts[i] = (lines / size) * samples; // Every process receive at least (lines / size) data points.
		if (i < remainder)						  // Ensures that only the exact number of extra points (remainder) is distributed.
			sendcounts[i] += samples;			  // Give extra point (remainder), for example if there're 3 remainders, processes 0,1,2 receive one data point each.
		displs[i] = sum;						  // Store the starting index of this process’s data portion in the displs array. The first process starts at 0, and the next process starts where the previous process’s data ended.
		sum += sendcounts[i];					  // Update the sum variable by adding the number of elements assigned to this process, so next process displs is correctly calculated.
	}

	// Calculate the number of local lines (data points) for this process.
	int local_lines = sendcounts[rank] / samples;
	// Allocate memory for local data points and their corresponding class assignments.
	float *local_data = (float *)calloc(local_lines * samples, sizeof(float)); // Stores the portion of data points assigned to the process.
	int *local_classMap = (int *)calloc(local_lines, sizeof(int));			   // Stores the class assignments for the data points.
	if (local_data == NULL || local_classMap == NULL)
	{
		fprintf(stderr, "Memory allocation error.\n");
		MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
	}

	// Scatter the data points from the root process to all processes
	// MPI_Scatterv allows varying counts of data to be sent to each process
	MPI_Scatterv(data, sendcounts, displs, MPI_FLOAT, local_data, sendcounts[rank], MPI_FLOAT, 0, MPI_COMM_WORLD);

    do
    {
        it++; // Increment iteration counter

        /* -------------------------------------------------------------------
		 * STEP 1: Assign points to nearest centroid
		 *
		 * Each process computes the squared Euclidean distance (using fmaf)
         * between its local data points and all centroids. The point is assigned
         * to the cluster with the minimum distance.
		 ------------------------------------------------------------------- */
        
        int local_changes = 0; // Local counter for changes in cluster assignments

        // 'local_changes' is shared, but 'reduction(+ : local_changes)' ensures that each 
        // thread accumulates its own local count of reassignments, and 
        // OpenMP sums them at the end of the loop.
        #pragma omp parallel for reduction(+:local_changes) schedule(static)
        // For each local point...
        for (int i = 0; i < local_lines; i++)
        {
            int class = 1;
            float minDist = FLT_MAX;

            // For each centroid...
            for (int j = 0; j < K; j++)
            {
                // Compute l_2 (squared, without sqrt)
                float dist = euclideanDistance(&local_data[i * samples], &centroids[j * samples], samples);
                
                // If the distance is smallest so far, update minDist and the class of the point
                if (dist < minDist)
                {
                    minDist = dist;
                    class = j + 1;
                }
            }

            // If the class changed, increment the local change counter
            if (local_classMap[i] != class)
            {
                local_changes++;
            }
            
            // Assign the new class to the point
			local_classMap[i] = class;
        }

		// Gather all the changes from each process and sum them up
		MPI_Request MPI_REQUEST; // Handle for the non-blocking reduction
		// MPI_Iallreduce initiates a non-blocking reduction operation where all processes contribute
		// their local_changes, and the sum is stored in 'changes' for all the processes
		MPI_Iallreduce(&local_changes, &changes, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD, &MPI_REQUEST);

       /* -------------------------------------------------------------------
		 * STEP 2: Recalculate centroids (cluster means)
		 *
		 * Each process computes a partial sum (auxCentroids) and count (pointsPerClass)
         * for the data points assigned to each cluster. Then, MPI_Allreduce is used to sum
         * these partial results across all processes so that every process obtains the global sums.
		 ------------------------------------------------------------------- */
        
        #pragma omp for schedule(static)
		// Reset pointsPerClass and auxCentroids in a single loop
        for (int i = 0; i < K * samples; i++)
        {
            if (i < K)
            {
                // pointsPerClass[i * PADDING] is used to avoid false sharing.
                pointsPerClass[i * PADDING] = 0;
            }
            auxCentroids[i] = 0.0f;
        }

        // For each local points, add its coordinates to the corresponding cluster's entry in auxCentroids,
		// and increment the cluster count in pointsPerClass.
		// Reduction clause with array sections ensures each thread uses its own local
        // copy of the arrays, and then OpenMP combines them element-wise at the end.
        #pragma omp parallel for schedule(static) \
        reduction (+:pointsPerClass[ : (K * PADDING)], auxCentroids[ : (K * samples)])
        // For each local point...
        for (int i = 0; i < local_lines; i++)
        {
            int class = local_classMap[i] - 1;
            pointsPerClass[class * PADDING]++;
            
            // For each dimension...
            for (int j = 0; j < samples; j++)
            {
                auxCentroids[class * samples + j] += local_data[i * samples + j];
            }
        }

        // Before MPI reduction, copy padded pointsPerClass values into the contiguous array for MPI_Allreduce.
        for (int i = 0; i < K; i++)
        {
            pointsPerClassContig[i] = pointsPerClass[i * PADDING];
        }

        // All the processes receive the other pointsPerClassContig and auxCentroids
		// MPI_Allreduce sums up the pointsPerClass and auxCentroids from all processes and distributes result to all, 
		// MPI_IN_PLACE so each process’s local data is updated with the result of the reduction.
        MPI_Allreduce(MPI_IN_PLACE, pointsPerClassContig, K, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, auxCentroids, K * samples, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        
        // Copy the reduced pointsPerClassContig values back into the padded array.
        for (int i = 0; i < K; i++)
        {
            pointsPerClass[i * PADDING] = pointsPerClassContig[i];
        }

        /* -------------------------------------------------------------------
		 * STEP 3: Check convergence
		 *
		 * For each centroid, compute the squared difference between its old and new position.
         * The maximum squared difference (local_maxDist) is calculated locally and then reduced
         * globally using MPI_Allreduce.
		 ------------------------------------------------------------------- */
        
        float local_maxDist = 0.0f;

        // 'local_maxDist' is shared, but 'reduction(+ : local_maxDist)' ensures that each 
        // thread computes its own local maximum movement, and 
        // OpenMP takes the maximum of these values at the end of the loop.
        #pragma omp parallel for reduction(max:local_maxDist) schedule(static)
        // For each centroid...
        for (int i = 0; i < K; i++)
        {
			// Only update the centroid if there is at least one point assigned to it.
            if (pointsPerClass[i * PADDING] > 0){

                float distance = 0.0f;
                
				// For each centroid's dimension...
                for (int j = 0; j < samples; j++)
                {
                    float new_val = auxCentroids[i * samples + j] / pointsPerClass[i * PADDING];
                    distance = fmaf(centroids[i * samples + j] - new_val,
                                    centroids[i * samples + j] - new_val,
                                    distance);
                    centroids[i * samples + j] = new_val;
                }

                if (distance > local_maxDist)
                {
                    local_maxDist = distance;
                }
            }
        }

       	// Reduce to find the maximum distance across all processes
		// This ensures all process receives the largest maxDist found
		MPI_Allreduce(&local_maxDist, &maxDist, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);

		// Wait if the non-blocking reduction didn't complete
		MPI_Wait(&MPI_REQUEST, MPI_STATUS_IGNORE);

    } while ((changes > minChanges) && (it < maxIterations) && (maxDist > pow(maxThreshold, 2)));

     /*
     * Gather the final class assignments from all processes.
     *
     * Since the number of data points per process may differ,
     * MPI_Gatherv is used to collect the local_classMap arrays from each process
     * back to the root process.
     */
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
	MPI_Gatherv(local_classMap, local_lines, MPI_INT, classMap, recvcounts, rdispls, MPI_INT, 0, MPI_COMM_WORLD);

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
	// Process 0 print the maximum computation time
	if (rank == 0)
	{
		printf("\n Computation: %f seconds\n", max_computation_time);
		fflush(stdout);
	}
	//**************************************************
	// START CLOCK***************************************
	MPI_Barrier(MPI_COMM_WORLD); // Synchronize all processes, so they start timer at the same time.
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
		{
			showFileError(error, argv[6]);
			MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
		}
		fflush(stdout);
	}

	// FREE LOCAL ARRAYS: Free memory allocated for each process
    free(local_data);
    free(local_classMap);
    free(sendcounts);
    free(displs);
    free(recvcounts);
    free(rdispls);

    // Free memory on the root process
    if (rank == 0)
    {
        free(data);
        free(classMap);
        free(outputMsg);
    }

    free(centroids);
    free(pointsPerClass);
    free(auxCentroids);

	// END CLOCK*****************************************
	end = MPI_Wtime();
	printf("\n\n%d |Memory deallocation: %f seconds\n", rank, end - start);
	fflush(stdout);
	//***************************************************/

	// FINALIZE: Clean up the MPI environment
	MPI_Finalize();
	return 0;
}