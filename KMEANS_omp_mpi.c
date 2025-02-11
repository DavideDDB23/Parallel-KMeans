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

// Use a padding factor to separate cluster counters in memory (to reduce false sharing)
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
    int contlines = 0, contsamples = 0;

    if ((fp = fopen(filename, "r")) != NULL)
    {
        while (fgets(line, MAXLINE, fp) != NULL)
        {
            if (strchr(line, '\n') == NULL)
            {
                fclose(fp);
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
int writeResult(int *classMap, int lines, const char *filename, double max_computation_time)
{
    FILE *fp;

    if ((fp = fopen(filename, "wt")) != NULL)
    {
        for (int i = 0; i < lines; i++)
        {
            fprintf(fp, "%d\n", classMap[i]);
        }

        fprintf(fp, "Computation: %f seconds\n", max_computation_time);
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
    for (int i = 0; i < K; i++)
    {
        int idx = centroidPos[i];
        memcpy(&centroids[i * samples], &data[idx * samples], samples * sizeof(float));
    }
}

/*
Function euclideanDistance: Euclidean distance (squared)
*/
float euclideanDistance(float *point, float *center, int samples)
{
    float dist = 0.0;
    for (int i = 0; i < samples; i++)
    {
        float diff = point[i] - center[i];
        dist = fmaf(diff, diff, dist);
    }
    return dist; // Squared distance
}

int main(int argc, char *argv[])
{
    int provided;
    // Request MPI thread support (FUNNELED: only main thread will make MPI calls)
    MPI_Init_thread(NULL, NULL, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED)
    {
        fprintf(stderr, "Error: MPI does not provide required thread support level\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);

    // START CLOCK: Memory allocation and I/O
    double start, end;
    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();

    /*
     * PARAMETERS
     *
     * argv[1]: Input data file
     * argv[2]: Number of clusters (K)
     * argv[3]: Maximum number of iterations
     * argv[4]: Minimum percentage of class changes (termination condition)
     * argv[5]: Threshold for centroid precision (termination condition)
     * argv[6]: Output file (cluster assignments)
     * argv[7]: (OPTIONAL) Number of OpenMP threads
     */
    if ((argc != 7) && (argc != 8))
    {
        if (rank == 0)
        {
            fprintf(stderr, "EXECUTION ERROR K-MEANS: Parameters are not correct.\n");
            fprintf(stderr, "./KMEANS [Input Filename] [Number of clusters] [Number of iterations] [Percentage changes] [Threshold] [Output data file] [Optional: Number of OpenMP threads]\n");
            fflush(stderr);
        }
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    // Set the number of OpenMP threads (default 8)
    int threads = 8;
    if (argc == 8)
    {
        threads = atoi(argv[7]);
    }
    omp_set_num_threads(threads);

    // Reading the input data on the root process (rank 0)
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
            fprintf(stderr, "Memory allocation error (data).\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        error = readInput2(argv[1], data);
        if (error != 0)
        {
            showFileError(error, argv[1]);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
    }

    // Broadcast number of points (lines) and dimensions (samples) to all processes
    MPI_Bcast(&lines, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&samples, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Program parameters
    int K = atoi(argv[2]);
    int maxIterations = atoi(argv[3]);
    int minChanges = (int)(lines * atof(argv[4]) / 100.0);
    float maxThreshold = atof(argv[5]);

    float *centroids = (float *)calloc(K * samples, sizeof(float));
    if (centroids == NULL)
    {
        fprintf(stderr, "Memory allocation error (centroids).\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    int *classMap = NULL;

    // Rank 0 initializes centroids and classMap
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
        initCentroids(data, centroids, centroidPos, samples, K);
        free(centroidPos);

        printf("\n\tData file: %s \n\tPoints: %d\n\tDimensions: %d\n", argv[1], lines, samples);
        printf("\tNumber of clusters: %d\n", K);
        printf("\tMaximum number of iterations: %d\n", maxIterations);
        printf("\tMinimum number of changes: %d [%g%% of %d points]\n", minChanges, atof(argv[4]), lines);
        printf("\tMaximum centroid precision: %f\n", maxThreshold);
    }
    // Broadcast initial centroids to all processes
    MPI_Bcast(centroids, K * samples, MPI_FLOAT, 0, MPI_COMM_WORLD);

    end = MPI_Wtime();
    printf("\n%d |Memory allocation: %f seconds\n", rank, end - start);
    fflush(stdout);

    // Distribute the data among MPI processes
    int *sendcounts = (int *)malloc(size * sizeof(int));
    int *displs = (int *)malloc(size * sizeof(int));
    int remainder = lines % size;
    int sum = 0;
    for (int i = 0; i < size; ++i)
    {
        sendcounts[i] = (lines / size) * samples;
        if (i < remainder)
            sendcounts[i] += samples; // Distribute the remainder
        displs[i] = sum;
        sum += sendcounts[i];
    }
    int local_lines = sendcounts[rank] / samples;
    float *local_data = (float *)calloc(local_lines * samples, sizeof(float));
    int *local_classMap = (int *)calloc(local_lines, sizeof(int));
    if (local_data == NULL || local_classMap == NULL)
    {
        fprintf(stderr, "Memory allocation error (local arrays).\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    MPI_Scatterv(data, sendcounts, displs, MPI_FLOAT, local_data, sendcounts[rank], MPI_FLOAT, 0, MPI_COMM_WORLD);

    // --- Allocate arrays for centroid recalculation ---
    // Allocate the pointsPerClass array with padding to avoid false sharing.
    int *pointsPerClass = NULL;
    if (posix_memalign((void **)&pointsPerClass, 64, sizeof(int) * K * PADDING) != 0)
    {
        fprintf(stderr, "posix_memalign for pointsPerClass failed.\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    // Allocate auxCentroids normally.
    float *auxCentroids = (float *)malloc(K * samples * sizeof(float));
    if (auxCentroids == NULL)
    {
        fprintf(stderr, "Memory allocation error (auxCentroids).\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    // Allocate the contiguous array for the padded counters once (outside the loop)
    int *pointsPerClassContig = (int *)malloc(K * sizeof(int));
    if (pointsPerClassContig == NULL)
    {
        fprintf(stderr, "Memory allocation error (pointsPerClassContig).\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    char *outputMsg = NULL;
    if (rank == 0)
    {
        outputMsg = (char *)calloc(10000, sizeof(char));
    }

    int it = 0;
    int changes;
    float maxDist;

    // --- Main k-Means Iteration Loop ---
    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();
    do
    {
        it++; // Increment iteration counter

        /* -------------------------------------------------------------------
         *  STEP 1: Assign points to nearest centroid
         *  For each local point, compute the distance to each centroid and assign the nearest.
         ------------------------------------------------------------------- */
        int local_changes = 0; // Local counter for changes in cluster assignments

        #pragma omp parallel for reduction(+:local_changes) schedule(static)
        for (int i = 0; i < local_lines; i++)
        {
            int best_class = 1;
            float minDist = FLT_MAX;
            for (int j = 0; j < K; j++)
            {
                float dist = euclideanDistance(&local_data[i * samples], &centroids[j * samples], samples);
                if (dist < minDist)
                {
                    minDist = dist;
                    best_class = j + 1;
                }
            }
            if (local_classMap[i] != best_class)
                local_changes++;
            local_classMap[i] = best_class;
        }

        // Non-blocking reduction to sum local_changes across all processes.
        MPI_Request MPI_REQUEST;
        MPI_Iallreduce(&local_changes, &changes, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD, &MPI_REQUEST);

        /* -------------------------------------------------------------------
         *  STEP 2: Recalculate centroids (cluster means)
         *  Accumulate the coordinates for each cluster.
         ------------------------------------------------------------------- */
#pragma omp for schedule(static)
			// Reset pointsPerClass and auxCentroids in a single loop
			for (int i = 0; i < K * samples; i++)
			{
				if (i < K)
				{
					pointsPerClass[i * PADDING] = 0;
				}
				auxCentroids[i] = 0.0f;
			}

        // Accumulate local contributions.
        // Note: Use the padded index: (class - 1)*PADDING.
        #pragma omp parallel for reduction(+:pointsPerClass[0:K * PADDING], auxCentroids[0:K * samples]) schedule(static)
        for (int i = 0; i < local_lines; i++)
        {
            int class = local_classMap[i];
            pointsPerClass[(class - 1) * PADDING]++;  // Update only the first element of each padded block.
            for (int j = 0; j < samples; j++)
            {
                auxCentroids[(class - 1) * samples + j] += local_data[i * samples + j];
            }
        }

         // Before MPI reduction, copy padded counters into the contiguous array.
        for (int i = 0; i < K; i++)
        {
            pointsPerClassContig[i] = pointsPerClass[i * PADDING];
        }
        // Reduce the cluster counts and centroid sums across all processes.
        MPI_Allreduce(MPI_IN_PLACE, pointsPerClassContig, K, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, auxCentroids, K * samples, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        // Copy the reduced counts back into the padded array.
        for (int i = 0; i < K; i++)
        {
            pointsPerClass[i * PADDING] = pointsPerClassContig[i];
        }

        /* -------------------------------------------------------------------
         *  STEP 3: Check convergence and update centroids.
         *  Compute the maximum change between old and new centroids.
         ------------------------------------------------------------------- */
        float local_maxDist = 0.0f;
        #pragma omp parallel for reduction(max:local_maxDist) schedule(static)
        for (int i = 0; i < K; i++)
        {
            // Avoid division by zero.
            if (pointsPerClass[i * PADDING] > 0){

                float distance = 0.0f;
                for (int j = 0; j < samples; j++)
                {
                    // Use the contiguous count from the padded array.
                    float new_val = auxCentroids[i * samples + j] / pointsPerClass[i * PADDING];
                    distance = fmaf(centroids[i * samples + j] - new_val,
                                    centroids[i * samples + j] - new_val,
                                    distance);
                    centroids[i * samples + j] = new_val;
                }
                if (distance > local_maxDist)
                    local_maxDist = distance;
            }
        }
        MPI_Allreduce(&local_maxDist, &maxDist, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);

        // Wait for the non-blocking reduction of changes to complete.
        MPI_Wait(&MPI_REQUEST, MPI_STATUS_IGNORE);

    } while ((changes > minChanges) && (it < maxIterations) && (maxDist > pow(maxThreshold, 2)));

    // --- Gather Class Assignments ---
    int *recvcounts = (int *)malloc(size * sizeof(int));
    int *rdispls = (int *)malloc(size * sizeof(int));
    sum = 0;
    for (int i = 0; i < size; ++i)
    {
        recvcounts[i] = sendcounts[i] / samples;
        rdispls[i] = sum;
        sum += recvcounts[i];
    }
    MPI_Gatherv(local_classMap, local_lines, MPI_INT,
                classMap, recvcounts, rdispls, MPI_INT,
                0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    end = MPI_Wtime();
    double computation_time = end - start;
    double max_computation_time;
    MPI_Reduce(&computation_time, &max_computation_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0)
    {
        printf("\n Computation: %f seconds\n", max_computation_time);
        fflush(stdout);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();
    if (rank == 0)
    {
        if (changes <= minChanges)
            printf("\n\nTermination condition: Minimum number of changes reached: %d [%d]", changes, minChanges);
        else if (it >= maxIterations)
            printf("\n\nTermination condition: Maximum number of iterations reached: %d [%d]", it, maxIterations);
        else
            printf("\n\nTermination condition: Centroid update precision reached: %g [%g]", maxDist, maxThreshold);

        int error = writeResult(classMap, lines, argv[6], max_computation_time);
        if (error != 0)
        {
            showFileError(error, argv[6]);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
    }

    // --- Free Memory ---
    free(local_data);
    free(local_classMap);
    free(sendcounts);
    free(displs);
    free(recvcounts);
    free(rdispls);
    if (rank == 0)
    {
        free(data);
        free(classMap);
        free(outputMsg);
    }
    free(centroids);
    free(pointsPerClass);
    free(auxCentroids);

    end = MPI_Wtime();
    printf("\n\n%d |Memory deallocation: %f seconds\n", rank, end - start);
    fflush(stdout);

    MPI_Finalize();
    return 0;
}