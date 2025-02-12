/*
 * k-Means clustering algorithm
 *
 * OpenMP version
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
	// START CLOCK***************************************
	double start, end;
	start = omp_get_wtime();
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
	if ((argc != 7) && !(argc == 8))
	{
		fprintf(stderr, "EXECUTION ERROR K-MEANS: Parameters are not correct.\n");
		fprintf(stderr, "./KMEANS [Input Filename] [Number of clusters] [Number of iterations] [Number of changes] [Threshold] [Output data file] Optional: [Number of Threads]\n");
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
		fprintf(stderr, "Memory allocation error (data).\n");
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

	// pointPerClass: number of points classified in each class
	// auxCentroids: mean of the points in each class
	int *pointsPerClass = NULL;
	float *auxCentroids = NULL;
	// Allocate padded memory for pointsPerClass to minimize false sharing (each class gets extra cache line space)
	if (posix_memalign((void **)&pointsPerClass, 64, sizeof(int) * K * PADDING) != 0)
	{
		fprintf(stderr, "posix_memalign for pointsPerClass failed.\n");
		exit(-4);
	}
	// Allocate aligned memory for auxCentroids on a 64-byte boundary for efficient cache access
	if (posix_memalign((void **)&auxCentroids, 64, sizeof(float) * K * samples) != 0)
	{
		fprintf(stderr, "posix_memalign for auxCentroids failed.\n");
		exit(-4);
	}

	// Set the number of OpenMP threads
	int nThreads = 4;
	if (argc == 8)
	{
		nThreads = atoi(argv[7]); // Set thread count from command-line argument
	}
	omp_set_num_threads(nThreads);

	// END CLOCK*****************************************
	end = omp_get_wtime();
	printf("\nMemory allocation: %f seconds\n", end - start);
	fflush(stdout);
	//**************************************************
	// START CLOCK***************************************
	start = omp_get_wtime();
	//**************************************************
	char *outputMsg = (char *)calloc(10000, sizeof(char));

	int it = 0;
	int changes = 0;
	float maxDist = 0.0f;
	int terminate = 0; // boolean to control the loop

	// Start the parallel region
	#pragma omp parallel default(none)                                     \
	firstprivate(lines, samples, K)                                        \
	shared(data, centroids, auxCentroids, classMap, pointsPerClass,        \
			   it, changes, maxDist, terminate, minChanges, maxIterations, \
			   maxThreshold)
	{
		while (1)
		{
			// Barrier so all threads check 'terminate' flag before proceeding
			#pragma omp barrier
			if (terminate)
			{
				break; // Exit the while loop in all threads, termination conditions met
			}

			/* -------------------------------------------------------------------
			*  STEP 1: Assign points to the nearest centroid
			*    
			*  Single thread increments iteration counter and resets 'changes'.
			*  #pragma omp for with a reduction to count how many points get reassigned 
			*  to a different cluster.
			------------------------------------------------------------------- */

			#pragma omp single
			{
				// Only one thread does this.
				it++;		 // Increment iteration counter
				changes = 0; // Counter for changes in cluster assignments
			}

			// 'changes' is shared, but 'reduction(+ : changes)' ensures that each 
        	// thread accumulates its own local count of reassignments, and 
        	// OpenMP sums them at the end of the loop.
			#pragma omp for reduction(+ : changes) schedule(static)
			// For each point...
			for (int i = 0; i < lines; i++)
			{
				int class = 1;
				float minDist = FLT_MAX;

				// For each centroid...
				for (int j = 0; j < K; j++)
				{
					// Compute l_2 (squared, without sqrt)
					float dist = euclideanDistance(&data[i * samples], &centroids[j * samples], samples);

					// If the distance is smallest so far, update minDist and the class of the point
					if (dist < minDist)
					{
						minDist = dist;
						class = j + 1;
					}
				}

				// If the class changed, increment the local change counter
				if (classMap[i] != class)
				{
					changes++;
				}

				// Assign the new class to the point
				classMap[i] = class;
			}

			/* -------------------------------------------------------------------
			* STEP 2: Recomput centroids (cluster means)
			*    
			* Sum all points belonging to a cluster in auxCentroids, 
			* and keep counts of how many points in pointsPerClass
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

			// For each point, add its coordinates to the corresponding cluster's entry in auxCentroids,
			// and increment the cluster count in pointsPerClass.
			// Reduction clause with array sections ensures each thread uses its own local
        	// copy of the arrays, and then OpenMP combines them element-wise at the end.
			#pragma omp for schedule(static) \
			reduction(+ : pointsPerClass[ : (K * PADDING)], auxCentroids[ : (K * samples)])
			for (int i = 0; i < lines; i++)
			{
				int class = classMap[i] - 1;
				pointsPerClass[class * PADDING]++;

				for (int j = 0; j < samples; j++)
				{
					auxCentroids[class * samples + j] += data[i * samples + j];
				}
			}

			/* -------------------------------------------------------------------
			*  STEP 3: Check convergence
			*  
			*  Compute the maximum distance between old and new centroids
			 ------------------------------------------------------------------- */
			
			#pragma omp single
			{
				// Only one thread does this.
				maxDist = 0.0f;
			}

			// 'maxDist' is shared, but 'reduction(+ : maxDist)' ensures that each 
        	// thread computes its own local maximum distance, and 
        	// OpenMP takes the maximum of these values at the end of the loop.
			#pragma omp for reduction(max : maxDist) schedule(static)
			// For each centroid...
			for (int i = 0; i < K; i++)
			{
				// Only update the centroid if there is at least one point assigned to it.
				if (pointsPerClass[i * PADDING] != 0)
				{	
					// For each centroid's dimension...
					for (int j = 0; j < samples; j++)
					{
						auxCentroids[i * samples + j] /= (float)pointsPerClass[i * PADDING];
					}

					float dist = euclideanDistance(&centroids[i * samples],
												   &auxCentroids[i * samples],
												   samples);
					if (dist > maxDist)
					{
						maxDist = dist;
					}
				}
			}

			#pragma omp single
			{
				// Only one thread does this.

				// Copy new centroids into the real centroids array -- Update centroids
				memcpy(centroids, auxCentroids, sizeof(float) * K * samples);

				// If convergence criteria are met, we set 'terminate = 1', so that in the
				// next iteration all threads will exit from the while loop.
				if ((changes <= minChanges) || (it >= maxIterations) || (maxDist <= pow(maxThreshold, 2)))
				{
					terminate = 1;
				}
			}
		}
	}

	// Output and termination conditions
	printf("%s", outputMsg);

	// END CLOCK*****************************************
	end = omp_get_wtime();
	float computationTime = end - start;
	printf("\nComputation: %f seconds", computationTime);
	fflush(stdout);
	//**************************************************
	// START CLOCK***************************************
	start = omp_get_wtime();
	//**************************************************

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

	// Writing the classification of each point to the output file.
	error = writeResult(classMap, lines, argv[6]);
	if (error != 0)
	{
		showFileError(error, argv[6]);
		exit(error);
	}

	// Free memory
	free(data);
	free(classMap);
	free(centroids);
	free(pointsPerClass);
	free(auxCentroids);

	// END CLOCK*****************************************
	end = omp_get_wtime();
	printf("\n\nMemory deallocation: %f seconds\n", end - start);
	fflush(stdout);
	//***************************************************/
	return 0;
}