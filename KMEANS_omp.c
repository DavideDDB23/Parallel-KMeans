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

// We choose 16 so that each cluster counter is 16 ints = 64 bytes apart
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
int writeResult(int *classMap, int lines, const char *filename, float computationTime)
{
	FILE *fp;

	if ((fp = fopen(filename, "wt")) != NULL)
	{
		for (int i = 0; i < lines; i++)
		{
			fprintf(fp, "%d\n", classMap[i]);
		}

		fprintf(fp, "Computation: %f seconds\n", computationTime);
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
		// dist+= (point[i]-center[i])*(point[i]-center[i]);
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

	// Aligned/padded arrays to reduce false sharing
	// We'll do a normal array reduction on them (size = K*PADDING, K*samples)
	int *pointsPerClass = NULL;
	float *auxCentroids = NULL;
	if (posix_memalign((void **)&pointsPerClass, 64, sizeof(int) * K * PADDING) != 0)
	{
		fprintf(stderr, "posix_memalign for pointsPerClass failed.\n");
		exit(-4);
	}
	if (posix_memalign((void **)&auxCentroids, 64, sizeof(float) * K * samples) != 0)
	{
		fprintf(stderr, "posix_memalign for auxCentroids failed.\n");
		exit(-4);
	}

	// Set the number of OpenMP threads
	int nThreads = 4;
	if (argc == 8)
	{
		nThreads = atoi(argv[7]);
	}
	omp_set_num_threads(nThreads);

	printf("\n\tData file: %s \n\tPoints: %d\n\tDimensions: %d\n", argv[1], lines, samples);
	printf("\tNumber of clusters: %d\n", K);
	printf("\tMaximum number of iterations: %d\n", maxIterations);
	printf("\tMinimum number of changes: %d [%g%% of %d points]\n", minChanges, atof(argv[4]), lines);
	printf("\tMaximum centroid precision: %f\n", maxThreshold);

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
#pragma omp parallel default(none)                                         \
	firstprivate(lines, samples, K)                                        \
	shared(data, centroids, auxCentroids, classMap, pointsPerClass,        \
			   it, changes, maxDist, terminate, minChanges, maxIterations, \
			   maxThreshold)
	{
		while (1)
		{
			// Barrier so all threads see if 'terminate' was set at the end of the previous iteration.
#pragma omp barrier
			if (terminate)
			{
				break; // all threads exit the while loop
			}

			/* -------------------------------------------------------------------
			*  STEP 1: Assign points to nearest centroid
			*	Calculate the distance from each point to the centroid
			*	Assign each point to the nearest centroid.
			------------------------------------------------------------------- */
#pragma omp single
			{
				it++;		 // Increment iteration counter
				changes = 0; // counter for changes in cluster assignments
			}
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
			 *    STEP 2: Recalculate centroids (cluster means)
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
			*  Compute the maximum distance between old and new centroids
			 ------------------------------------------------------------------- */
#pragma omp single
			{
				maxDist = 0.0f;
			}

#pragma omp for reduction(max : maxDist) schedule(static)
			// For each centroid...
			for (int i = 0; i < K; i++)
			{
				if (pointsPerClass[i * PADDING] != 0)
				{
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
				// Copy new centroids into the real centroids array -- Update centroids
				memcpy(centroids, auxCentroids, sizeof(float) * K * samples);

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
	printf("\nComputation: %f seconds", end - start);
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
	error = writeResult(classMap, lines, argv[6], computationTime);
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