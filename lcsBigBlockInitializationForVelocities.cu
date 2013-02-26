/******************************************************************
File		:	lcsBigBlockInitializationForVelocities.cu
Author		:	Mingcheng Chen
Last Update	:	February 22nd, 2013
*******************************************************************/

#include <stdio.h>

#define BLOCK_SIZE 512

__global__ void BigBlockInitializationForVelocitiesKernel(double *globalVelocities,
			     			  int *blockedGlobalPointIDs,
   			     			  int *startOffsetInPoint,
			     			  double *velocitiesForBig,
						  int startArrayID, int numOfTimePoints, int maxNumOfTimePoints,
						  int globalNumOfPoints
			     			  ) {
	// Get number of threads in a work group
	int numOfThreads = blockDim.x;

	// Get local thread ID
	int localID = threadIdx.x;

	// Get interesting block ID of the current big block
	int interestingBlockID = blockIdx.x;

	// Declare some work arrays
	double *localVelocities;
		
	int startPoint = startOffsetInPoint[interestingBlockID];
	int numOfPoints = startOffsetInPoint[interestingBlockID + 1] - startPoint;

	// Initialize localVelocities;
	localVelocities = velocitiesForBig + startPoint * 3 * numOfTimePoints;

	for (int i = localID; i < numOfPoints * 3 * numOfTimePoints; i += numOfThreads) {
		int localPointID = i / (3 * numOfTimePoints);
		int timePoint = i % (3 * numOfTimePoints) / 3;
		int dimensionID = i % 3;

		int globalPointID = blockedGlobalPointIDs[startPoint + localPointID];

		localVelocities[i] = globalVelocities[((startArrayID + timePoint) % maxNumOfTimePoints) * 3 * globalNumOfPoints + globalPointID * 3 + dimensionID];
	}
}

extern "C"
void BigBlockInitializationForVelocities(double *globalVelocities,
					int *blockedGlobalPointIDs,
					int *startOffsetInPoint,
					double *velocitiesForBig,
					int startArrayID, int numOfTimePoints, int maxNumOfTimePoints,
					int globalNumOfPoints,
					int numOfInterestingBlocks
			     		) {
	dim3 dimBlock(BLOCK_SIZE, 1, 1);
	dim3 dimGrid(numOfInterestingBlocks, 1, 1);

	BigBlockInitializationForVelocitiesKernel<<<dimGrid, dimBlock>>>(globalVelocities, blockedGlobalPointIDs,
									startOffsetInPoint, velocitiesForBig, startArrayID, numOfTimePoints, maxNumOfTimePoints,
									globalNumOfPoints);

	cudaError_t err = cudaDeviceSynchronize();
	if (err) {
		cudaGetErrorString(err);
		exit(0);
	}
}
