/******************************************************************
File		:	lcsBigBlockInitializationForVelocities.cu
Author		:	Mingcheng Chen
Last Update	:	January 30th, 2013
*******************************************************************/

#include <stdio.h>

#define BLOCK_SIZE 512

__global__ void BigBlockInitializationForVelocitiesKernel(double *globalStartVelocities,
			     			  double *globalEndVelocities,
			
			     			  int *blockedGlobalPointIDs,

   			     			  int *startOffsetInPoint,

			     			  int *startOffsetInPointForBig,
			     			  double *startVelocitiesForBig,
			     			  double *endVelocitiesForBig,

			     			  int *bigBlocks
			     			  ) {
	// Get work group ID
	int workGroupID = blockIdx.x;
	
	// Get number of threads in a work group
	int numOfThreads = blockDim.x;

	// Get local thread ID
	int localID = threadIdx.x;

	// Get interesting block ID of the current big block
	int interestingBlockID = bigBlocks[workGroupID];

	// Declare some work arrays
	double *gStartVelocities;
	double *gEndVelocities;
		
	int startPoint = startOffsetInPoint[interestingBlockID];

	int numOfPoints = startOffsetInPoint[interestingBlockID + 1] - startPoint;

	int startPointForBig = startOffsetInPointForBig[interestingBlockID];

	// Initialize startVelocities and endVelocities
	gStartVelocities = startVelocitiesForBig + startPointForBig * 3;
	gEndVelocities = endVelocitiesForBig + startPointForBig * 3;

	for (int i = localID; i < numOfPoints * 3; i += numOfThreads) {
		int localPointID = i / 3;
		int dimensionID = i % 3;
		int globalPointID = blockedGlobalPointIDs[startPoint + localPointID];

		gStartVelocities[i] = globalStartVelocities[globalPointID * 3 + dimensionID];
		gEndVelocities[i] = globalEndVelocities[globalPointID * 3 + dimensionID];
	}
}

extern "C"
void BigBlockInitializationForVelocities(double *globalStartVelocities,
			     			  double *globalEndVelocities,
			
			     			  int *blockedGlobalPointIDs,

   			     			  int *startOffsetInPoint,

			     			  int *startOffsetInPointForBig,
			     			  double *startVelocitiesForBig,
			     			  double *endVelocitiesForBig,

			     			  int *bigBlocks, int numOfBigBlocks
			     			  ) {
	dim3 dimBlock(BLOCK_SIZE, 1, 1);
	dim3 dimGrid(numOfBigBlocks, 1, 1);

	BigBlockInitializationForVelocitiesKernel<<<dimGrid, dimBlock>>>(globalStartVelocities, globalEndVelocities, blockedGlobalPointIDs,
									startOffsetInPoint, startOffsetInPointForBig, startVelocitiesForBig,
									endVelocitiesForBig, bigBlocks);

	cudaError_t err = cudaDeviceSynchronize();
	if (err) {
		cudaGetErrorString(err);
		exit(0);
	}
}
