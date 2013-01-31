/******************************************************************
File		:	lcsBigBlockInitializationForPositions.cu
Author		:	Mingcheng Chen
Last Update	:	January 31st, 2013
*******************************************************************/

#include <stdio.h>

#define BLOCK_SIZE 512

__global__ void BigBlockInitializationForPositionsKernel(double *globalVertexPositions,
						 int *blockedGlobalPointIDs,
   						 int *startOffsetInPoint,
						 double *vertexPositionsForBig
			     			 ) {
	// Get number of threads in a work group
	int numOfThreads = blockDim.x;

	// Get local thread ID
	int localID = threadIdx.x;

	// Get interesting block ID of the current big block
	int interestingBlockID = blockIdx.x;

	// Declare some work arrays
	double *gVertexPositions;
		
	int startPoint = startOffsetInPoint[interestingBlockID];
	int numOfPoints = startOffsetInPoint[interestingBlockID + 1] - startPoint;

	// Initialize vertexPositions
	gVertexPositions = vertexPositionsForBig + startPoint * 3;

	for (int i = localID; i < numOfPoints * 3; i += numOfThreads) {
		int localPointID = i / 3;
		int dimensionID = i % 3;
		int globalPointID = blockedGlobalPointIDs[startPoint + localPointID];

		gVertexPositions[i] = globalVertexPositions[globalPointID * 3 + dimensionID];
	}
}

extern "C"
void BigBlockInitializationForPositions(double *globalVertexPositions,	
					int *blockedGlobalPointIDs,
   					int *startOffsetInPoint,
					double *vertexPositionsForBig,
					int numOfInterestingBlocks
			     		) {
	dim3 dimBlock(BLOCK_SIZE, 1, 1);
	dim3 dimGrid(numOfInterestingBlocks, 1, 1);

	BigBlockInitializationForPositionsKernel<<<dimGrid, dimBlock>>>(globalVertexPositions, blockedGlobalPointIDs,
									startOffsetInPoint, vertexPositionsForBig);

	cudaError_t err = cudaDeviceSynchronize();
	if (err) {
		cudaGetErrorString(err);
		exit(0);
	}
}
