/******************************************************************
File		:	lcsBigBlockInitializationForPositions.cu
Author		:	Mingcheng Chen
Last Update	:	January 30th, 2013
*******************************************************************/

#include <stdio.h>

#define BLOCK_SIZE 512

__global__ void BigBlockInitializationForPositionsKernel(double *globalVertexPositions,
			
						 int *blockedGlobalPointIDs,

   						 int *startOffsetInPoint,

						 int *startOffsetInPointForBig,
						 double *vertexPositionsForBig,

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
	double *gVertexPositions;
		
	int startPoint = startOffsetInPoint[interestingBlockID];

	int numOfPoints = startOffsetInPoint[interestingBlockID + 1] - startPoint;

	int startPointForBig = startOffsetInPointForBig[interestingBlockID];

	// Initialize vertexPositions
	gVertexPositions = vertexPositionsForBig + startPointForBig * 3;

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

						 int *startOffsetInPointForBig,
						 double *vertexPositionsForBig,

						 int *bigBlocks, int numOfBigBlocks
			     			 ) {
	dim3 dimBlock(BLOCK_SIZE, 1, 1);
	dim3 dimGrid(numOfBigBlocks, 1, 1);

	BigBlockInitializationForPositionsKernel<<<dimGrid, dimBlock>>>(globalVertexPositions, blockedGlobalPointIDs, startOffsetInPoint,
									startOffsetInPointForBig, vertexPositionsForBig, bigBlocks);

	cudaError_t err = cudaDeviceSynchronize();
	if (err) {
		cudaGetErrorString(err);
		exit(0);
	}
}
