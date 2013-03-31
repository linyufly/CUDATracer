/******************************************************************
File			:		lcsBigBlockInitialization.cu
Author			:		Mingcheng Chen
Last Update		:		July 4th, 2012
*******************************************************************/

#include "device_launch_parameters.h"
#include "CUDAKernels.h"

__global__ void BigBlockInitialization(double *globalVertexPositions,
			     double *globalStartVelocities,
			     double *globalEndVelocities,
			
			     int *blockedGlobalPointIDs,

   			     int *startOffsetInPoint,

			     int *startOffsetInPointForBig,
			     double *vertexPositionsForBig,
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

	// Declare some arrays
	double *gVertexPositions;
	double *gStartVelocities;
	double *gEndVelocities;
		
	int startPoint = startOffsetInPoint[interestingBlockID];

	int numOfPoints = startOffsetInPoint[interestingBlockID + 1] - startPoint;

	int startPointForBig = startOffsetInPointForBig[interestingBlockID];

	// Initialize vertexPositions, startVelocities and endVelocities
	gVertexPositions = vertexPositionsForBig + startPointForBig * 3;
	gStartVelocities = startVelocitiesForBig + startPointForBig * 3;
	gEndVelocities = endVelocitiesForBig + startPointForBig * 3;

	for (int i = localID; i < numOfPoints * 3; i += numOfThreads) {
		int localPointID = i / 3;
		int dimensionID = i % 3;
		int globalPointID = blockedGlobalPointIDs[startPoint + localPointID];

		gVertexPositions[i] = globalVertexPositions[globalPointID * 3 + dimensionID];
		gStartVelocities[i] = globalStartVelocities[globalPointID * 3 + dimensionID];
		gEndVelocities[i] = globalEndVelocities[globalPointID * 3 + dimensionID];
	}
}	
