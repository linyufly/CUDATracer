/*********************************************************
File		:	lcsGetGroupsForBlocks.cu
Author		:	Mingcheng Chen
Last Update	:	January 29th, 2013
**********************************************************/

#include <stdio.h>

#define BLOCK_SIZE 1024

__global__ void GetNumOfGroupsForBlocksKernel(int *startOffsetInParticles,
				      int *numOfGroupsForBlocks,
				      int numOfActiveBlocks, int groupSize) {
	int globalID = blockDim.x * blockIdx.x + threadIdx.x;
	if (globalID < numOfActiveBlocks) {
		int numOfParticles = startOffsetInParticles[globalID + 1] - startOffsetInParticles[globalID];
		numOfGroupsForBlocks[globalID] = (numOfParticles - 1) / groupSize + 1;
	}
}

__global__ void AssignGroupsKernel(int *numOfGroupsForBlocks, // It should be the prefix sum now.
			   int *blockOfGroups,
			   int *offsetInBlocks) {
	int blockID = blockIdx.x;
	int threadID = threadIdx.x;
	int workSize = blockDim.x;
	int startOffset = numOfGroupsForBlocks[blockID];
	int numOfGroups = numOfGroupsForBlocks[blockID + 1] - startOffset;
	//int numOfGroupsPerThread = (numOfGroups - 1) / workSize + 1;

	for (int i = threadID; i < numOfGroups; i += workSize) {
		blockOfGroups[startOffset + i] = blockID;
		offsetInBlocks[startOffset + i] = i;
	}
}

extern "C"
void GetNumOfGroupsForBlocks(int *startOffsetInParticles, int *numOfGroupsForBlocks, int numOfActiveBlocks, int groupSize) {
	dim3 dimBlock(BLOCK_SIZE, 1, 1);
	dim3 dimGrid((numOfActiveBlocks - 1) / dimBlock.x + 1, 1, 1);

	GetNumOfGroupsForBlocksKernel<<<dimGrid, dimBlock>>>(startOffsetInParticles, numOfGroupsForBlocks, numOfActiveBlocks, groupSize);

	cudaError_t err = cudaDeviceSynchronize();
	if (err) {
		cudaGetErrorString(err);
		exit(0);
	}
}

extern "C"
void AssignGroups(int *numOfGroupsForBlocks, // It should be the prefix sum now.
		int *blockOfGroups,
		int *offsetInBlocks,
		int numOfActiveBlocks) {
	dim3 dimBlock(BLOCK_SIZE, 1, 1);
	dim3 dimGrid(numOfActiveBlocks, 1, 1);

	AssignGroupsKernel<<<dimGrid, dimBlock>>>(numOfGroupsForBlocks, // It should be the prefix sum now.
						blockOfGroups, offsetInBlocks);

	cudaError_t err = cudaDeviceSynchronize();
	if (err) {
		cudaGetErrorString(err);
		exit(0);
	}

}
