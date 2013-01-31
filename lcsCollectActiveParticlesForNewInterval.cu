/*************************************************************************
File		:	lcsCollectActiveParticlesForNewInterval.cu
Author		:	Mingcheng Chen
Last Update	:	September 3rd, 2012
**************************************************************************/

#include <stdio.h>

#define BLOCK_SIZE 1024

__global__ void InitializeScanArrayKernel(int *exitCells, int *scanArray, int length) {
	int globalID = blockDim.x * blockIdx.x + threadIdx.x;
	if (globalID < length) {
		if (exitCells[globalID] < -1) exitCells[globalID] = -(exitCells[globalID] + 2);
		scanArray[globalID] = exitCells[globalID] == -1 ? 0 : 1;
	}
}

__global__ void CollectActiveParticlesKernel(int *exitCells, int *scanArray, int *activeParticles, int length) {
	int globalID = blockDim.x * blockIdx.x + threadIdx.x;
	if (globalID < length) {
		if (exitCells[globalID] != -1)
			activeParticles[scanArray[globalID]] = globalID;
	}
}

extern "C"
void InitializeScanArray(int *exitCells, int *scanArray, int length) {
	dim3 dimBlock(BLOCK_SIZE, 1, 1);
	dim3 dimGrid((length - 1) / dimBlock.x + 1, 1, 1);

	InitializeScanArrayKernel<<<dimGrid, dimBlock>>>(exitCells, scanArray, length);

	cudaError_t err = cudaDeviceSynchronize();
	if (err) {
		cudaGetErrorString(err);
		exit(0);
	}
}

extern "C"
void CollectActiveParticles(int *exitCells, int *scanArray, int *activeParticles, int length) {
	dim3 dimBlock(BLOCK_SIZE, 1, 1);
	dim3 dimGrid((length - 1) / dimBlock.x + 1, 1, 1);

	CollectActiveParticlesKernel<<<dimGrid, dimBlock>>>(exitCells, scanArray, activeParticles, length);

	cudaError_t err = cudaDeviceSynchronize();
	if (err) {
		cudaGetErrorString(err);
		exit(0);
	}
}
