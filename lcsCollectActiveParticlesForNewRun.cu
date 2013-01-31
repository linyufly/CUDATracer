/************************************************************************
File		:	lcsCollectActiveParticlesForNewRun.cu
Author		:	Mingcheng Chen
Last Update	:	January 29th, 2013
*************************************************************************/

#include <stdio.h>

#define BLOCK_SIZE 1024

__global__ void InitializeScanArrayKernel(int *exitCells, int *oldActiveParticles, int *scanArray, int length) {
	int globalID = blockDim.x * blockIdx.x + threadIdx.x;
	if (globalID < length)
		scanArray[globalID] = exitCells[oldActiveParticles[globalID]] < 0 ? 0 : 1;
}

__global__ void CollectActiveParticlesKernel(int *exitCells, int *oldActiveParticles, int *scanArray,
				    int *newActiveParticles, int length) {
	int globalID = blockDim.x * blockIdx.x + threadIdx.x;
	if (globalID < length)
		if (exitCells[oldActiveParticles[globalID]] >= 0)
			newActiveParticles[scanArray[globalID]] = oldActiveParticles[globalID];
}

extern "C"
void InitializeScanArray2(int *exitCells, int *oldActiveParticles, int *scanArray, int length) {
	dim3 dimBlock(BLOCK_SIZE, 1, 1);
	dim3 dimGrid((length - 1) / dimBlock.x + 1, 1, 1);

	InitializeScanArrayKernel<<<dimGrid, dimBlock>>>(exitCells, oldActiveParticles, scanArray, length);

	cudaError_t err = cudaDeviceSynchronize();
	if (err) {
		cudaGetErrorString(err);
		exit(0);
	}
}

extern "C"
void CollectActiveParticles2(int *exitCells, int *oldActiveParticles, int *scanArray,
				    int *newActiveParticles, int length) {
	dim3 dimBlock(BLOCK_SIZE, 1, 1);
	dim3 dimGrid((length - 1) / dimBlock.x + 1, 1, 1);

	CollectActiveParticlesKernel<<<dimGrid, dimBlock>>>(exitCells, oldActiveParticles, scanArray,
							newActiveParticles, length);

	cudaError_t err = cudaDeviceSynchronize();
	if (err) {
		cudaGetErrorString(err);
		exit(0);
	}
}

