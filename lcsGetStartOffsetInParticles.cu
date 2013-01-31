/**************************************************************
File		:	lcsGetStartOffsetInParticles.cu
Author		:	Mingcheng Chen
Last Update	:	January 29th, 2013
***************************************************************/

#include <stdio.h>

#define BLOCK_SIZE 1024

__global__ void CollectEveryKElementKernel(int* input, int *output, int k, int length) {
	int globalID = blockDim.x * blockIdx.x + threadIdx.x;
	if (globalID < length)
		output[globalID] = input[globalID * k];
}

extern "C"
void CollectEveryKElement(int *input, int *output, int k, int length) {
	dim3 dimBlock(BLOCK_SIZE, 1, 1);
	dim3 dimGrid((length - 1) / dimBlock.x + 1, 1, 1);

	CollectEveryKElementKernel<<<dimGrid, dimBlock>>>(input, output, k, length);

	cudaError_t err = cudaDeviceSynchronize();
	if (err) {
		cudaGetErrorString(err);
		exit(0);
	}
}
