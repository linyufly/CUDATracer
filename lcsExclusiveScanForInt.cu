/******************************************************************
File			:		lcsExclusiveScanForInt.cu
Author			:		Mingcheng Chen
Last Update		:		January 29th, 2013
*******************************************************************/

#include <stdio.h>

#define BLOCK_SIZE 512

#define NUM_BANKS 32
#define LOG_NUM_BANKS 5

#define CONFLICT_FREE_OFFSET(n) ((n) >> (LOG_NUM_BANKS))
#define POSI(n) ((n) + CONFLICT_FREE_OFFSET(n))

__global__ void ScanKernel(int *globalArray, int length, int step) {
	__shared__ int localArray[POSI(BLOCK_SIZE << 1) + 1];

	int localID = threadIdx.x;
	int groupID = blockIdx.x;
	int groupSize = blockDim.x;
	int startOffset = (groupSize << 1) * groupID * step;
	
	int posi1 = startOffset + localID * step;
	int posi2 = posi1 + groupSize * step;
	
	localArray[POSI(localID)] = posi1 < length ? globalArray[posi1] : 0;
	localArray[POSI(localID + groupSize)] = posi2 < length ? globalArray[posi2] : 0;
	
	// Up-sweep
	for (int stride = 1, d = groupSize; stride <= groupSize; stride <<= 1, d >>= 1) {
		__syncthreads();
		
		if (localID < d) {
			posi1 = stride * ((localID << 1) + 1) - 1;
			posi2 = posi1 + stride;
			localArray[POSI(posi2)] += localArray[POSI(posi1)];
		}
	}
	
	// Down-sweep
	for (int stride = groupSize, d = 1; stride >= 1; stride >>= 1, d <<= 1) {
		__syncthreads();
		
		if (localID < d) {
			posi1 = stride * ((localID << 1) + 1) - 1;
			posi2 = POSI(posi1 + stride);
			posi1 = POSI(posi1);
			
			int t = localArray[posi1];
			localArray[posi1] = localArray[posi2];
			localArray[posi2] = localArray[posi2] * !!localID + t;
		}
	}
	
	__syncthreads();
	
	// Write to global memory
	posi1 = startOffset + localID * step;
	posi2 = posi1 + groupSize * step;
	
	if (posi1 < length) globalArray[posi1] = localArray[POSI(localID)];
	if (posi2 < length) globalArray[posi2] = localArray[POSI(localID + groupSize)];
}

__global__ void ReverseUpdateKernel(int *globalArray, int length, int step) {
	int localID = threadIdx.x;
	int groupID = blockIdx.x;
	int groupSize = blockDim.x;
	int startOffset = groupID * (groupSize << 1) * step;
	
	if (groupID) {
		int value = globalArray[startOffset];
		int posi1 = startOffset + localID * step;
		int posi2 = posi1 + groupSize * step;
		if (posi1 < length && localID) globalArray[posi1] += value;
		if (posi2 < length) globalArray[posi2] += value;
	}
}

extern "C"
int ExclusiveScanForInt(int *d_arr, int length) {
	cudaError_t err;

	// Get the work group size
	int localWorkSize = BLOCK_SIZE;

	// Up-sweep and down-sweep	
	static int records[10];
	
	int problemSize = length;
	int numOfRecords = 0;

	int d_step = 1;

	/// DEBUG ///
	//printf("length = %d\n", length);

	dim3 dimBlock(BLOCK_SIZE, 1, 1);
	dim3 dimGrid(1, 1, 1);

	for (; problemSize > 1; problemSize = (problemSize - 1) / (localWorkSize * 2) + 1) {
		if (numOfRecords) d_step *= localWorkSize * 2;
		records[numOfRecords++] = problemSize;

		dimGrid.x = (problemSize - 1) / (localWorkSize * 2) + 1;
	
		ScanKernel<<<dimGrid, dimBlock>>>(d_arr, length, d_step);

		err = cudaDeviceSynchronize();
		if (err) {
			printf("Fail to finish scan kernel");
			cudaGetErrorString(err);
			exit(0);
		}
	}

	int sum;
	err = cudaMemcpy(&sum, d_arr, sizeof(int), cudaMemcpyDeviceToHost);
	if (err) {
		cudaGetErrorString(err);
		exit(0);
	}

	err = cudaMemset(d_arr, 0, sizeof(int));
	if (err) {
		cudaGetErrorString(err);
		exit(0);
	}

	// Reverse updates
	for (int i = numOfRecords - 1; i >= 0; i--, d_step /= localWorkSize * 2) {
		dimGrid.x = (records[i] - 1) / (localWorkSize * 2) + 1;

		ReverseUpdateKernel<<<dimGrid, dimBlock>>>(d_arr, length, d_step);

		err = cudaDeviceSynchronize();
		if (err) {
			printf("Fail to finish reverse update kernel");
			cudaGetErrorString(err);
			exit(0);
		}
	}

	return sum;
}
