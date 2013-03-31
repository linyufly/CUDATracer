#include <cstdio>
#include <ctime>
#include "lcs.h"
#include "cuda_runtime.h"
#include "CUDAHostFunctions.h"

#include "CUDAKernels.h"

void LaunchGPUForInitialCellLocation(double minX, double maxX, double minY, double maxY, double minZ, double maxZ,
									 int xRes, int yRes, int zRes,
									 int *&initialCellLocations,
									 int *&gridCounts,
									 int *&d_cellLocations,
									 int *&d_gridCounts,
									 int globalNumOfCells,
									 double *d_vertexPositions,
									 int *d_tetrahedralConnectivities,
									 double epsilon) {
	/// DEBUG ///
	//printf("**********************************************epsilon = %lf\n", epsilon);

	cudaError_t err;

	double dx = (maxX - minX) / xRes;
	double dy = (maxY - minY) / yRes;
	double dz = (maxZ - minZ) / zRes;

	int numOfGridPoints = (xRes + 1) * (yRes + 1) * (zRes + 1);
	initialCellLocations = new int [numOfGridPoints];
	//memset(initialCellLocations, 255, sizeof(int) * numOfGridPoints);
	for (int i = 0; i < numOfGridPoints; i++)
		initialCellLocations[i] = -1;

	gridCounts = new int [numOfGridPoints];
	memset(gridCounts, 0, sizeof(int) * numOfGridPoints);

	// Create CUDA C buffer pointing to the device cellLocations (output)
	err = cudaMalloc((void **)&d_cellLocations, sizeof(int) * numOfGridPoints);
	if (err) lcs::Error("Fail to create a buffer for device cellLocations");

	err = cudaMemset(d_cellLocations, 255, sizeof(int) * numOfGridPoints);
	if (err) lcs::Error("Fail to initialize d_cellLocations");

	/// DEBUG ///
	err = cudaMalloc((void **)&d_gridCounts, sizeof(int) * numOfGridPoints);
	if (err) lcs::Error("Fail to create a buffer for device gridCounts");

	err = cudaMemset(d_gridCounts, 0, sizeof(int) * numOfGridPoints);
	if (err) lcs::Error("Fail to initialize d_gridCounts");

	int threadBlockSize = BLOCK_SIZE;
	dim3 dimGrid;
	dimGrid.x = (globalNumOfCells - 1) / threadBlockSize + 1;
	dimGrid.y = dimGrid.z = 1;
	dim3 dimBlock(threadBlockSize, 1, 1);

	InitialCellLocation<<<dimGrid, dimBlock>>>(d_vertexPositions,
											   d_tetrahedralConnectivities,
											   d_cellLocations,
											   xRes, yRes, zRes,
											   minX, minY, minZ,
											   dx, dy, dz,
											   epsilon,
											   globalNumOfCells,

											   d_gridCounts);

	err = cudaDeviceSynchronize();
	if (err) lcs::Error("Fail to launch the initial location kernel");

	err = cudaMemcpy(initialCellLocations, d_cellLocations, sizeof(int) * numOfGridPoints, cudaMemcpyDeviceToHost);
	if (err) lcs::Error("Fail to read device initialCellLocations");

	err = cudaMemcpy(gridCounts, d_gridCounts, sizeof(int) * numOfGridPoints, cudaMemcpyDeviceToHost);
	if (err) lcs::Error("Fail to read device gridCounts");

	cudaFree(d_cellLocations);
	cudaFree(d_gridCounts);
}

void LaunchGPUForNaiveTracing(double *globalVertexPositions,
								 double *globalStartVelocities,
								 double *globalEndVelocities,
								 int *globalTetrahedralConnectivities,
								 int *globalTetrahedralLinks,

								 int *stage,
								 double *lastPosition,
								 double *k1,
								 double *k2,
								 double *k3,
								 double *pastTimes,

								 int *cellLocations,

								 double startTime, double endTime, double timeStep,
								 double epsilon,

								 int *activeParticles,
								 int numOfActiveParticles
								 ) {
	int threadBlockSize = BLOCK_SIZE;
	dim3 dimGrid;
	dimGrid.x = (numOfActiveParticles - 1) / threadBlockSize + 1;
	dimGrid.y = dimGrid.z = 1;
	dim3 dimBlock(threadBlockSize, 1, 1);

	cudaError_t err;

	cudaFuncSetCacheConfig(NaiveTracing, cudaFuncCachePreferShared);

	int tt = clock();

	NaiveTracing<<<dimGrid, dimBlock>>>(globalVertexPositions,
		                                globalStartVelocities,
										globalEndVelocities,
									    globalTetrahedralConnectivities,
										globalTetrahedralLinks,

										//stage,
										lastPosition,
										//k1, k2, k3,
										pastTimes,

										cellLocations,

										startTime, endTime, timeStep, epsilon,

										activeParticles,
										numOfActiveParticles);

	printf("time : %lf\n", (double)(clock() - tt) / CLOCKS_PER_SEC);

	err = cudaDeviceSynchronize();

	printf("err string = %s\n", cudaGetErrorString(err));
	if (err) lcs::Error("Fail to launch the naive tracing kernel");
}
