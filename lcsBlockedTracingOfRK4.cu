/*****************************************************
File		:	lcsBlockedTracingOfRK4.cu
Author		:	Mingcheng Chen
Last Update	:	January 31st, 2013
******************************************************/

#include <stdio.h>

__device__ inline double DeterminantThree(double *a) {
	// a[0] a[1] a[2]
	// a[3] a[4] a[5]
	// a[6] a[7] a[8]
	return a[0] * a[4] * a[8] + a[1] * a[5] * a[6] + a[2] * a[3] * a[7] -
	       a[0] * a[5] * a[7] - a[1] * a[3] * a[8] - a[2] * a[4] * a[6];
}

__device__ inline void CalculateNaturalCoordinates(double X, double Y, double Z,
					double *tetX, double *tetY, double *tetZ, double *coordinates) {
	X -= tetX[0];
	Y -= tetY[0];
	Z -= tetZ[0];

	double det[9] = {tetX[1] - tetX[0], tetY[1] - tetY[0], tetZ[1] - tetZ[0],
			 tetX[2] - tetX[0], tetY[2] - tetY[0], tetZ[2] - tetZ[0],
			 tetX[3] - tetX[0], tetY[3] - tetY[0], tetZ[3] - tetZ[0]};

	double V = 1.0 / DeterminantThree(det);

	double z41 = tetZ[3] - tetZ[0];
	double y34 = tetY[2] - tetY[3];
	double z34 = tetZ[2] - tetZ[3];
	double y41 = tetY[3] - tetY[0];
	double a11 = z41 * y34 - z34 * y41;

	double x41 = tetX[3] - tetX[0];
	double x34 = tetX[2] - tetX[3];
	double a12 = x41 * z34 - x34 * z41;

	double a13 = y41 * x34 - y34 * x41;

	coordinates[1] = (a11 * X + a12 * Y + a13 * Z) * V;

	double y12 = tetY[0] - tetY[1];
	double z12 = tetZ[0] - tetZ[1];
	double a21 = z41 * y12 - z12 * y41;

	double x12 = tetX[0] - tetX[1];
	double a22 = x41 * z12 - x12 * z41;

	double a23 = y41 * x12 - y12 * x41;

	coordinates[2] = (a21 * X + a22 * Y + a23 * Z) * V;

	double z23 = tetZ[1] - tetZ[2];
	double y23 = tetY[1] - tetY[2];
	double a31 = z23 * y12 - z12 * y23;

	double x23 = tetX[1] - tetX[2];
	double a32 = x23 * z12 - x12 * z23;

	double a33 = y23 * x12 - y12 * x23;

	coordinates[3] = (a31 * X + a32 * Y + a33 * Z) * V;

	coordinates[0] = 1.0 - coordinates[1] - coordinates[2] - coordinates[3];
}

__device__ inline int FindCell(double *particle, int *connectivities, int *links, double *vertexPositions,
			double epsilon, int guess, double *coordinates) {
	double tetX[4], tetY[4], tetZ[4];

	while (true) {
		for (int i = 0; i < 4; i++) {
			int pointID = connectivities[(guess << 2) | i];

			tetX[i] = vertexPositions[pointID * 3];
			tetY[i] = vertexPositions[pointID * 3 + 1];
			tetZ[i] = vertexPositions[pointID * 3 + 2];
		}

		CalculateNaturalCoordinates(particle[0], particle[1], particle[2], tetX, tetY, tetZ, coordinates);
		
		int index = 0;

		for (int i = 1; i < 4; i++)
			if (coordinates[i] < coordinates[index]) index = i;
		if (coordinates[index] >= -epsilon) break;

		guess = links[(guess << 2) | index];
		
		if (guess == -1) break;
	}

	return guess;
}

__global__ void BlockedTracingKernelOfRK4(double *globalVertexPositions,
					int *globalTetrahedralConnectivities,
					int *globalTetrahedralLinks,

					int *startOffsetInCell,
					int *startOffsetInPoint,

					double *vertexPositionsForBig,
					double *startVelocitiesForBig,
					double *endVelocitiesForBig,

					int *blockedLocalConnectivities,
					int *blockedLocalLinks,
					int *blockedGlobalCellIDs,
					int *blockedGlobalPointIDs,

					int *activeBlockList, // Map active block ID to interesting block ID

					int *blockOfGroups,
					int *offsetInBlocks,

					int *stage,
					double *lastPosition,
					double *k1,
					double *k2,
					double *k3,
					double *pastTimes,

					double *placesOfInterest,

					int *startOffsetInParticle,
					int *blockedActiveParticleIDList,
					int *cellLocations,

					int *exitCells,

					double startTime, double endTime, double timeStep, double epsilon,

					int sharedMemorySize) {
	__shared__ extern char sharedMemory[];

	// Get work group ID
	int groupID = blockIdx.x;
	
	// Get number of threads in a work group
	int numOfThreads = blockDim.x;

	// Get local thread ID
	int localID = threadIdx.x;

	// Get active block ID
	int activeBlockID = blockOfGroups[groupID];

	// Get interesting block ID of the work group
	int interestingBlockID = activeBlockList[activeBlockID];

	// Declare some arrays
	double *vertexPositions;
	double *startVelocities;
	double *endVelocities;
	int *connectivities;
	int *links;

	int startCell = startOffsetInCell[interestingBlockID];
	int startPoint = startOffsetInPoint[interestingBlockID];

	int numOfCells = startOffsetInCell[interestingBlockID + 1] - startCell;
	int numOfPoints = startOffsetInPoint[interestingBlockID + 1] - startPoint;

	// Assuming int is 4 bytes and double is 8 bytes
	//	 localNumOfCells * sizeof(int) * 4 +		// this->localConnectivities
	//	 localNumOfCells * sizeof(int) * 4 +		// this->localLinks
	//	 localNumOfPoints * sizeof(double) * 3 +	// point positions
	//	 localNumOfPoints * sizeof(double) * 3 * 2;	// point velocities (start and end)
	if (((numOfCells << 5) + ((numOfPoints * 9) << 3)) <= sharedMemorySize) { // This branch fills in the shared memory
		// Initialize vertexPositions, startVelocities and endVelocities
		vertexPositions = (double *)sharedMemory;
		startVelocities = vertexPositions + numOfPoints * 3;
		endVelocities = startVelocities + numOfPoints * 3;

		// Initialize connectivities and links
		connectivities = (int *)(endVelocities + numOfPoints * 3);
		links = connectivities + (numOfCells << 2);

		for (int i = localID; i < numOfPoints * 3; i += numOfThreads) {
			vertexPositions[i] = vertexPositionsForBig[startPoint * 3 + i];
			startVelocities[i] = startVelocitiesForBig[startPoint * 3 + i];
			endVelocities[i] = endVelocitiesForBig[startPoint * 3 + i];
		}

		for (int i = localID; i < (numOfCells << 2); i += numOfThreads) {
			connectivities[i] = blockedLocalConnectivities[(startCell << 2) + i];
			links[i] = blockedLocalLinks[(startCell << 2) + i];
		}
	} else { // This branch fills in the global memory
		// Initialize vertexPositions, startVelocities and endVelocities
		vertexPositions = vertexPositionsForBig + startPoint * 3;
		startVelocities = startVelocitiesForBig + startPoint * 3;
		endVelocities = endVelocitiesForBig + startPoint * 3;

		// Initialize connectivities and links
		connectivities = blockedLocalConnectivities + (startCell << 2);
		links = blockedLocalLinks + (startCell << 2);
	}

	__syncthreads();
	
	int numOfActiveParticles = startOffsetInParticle[activeBlockID + 1] - startOffsetInParticle[activeBlockID];

	int arrayIdx = offsetInBlocks[groupID] * numOfThreads + localID;

	if (arrayIdx < numOfActiveParticles) {
		// activeParticleID here means the initial active particle ID
		arrayIdx += startOffsetInParticle[activeBlockID];
		int activeParticleID = blockedActiveParticleIDList[arrayIdx];

		// Initialize the particle status
		int currStage = stage[activeParticleID];
		int currCell = cellLocations[activeParticleID];

		double currTime = pastTimes[activeParticleID];

		double currLastPosition[3];
		currLastPosition[0] = lastPosition[activeParticleID * 3];
		currLastPosition[1] = lastPosition[activeParticleID * 3 + 1];
		currLastPosition[2] = lastPosition[activeParticleID * 3 + 2];
		double currK1[3], currK2[3], currK3[3], currK4[3];
		if (currStage > 0) {
			currK1[0] = k1[activeParticleID * 3];
			currK1[1] = k1[activeParticleID * 3 + 1];
			currK1[2] = k1[activeParticleID * 3 + 2];
		}
		if (currStage > 1) {
			currK2[0] = k2[activeParticleID * 3];
			currK2[1] = k2[activeParticleID * 3 + 1];
			currK2[2] = k2[activeParticleID * 3 + 2];
		}
		if (currStage > 2) {
			currK3[0] = k3[activeParticleID * 3];
			currK3[1] = k3[activeParticleID * 3 + 1];
			currK3[2] = k3[activeParticleID * 3 + 2];
		}

		// At least one loop is executed.
		while (true) {
			double placeOfInterest[3];
			placeOfInterest[0] = currLastPosition[0];
			placeOfInterest[1] = currLastPosition[1];
			placeOfInterest[2] = currLastPosition[2];
			switch (currStage) {
			case 1: {
				placeOfInterest[0] += 0.5 * currK1[0];
				placeOfInterest[1] += 0.5 * currK1[1];
				placeOfInterest[2] += 0.5 * currK1[2];
					} break;
			case 2: {
				placeOfInterest[0] += 0.5 * currK2[0];
				placeOfInterest[1] += 0.5 * currK2[1];
				placeOfInterest[2] += 0.5 * currK2[2];
					} break;
			case 3: {
				placeOfInterest[0] += currK3[0];
				placeOfInterest[1] += currK3[1];
				placeOfInterest[2] += currK3[2];
					} break;
			}

			double coordinates[4];

			int nextCell = FindCell(placeOfInterest, connectivities, links, vertexPositions, epsilon, currCell, coordinates);

			if (nextCell == -1 || currTime >= endTime) {
				// Find the next cell globally
				int globalCellID = blockedGlobalCellIDs[startCell + currCell];
				int nextGlobalCell;
				
				if (nextCell != -1)
					nextGlobalCell = blockedGlobalCellIDs[startCell + nextCell];
				else
					nextGlobalCell = FindCell(placeOfInterest, globalTetrahedralConnectivities,
								globalTetrahedralLinks, globalVertexPositions,
								epsilon, globalCellID, coordinates);

				if (currTime >= endTime && nextGlobalCell != -1) nextGlobalCell = -2 - nextGlobalCell;

				pastTimes[activeParticleID] = currTime;

				stage[activeParticleID] = currStage;

				lastPosition[activeParticleID * 3] = currLastPosition[0];
				lastPosition[activeParticleID * 3 + 1] = currLastPosition[1];
				lastPosition[activeParticleID * 3 + 2] = currLastPosition[2];

				placesOfInterest[activeParticleID * 3] = placeOfInterest[0];
				placesOfInterest[activeParticleID * 3 + 1] = placeOfInterest[1];
				placesOfInterest[activeParticleID * 3 + 2] = placeOfInterest[2];

				exitCells[activeParticleID] = nextGlobalCell;
		
				if (currStage > 0) {
					k1[activeParticleID * 3] = currK1[0];
					k1[activeParticleID * 3 + 1] = currK1[1];
					k1[activeParticleID * 3 + 2] = currK1[2];
				}
				if (currStage > 1) {
					k2[activeParticleID * 3] = currK2[0];
					k2[activeParticleID * 3 + 1] = currK2[1];
					k2[activeParticleID * 3 + 2] = currK2[2];
				}
				if (currStage > 2) {
					k3[activeParticleID * 3] = currK3[0];
					k3[activeParticleID * 3 + 1] = currK3[1];
					k3[activeParticleID * 3 + 2] = currK3[2];
				}
				break;
			}

			currCell = nextCell;

			double exactTime = currTime;
			switch (currStage) {
			case 0: break;
			case 1:
			case 2: exactTime += timeStep * 0.5; break;
			case 3: exactTime += timeStep; break;
			}

			double alpha = (endTime - exactTime) / (endTime - startTime);
			double beta = 1 - alpha;

			double vecX[4], vecY[4], vecZ[4];

			for (int i = 0; i < 4; i++) {
				int pointID = connectivities[(nextCell << 2) | i];
				vecX[i] = startVelocities[pointID * 3] * alpha + endVelocities[pointID * 3] * beta;
				vecY[i] = startVelocities[pointID * 3 + 1] * alpha + endVelocities[pointID * 3 + 1] * beta;
				vecZ[i] = startVelocities[pointID * 3 + 2] * alpha + endVelocities[pointID * 3 + 2] * beta;
			}

			double *currK;
			switch (currStage) {
			case 0: currK = currK1; break;
			case 1: currK = currK2; break;
			case 2: currK = currK3; break;
			case 3: currK = currK4; break;
			}

			currK[0] = currK[1] = currK[2] = 0;

			for (int i = 0; i < 4; i++) {
				currK[0] += vecX[i] * coordinates[i];
				currK[1] += vecY[i] * coordinates[i];
				currK[2] += vecZ[i] * coordinates[i];
			}

			currK[0] *= timeStep;
			currK[1] *= timeStep;
			currK[2] *= timeStep;

			if (currStage == 3) {
				currTime += timeStep;

				for (int i = 0; i < 3; i++)
					currLastPosition[i] += (currK1[i] + 2 * currK2[i] + 2 * currK3[i] + currK4[i]) / 6;

				currStage = 0;
			} else
				currStage++;
		}
	}
}

extern "C"
void BlockedTracingOfRK4(double *globalVertexPositions,
			int *globalTetrahedralConnectivities,
			int *globalTetrahedralLinks,

			int *startOffsetInCell,
			int *startOffsetInPoint,

			double *vertexPositionsForBig,
			double *startVelocitiesForBig,
			double *endVelocitiesForBig,

			int *blockedLocalConnectivities,
			int *blockedLocalLinks,
			int *blockedGlobalCellIDs,
			int *blockedGlobalPointIDs,

			int *activeBlockList, // Map active block ID to interesting block ID

			int *blockOfGroups,
			int *offsetInBlocks,

			int *stage,
			double *lastPosition,
			double *k1,
			double *k2,
			double *k3,
			double *pastTimes,

			double *placesOfInterest,

			int *startOffsetInParticle,
			int *blockedActiveParticleIDList,
			int *cellLocations,

			int *exitCells,

			double startTime, double endTime, double timeStep, double epsilon, int numOfActiveBlocks,

			int blockSize, int sharedMemorySize) {
	dim3 dimBlock(blockSize, 1, 1);
	dim3 dimGrid(numOfActiveBlocks, 1, 1);

	BlockedTracingKernelOfRK4<<<dimGrid, dimBlock, sharedMemorySize>>>(globalVertexPositions,
					globalTetrahedralConnectivities,
					globalTetrahedralLinks,

					startOffsetInCell,
					startOffsetInPoint,

					vertexPositionsForBig,
					startVelocitiesForBig,
					endVelocitiesForBig,

					blockedLocalConnectivities,
					blockedLocalLinks,
					blockedGlobalCellIDs,
					blockedGlobalPointIDs,

					activeBlockList, // Map active block ID to interesting block ID

					blockOfGroups,
					offsetInBlocks,

					stage,
					lastPosition,
					k1,
					k2,
					k3,
					pastTimes,

					placesOfInterest,

					startOffsetInParticle,
					blockedActiveParticleIDList,
					cellLocations,

					exitCells,

					startTime, endTime, timeStep, epsilon,

					sharedMemorySize);

	cudaError_t err = cudaDeviceSynchronize();

	if (err) {
		cudaGetErrorString(err);
		exit(0);
	}
}
