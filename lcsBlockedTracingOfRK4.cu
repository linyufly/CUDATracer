/*****************************************************
File		:	lcsBlockedTracingOfRK4.cu
Author		:	Mingcheng Chen
Last Update	:	February 19th, 2013
******************************************************/

#include <stdio.h>

/*
__device__ inline double DeterminantThree(double *a) {
	// a[0] a[1] a[2]
	// a[3] a[4] a[5]
	// a[6] a[7] a[8]
	//return a[0] * a[4] * a[8] + a[1] * a[5] * a[6] + a[2] * a[3] * a[7] -
	//       a[0] * a[5] * a[7] - a[1] * a[3] * a[8] - a[2] * a[4] * a[6];
	return a[0] * (a[4] * a[8] - a[5] * a[7]) + a[1] * (a[5] * a[6] - a[3] * a[8]) + a[2] * (a[3] * a[7] - a[4] * a[6]);
}
*/

//__constant__ void *pointers[25];
__constant__ double *globalVertexPositions;
__constant__ int *globalTetrahedralConnectivities, *globalTetrahedralLinks, *startOffsetInCell, *startOffsetInPoint;
__constant__ double *vertexPositionsForBig, *startVelocitiesForBig, *endVelocitiesForBig;
__constant__ int *blockedLocalConnectivities, *blockedLocalLinks, *blockedGlobalCellIDs, *activeBlockList, // Map active block ID to interesting block ID
		 *blockOfGroups, *offsetInBlocks, *stage;
__constant__ double *lastPosition, *k1, *k2, *k3, *pastTimes, *placesOfInterest;
__constant__ int *startOffsetInParticle, *blockedActiveParticleIDList, *cellLocations, *exitCells;

__constant__ double timeStep, epsilon;
//__constant__ double doubleValues[4];

__device__ inline int FindCell(double *particle, int *connectivities, int *links, double *vertexPositions,
			double epsilon, int guess, double *coordinates, double *tetX, double *tetY, double *tetZ) {
	//double tetX[4], tetY[4], tetZ[4];
	int index, pointID;
	double X, Y, Z, V;
	double z41, y34, z34, y41, a11, x41, x34, a12, a13, y12, z12, a21, x12, a22, a23, z23, y23, a31, x23, a32, a33;

	while (true) {
		for (index = 0; index < 4; index++) {
			pointID = connectivities[(guess << 2) | index] * 3;

			tetX[index] = vertexPositions[pointID /** 3*/];
			tetY[index] = vertexPositions[pointID /** 3*/ + 1];
			tetZ[index] = vertexPositions[pointID /** 3*/ + 2];
		}

		//CalculateNaturalCoordinates(particle[0], particle[1], particle[2], tetX, tetY, tetZ, coordinates);

		X = particle[0] - tetX[0];
		Y = particle[1] - tetY[0];
		Z = particle[2] - tetZ[0];

		tetX[1] -= tetX[0];
		tetX[2] -= tetX[0];
		tetX[3] -= tetX[0];

		tetY[1] -= tetY[0];
		tetY[2] -= tetY[0];
		tetY[3] -= tetY[0];

		tetZ[1] -= tetZ[0];
		tetZ[2] -= tetZ[0];
		tetZ[3] -= tetZ[0];

		//double a[9] = {tetX[1] - tetX[0], tetY[1] - tetY[0], tetZ[1] - tetZ[0],
		//		tetX[2] - tetX[0], tetY[2] - tetY[0], tetZ[2] - tetZ[0],
		//		tetX[3] - tetX[0], tetY[3] - tetY[0], tetZ[3] - tetZ[0]};

		//V = 1.0 / (a[0] * (a[4] * a[8] - a[5] * a[7]) + a[1] * (a[5] * a[6] - a[3] * a[8]) + a[2] * (a[3] * a[7] - a[4] * a[6]));
		V = 1.0 / (tetX[1] * (tetY[2] * tetZ[3] - tetZ[2] * tetY[3]) + 
			   tetY[1] * (tetZ[2] * tetX[3] - tetX[2] * tetZ[3]) +
			   tetZ[1] * (tetX[2] * tetY[3] - tetY[2] * tetX[3]));

		z41 = tetZ[3];// - tetZ[0];
		y34 = tetY[2] - tetY[3];
		z34 = tetZ[2] - tetZ[3];
		y41 = tetY[3];// - tetY[0];
		a11 = z41 * y34 - z34 * y41;

		x41 = tetX[3];// - tetX[0];
		x34 = tetX[2] - tetX[3];
		a12 = x41 * z34 - x34 * z41;

		a13 = y41 * x34 - y34 * x41;

		coordinates[1] = (a11 * X + a12 * Y + a13 * Z) * V;

		y12 = /*tetY[0]*/ - tetY[1];
		z12 = /*tetZ[0]*/ - tetZ[1];
		a21 = z41 * y12 - z12 * y41;

		x12 = /*tetX[0]*/ - tetX[1];
		a22 = x41 * z12 - x12 * z41;

		a23 = y41 * x12 - y12 * x41;

		coordinates[2] = (a21 * X + a22 * Y + a23 * Z) * V;
		index = coordinates[2] < coordinates[1] ? 2 : 1;

		z23 = tetZ[1] - tetZ[2];
		y23 = tetY[1] - tetY[2];
		a31 = z23 * y12 - z12 * y23;

		x23 = tetX[1] - tetX[2];
		a32 = x23 * z12 - x12 * z23;

		a33 = y23 * x12 - y12 * x23;

		if ((coordinates[3] = (a31 * X + a32 * Y + a33 * Z) * V) < coordinates[index]) index = 3;

		if ((coordinates[0] = 1.0 - coordinates[1] - coordinates[2] - coordinates[3]) < coordinates[index]) index = 0;

		if (coordinates[index] >= -epsilon) break;

		guess = links[(guess << 2) | index];
		
		if (guess == -1) break;
	}

	return guess;
}

__global__ void BlockedTracingKernelOfRK4(/*double *globalVertexPositions,
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
*/
					double startTime, double endTime, double timeStep, double epsilon,

					int sharedMemorySize, int multiple) {
/*
	cudaError_t err = cudaMemcpyToSymbol(pointers, &globalVertexPositions, sizeOfPointer, 0, cudaMemcpyHostToDevice) |
			  cudaMemcpyToSymbol(pointers, &globalTetrahedralConnectivities, sizeOfPointer, sizeOfPointer, cudaMemcpyHostToDevice) |
			  cudaMemcpyToSymbol(pointers, &globalTetrahedralLinks, sizeOfPointer, sizeOfPointer * 2, cudaMemcpyHostToDevice) |
			  cudaMemcpyToSymbol(pointers, &startOffsetInCell, sizeOfPointer, sizeOfPointer * 3, cudaMemcpyHostToDevice) |
			  cudaMemcpyToSymbol(pointers, &startOffsetInPoint, sizeOfPointer, sizeOfPointer * 4, cudaMemcpyHostToDevice) |
			  cudaMemcpyToSymbol(pointers, &vertexPositionsForBig, sizeOfPointer, sizeOfPointer * 5, cudaMemcpyHostToDevice) |
			  cudaMemcpyToSymbol(pointers, &startVelocitiesForBig, sizeOfPointer, sizeOfPointer * 6, cudaMemcpyHostToDevice) |
			  cudaMemcpyToSymbol(pointers, &endVelocitiesForBig, sizeOfPointer, sizeOfPointer * 7, cudaMemcpyHostToDevice) |
			  cudaMemcpyToSymbol(pointers, &blockedLocalConnectivities, sizeOfPointer, sizeOfPointer * 8, cudaMemcpyHostToDevice) |
			  cudaMemcpyToSymbol(pointers, &blockedLocalLinks, sizeOfPointer, sizeOfPointer * 9, cudaMemcpyHostToDevice) |
			  cudaMemcpyToSymbol(pointers, &blockedGlobalCellIDs, sizeOfPointer, sizeOfPointer * 10, cudaMemcpyHostToDevice) |
			  cudaMemcpyToSymbol(pointers, &activeBlockList, sizeOfPointer, sizeOfPointer * 11, cudaMemcpyHostToDevice) |
			  cudaMemcpyToSymbol(pointers, &blockOfGroups, sizeOfPointer, sizeOfPointer * 12, cudaMemcpyHostToDevice) |
			  cudaMemcpyToSymbol(pointers, &offsetInBlocks, sizeOfPointer, sizeOfPointer * 13, cudaMemcpyHostToDevice) |
			  cudaMemcpyToSymbol(pointers, &stage, sizeOfPointer, sizeOfPointer * 14, cudaMemcpyHostToDevice) |
			  cudaMemcpyToSymbol(pointers, &lastPosition, sizeOfPointer, sizeOfPointer * 15, cudaMemcpyHostToDevice) |
			  cudaMemcpyToSymbol(pointers, &k1, sizeOfPointer, sizeOfPointer * 16, cudaMemcpyHostToDevice) |
			  cudaMemcpyToSymbol(pointers, &k2, sizeOfPointer, sizeOfPointer * 17, cudaMemcpyHostToDevice) |
			  cudaMemcpyToSymbol(pointers, &k3, sizeOfPointer, sizeOfPointer * 18, cudaMemcpyHostToDevice) |
			  cudaMemcpyToSymbol(pointers, &pastTimes, sizeOfPointer * 19, cudaMemcpyHostToDevice) |
			  cudaMemcpyToSymbol(pointers, &placesOfInterest, sizeOfPointer, sizeOfPointer * 20, cudaMemcpyHostToDevice) |
			  cudaMemcpyToSymbol(pointers, &startOffsetInParticle, sizeOfPointer, sizeOfPointer * 21, cudaMemcpyHostToDevice) |
			  cudaMemcpyToSymbol(pointers, &blockedActiveParticleIDList, sizeOfPointer, sizeOfPointer * 22, cudaMemcpyHostToDevice) |
			  cudaMemcpyToSymbol(pointers, &cellLocations, sizeOfPointer, sizeOfPointer * 23, cudaMemcpyHostToDevice) |
			  cudaMemcpyToSymbol(pointers, &exitCells, sizeOfPointer, sizeOfPointer * 24, cudaMemcpyHostToDevice) |
*/


	__shared__ extern char sharedMemory[];
	//char *sharedMemory;

	// Get work group ID
	//int groupID = blockIdx.x;
	
	// Get number of threads in a work group
	//int numOfThreads = blockDim.x;

	// Get local thread ID
	//int localID = threadIdx.x;

	// Get active block ID
	int activeBlockID = blockOfGroups[blockIdx.x/*groupID*/];
	//int activeBlockID = blockOfGroups[groupID];

	// Get interesting block ID of the work group
	int i, idx;
	i = activeBlockList[activeBlockID];
	//int interestingBlockID = ((int *)pointers[11])[activeBlockID];
	//int interestingBlockID = activeBlockList[activeBlockID];

	// Declare some arrays
	double *vertexPositions;
	double *startVelocities;
	double *endVelocities;
	int *connectivities;
	int *links;

	int startCell = startOffsetInCell[i/*interestingBlockID*/];
	int offset/*int startPoint*/ = startOffsetInPoint[i/*interestingBlockID*/];
	idx = offset * 3;
	//int startCell = startOffsetInCell[interestingBlockID];
	//int startPoint = startOffsetInPoint[interestingBlockID];

	int numOfCells = startOffsetInCell[i/*interestingBlockID*/ + 1] - startCell;
	int numOfPoints = startOffsetInPoint[i/*interestingBlockID*/ + 1] - offset/*startPoint*/;

	//startPoint *= 3;
	//int numOfCells = startOffsetInCell[interestingBlockID + 1] - startCell;
	//int numOfPoints = startOffsetInPoint[interestingBlockID + 1] - startPoint;

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

		for (/*int*/ i = threadIdx.x/*localID*/; i < numOfPoints * 3; i += blockDim.x/*numOfThreads*/) {
			vertexPositions[i] = vertexPositionsForBig[idx/*startPoint * 3*/ + i];
			startVelocities[i] = startVelocitiesForBig[idx/*startPoint * 3*/ + i];
			endVelocities[i] = endVelocitiesForBig[idx/*startPoint * 3*/ + i];
			//vertexPositions[i] = vertexPositionsForBig[startPoint * 3 + i];
			//startVelocities[i] = startVelocitiesForBig[startPoint * 3 + i];
			//endVelocities[i] = endVelocitiesForBig[startPoint * 3 + i];
		}

		for (/*int*/ i = threadIdx.x/*localID*/; i < (numOfCells << 2); i += blockDim.x/*numOfThreads*/) {
			connectivities[i] = blockedLocalConnectivities[(startCell << 2) + i];
			links[i] = blockedLocalLinks[(startCell << 2) + i];
			//connectivities[i] = blockedLocalConnectivities[(startCell << 2) + i];
			//links[i] = blockedLocalLinks[(startCell << 2) + i];
		}

		//__syncthreads();
	} else { // This branch fills in the global memory
		// Initialize vertexPositions, startVelocities and endVelocities
		vertexPositions = vertexPositionsForBig + idx/*startPoint * 3*/;
		startVelocities = startVelocitiesForBig + idx/*startPoint * 3*/;
		endVelocities = endVelocitiesForBig + idx/*startPoint * 3*/;
		//vertexPositions = vertexPositionsForBig + startPoint * 3;
		//startVelocities = startVelocitiesForBig + startPoint * 3;
		//endVelocities = endVelocitiesForBig + startPoint * 3;

		// Initialize connectivities and links
		connectivities = blockedLocalConnectivities + (startCell << 2);
		links = blockedLocalLinks + (startCell << 2);
		//connectivities = blockedLocalConnectivities + (startCell << 2);
		//links = blockedLocalLinks + (startCell << 2);
	}

	__syncthreads();

	int numOfActiveParticles = startOffsetInParticle[activeBlockID + 1] - startOffsetInParticle[activeBlockID];
	/*int*/ offset = offsetInBlocks[blockIdx.x/*groupID*/] * blockDim.x/*numOfThreads*/ * multiple;
	//int numOfActiveParticles = startOffsetInParticle[activeBlockID + 1] - startOffsetInParticle[activeBlockID];
	//int offset = offsetInBlocks[groupID] * numOfThreads * multiple;

	int activeParticleID, currStage, currCell, nextCell;
	//int nextGlobalCell;

	double currTime;
	double currLastPosition[3], currK1[3], currK2[3], currK3[3], currK4[3];
	double placeOfInterest[3];
	double coordinates[4];
	double alpha, beta;

	double vecX[4], vecY[4], vecZ[4];
	double *currK;

	for (idx = threadIdx.x/*localID*/; idx < blockDim.x/*numOfThreads*/ * multiple; idx += blockDim.x/*numOfThreads*/) {
		//int arrayIdx = offsetInBlocks[groupID] * numOfThreads + localID;
		activeParticleID = offset + idx;
		//int arrayIdx = offset + idx;

		//if (arrayIdx < numOfActiveParticles) {
		if (activeParticleID < numOfActiveParticles) {
			// activeParticleID here means the initial active particle ID
			//arrayIdx += ((int *)pointers[21])[activeBlockID];
			//int activeParticleID = ((int *)pointers[22])[arrayIdx];
			activeParticleID = blockedActiveParticleIDList[activeParticleID + startOffsetInParticle[activeBlockID]];
			//arrayIdx += startOffsetInParticle[activeBlockID];
			//int activeParticleID = blockedActiveParticleIDList[arrayIdx];

			// Initialize the particle status
			/*int*/ currStage = stage[activeParticleID];
			/*int*/ currCell = cellLocations[activeParticleID];
			//int currStage = stage[activeParticleID];
			//int currCell = cellLocations[activeParticleID];

			/*double*/ currTime = pastTimes[activeParticleID];
			//double currTime = pastTimes[activeParticleID];

			/*double currLastPosition[3];*/
			currLastPosition[0] = lastPosition[activeParticleID * 3];
			currLastPosition[1] = lastPosition[activeParticleID * 3 + 1];
			currLastPosition[2] = lastPosition[activeParticleID * 3 + 2];
			//currLastPosition[0] = lastPosition[activeParticleID * 3];
			//currLastPosition[1] = lastPosition[activeParticleID * 3 + 1];
			//currLastPosition[2] = lastPosition[activeParticleID * 3 + 2];
			/*double currK1[3], currK2[3], currK3[3], currK4[3];*/
			if (currStage > 0) {
				currK1[0] = k1[activeParticleID * 3];
				currK1[1] = k1[activeParticleID * 3 + 1];
				currK1[2] = k1[activeParticleID * 3 + 2];
				//currK1[0] = k1[activeParticleID * 3];
				//currK1[1] = k1[activeParticleID * 3 + 1];
				//currK1[2] = k1[activeParticleID * 3 + 2];
			}
			if (currStage > 1) {
				currK2[0] = k2[activeParticleID * 3];
				currK2[1] = k2[activeParticleID * 3 + 1];
				currK2[2] = k2[activeParticleID * 3 + 2];
				//currK2[0] = k2[activeParticleID * 3];
				//currK2[1] = k2[activeParticleID * 3 + 1];
				//currK2[2] = k2[activeParticleID * 3 + 2];
			}
			if (currStage > 2) {
				currK3[0] = k3[activeParticleID * 3];
				currK3[1] = k3[activeParticleID * 3 + 1];
				currK3[2] = k3[activeParticleID * 3 + 2];
				//currK3[0] = k3[activeParticleID * 3];
				//currK3[1] = k3[activeParticleID * 3 + 1];
				//currK3[2] = k3[activeParticleID * 3 + 2];
			}

			// At least one loop is executed.
			while (true) {
				/*double placeOfInterest[3];*/
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

				/*double coordinates[4];*/

				//int nextCell;
				nextCell = FindCell(placeOfInterest, connectivities, links, vertexPositions, /*doubleValues[3]*/epsilon, currCell, coordinates,
							vecX, vecY, vecZ);

				if (nextCell == -1 || currTime >= /*doubleValues[1]*/endTime) {
					// Find the next cell globally
					//int globalCellID = ((int *)pointers[10])[startCell + currCell];
					//int globalCellID = blockedGlobalCellIDs[startCell + currCell];
					//int nextGlobalCell;
				
					if (nextCell != -1)
						nextCell/*nextGlobalCell*/ = blockedGlobalCellIDs[startCell + nextCell];
						//nextGlobalCell = blockedGlobalCellIDs[startCell + nextCell];
					else
						nextCell/*nextGlobalCell*/ = FindCell(placeOfInterest, globalTetrahedralConnectivities, globalTetrahedralLinks, globalVertexPositions,
									/*doubleValues[3]*/epsilon, /*globalCellID*/blockedGlobalCellIDs[startCell + currCell], coordinates,
									vecX, vecY, vecZ);
						//nextGlobalCell = FindCell(placeOfInterest, globalTetrahedralConnectivities,
						//			globalTetrahedralLinks, globalVertexPositions,
						//			epsilon, globalCellID, coordinates);

					if (currTime >= /*doubleValues[1]*/endTime && nextCell/*nextGlobalCell*/ != -1) nextCell = -2 - nextCell;//nextGlobalCell = -2 - nextGlobalCell;

					pastTimes[activeParticleID] = currTime;
					//pastTimes[activeParticleID] = currTime;

					stage[activeParticleID] = currStage;
					//stage[activeParticleID] = currStage;

					lastPosition[activeParticleID * 3] = currLastPosition[0];
					lastPosition[activeParticleID * 3 + 1] = currLastPosition[1];
					lastPosition[activeParticleID * 3 + 2] = currLastPosition[2];
					//lastPosition[activeParticleID * 3] = currLastPosition[0];
					//lastPosition[activeParticleID * 3 + 1] = currLastPosition[1];
					//lastPosition[activeParticleID * 3 + 2] = currLastPosition[2];

					placesOfInterest[activeParticleID * 3] = placeOfInterest[0];
					placesOfInterest[activeParticleID * 3 + 1] = placeOfInterest[1];
					placesOfInterest[activeParticleID * 3 + 2] = placeOfInterest[2];
					//placesOfInterest[activeParticleID * 3] = placeOfInterest[0];
					//placesOfInterest[activeParticleID * 3 + 1] = placeOfInterest[1];
					//placesOfInterest[activeParticleID * 3 + 2] = placeOfInterest[2];

					exitCells[activeParticleID] = nextCell;//nextGlobalCell;
					//exitCells[activeParticleID] = nextGlobalCell;
		
					if (currStage > 0) { // currStage > 0
						k1[activeParticleID * 3] = currK1[0];
						k1[activeParticleID * 3 + 1] = currK1[1];
						k1[activeParticleID * 3 + 2] = currK1[2];
						//k1[activeParticleID * 3] = currK1[0];
						//k1[activeParticleID * 3 + 1] = currK1[1];
						//k1[activeParticleID * 3 + 2] = currK1[2];
					}	
					if (currStage > 1) { // currStage > 1
						//((double *)pointers[17])[activeParticleID * 3] = currK2[0];
						//((double *)pointers[17])[activeParticleID * 3 + 1] = currK2[1];
						//((double *)pointers[17])[activeParticleID * 3 + 2] = currK2[2];
						k2[activeParticleID * 3] = currK2[0];
						k2[activeParticleID * 3 + 1] = currK2[1];
						k2[activeParticleID * 3 + 2] = currK2[2];
					}	
					if (currStage > 2) { // currStage > 2
						//((double *)pointers[18])[activeParticleID * 3] = currK3[0];
						//((double *)pointers[18])[activeParticleID * 3 + 1] = currK3[1];
						//((double *)pointers[18])[activeParticleID * 3 + 2] = currK3[2];
						k3[activeParticleID * 3] = currK3[0];
						k3[activeParticleID * 3 + 1] = currK3[1];
						k3[activeParticleID * 3 + 2] = currK3[2];
					}
					
					break;
				}

				currCell = nextCell;

				/*double*/ alpha /*exactTime*/ = currTime;
				switch (currStage) {
				case 0: break;
				case 1:
				case 2: alpha /*exactTime*/ += /*doubleValues[2]*/timeStep * 0.5; break;
				case 3: alpha /*exactTime*/ += /*doubleValues[2]*/timeStep; break;
				}

				/*double*/ alpha = (/*doubleValues[1]*/endTime - alpha/*exactTime*/) / (/*doubleValues[1]*/endTime - /*doubleValues[0]*/startTime);
				/*double*/ beta = 1.0 - alpha;

				/*double vecX[4], vecY[4], vecZ[4];*/

				for (/*int*/ i = 0; i < 4; i++) {
					//int pointID = connectivities[(nextCell << 2) | i];
					nextCell = connectivities[(currCell << 2) | i] * 3;

					vecX[i] = startVelocities[nextCell] * alpha + endVelocities[nextCell] * beta;
					vecY[i] = startVelocities[nextCell + 1] * alpha + endVelocities[nextCell + 1] * beta;
					vecZ[i] = startVelocities[nextCell + 2] * alpha + endVelocities[nextCell + 2] * beta;

					//vecX[i] = startVelocities[pointID * 3] * alpha + endVelocities[pointID * 3] * beta;
					//vecY[i] = startVelocities[pointID * 3 + 1] * alpha + endVelocities[pointID * 3 + 1] * beta;
					//vecZ[i] = startVelocities[pointID * 3 + 2] * alpha + endVelocities[pointID * 3 + 2] * beta;
				}

				/*double *currK;*/
				switch (currStage) {
				case 0: currK = currK1; break;
				case 1: currK = currK2; break;
				case 2: currK = currK3; break;
				case 3: currK = currK4; break;
				}

				currK[0] = currK[1] = currK[2] = 0;

				for (/*int*/ i = 0; i < 4; i++) {
					currK[0] += vecX[i] * coordinates[i];
					currK[1] += vecY[i] * coordinates[i];
					currK[2] += vecZ[i] * coordinates[i];
				}

				currK[0] *= /*doubleValues[2]*/timeStep;
				currK[1] *= /*doubleValues[2]*/timeStep;
				currK[2] *= /*doubleValues[2]*/timeStep;

				if (currStage == 3) {
					currTime += /*doubleValues[2]*/timeStep;

					for (/*int*/ i = 0; i < 3; i++)
						currLastPosition[i] += (currK1[i] + 2.0 * (currK2[i] + currK3[i]) + currK4[i]) / 6.0;

					currStage = 0;
				} else
					currStage++;
			}
		} else break;

	}

}

extern "C"
void InitializeConstantsForBlockedTracingKernelOfRK4(double *deviceGlobalVertexPositions,
			int *deviceGlobalTetrahedralConnectivities,
			int *deviceGlobalTetrahedralLinks,

			int *deviceStartOffsetInCell,
			int *deviceStartOffsetInPoint,

			double *deviceVertexPositionsForBig,
			double *deviceStartVelocitiesForBig,
			double *deviceEndVelocitiesForBig,

			int *deviceBlockedLocalConnectivities,
			int *deviceBlockedLocalLinks,
			int *deviceBlockedGlobalCellIDs,

			int *deviceActiveBlockList, // Map active block ID to interesting block ID

			int *deviceBlockOfGroups,
			int *deviceOffsetInBlocks,

			int *deviceStage,
			double *deviceLastPosition,
			double *deviceK1,
			double *deviceK2,
			double *deviceK3,
			double *devicePastTimes,

			double *devicePlacesOfInterest,

			int *deviceStartOffsetInParticle,
			int *deviceBlockedActiveParticleIDList,
			int *deviceCellLocations,

			int *deviceExitCells,

			double hostTimeStep, double hostEpsilon) {
	int sizeOfPointer = sizeof(void *);

	cudaError_t err = (cudaError_t)(cudaMemcpyToSymbol(globalVertexPositions, &deviceGlobalVertexPositions, sizeOfPointer, 0, cudaMemcpyHostToDevice) |
			  cudaMemcpyToSymbol(globalTetrahedralConnectivities, &deviceGlobalTetrahedralConnectivities, sizeOfPointer, 0, cudaMemcpyHostToDevice) |
			  cudaMemcpyToSymbol(globalTetrahedralLinks, &deviceGlobalTetrahedralLinks, sizeOfPointer, 0, cudaMemcpyHostToDevice) |
			  cudaMemcpyToSymbol(startOffsetInCell, &deviceStartOffsetInCell, sizeOfPointer, 0, cudaMemcpyHostToDevice) |
			  cudaMemcpyToSymbol(startOffsetInPoint, &deviceStartOffsetInPoint, sizeOfPointer, 0, cudaMemcpyHostToDevice) |
			  cudaMemcpyToSymbol(vertexPositionsForBig, &deviceVertexPositionsForBig, sizeOfPointer, 0, cudaMemcpyHostToDevice) |
			  cudaMemcpyToSymbol(startVelocitiesForBig, &deviceStartVelocitiesForBig, sizeOfPointer, 0, cudaMemcpyHostToDevice) |
			  cudaMemcpyToSymbol(endVelocitiesForBig, &deviceEndVelocitiesForBig, sizeOfPointer, 0, cudaMemcpyHostToDevice) |
			  cudaMemcpyToSymbol(blockedLocalConnectivities, &deviceBlockedLocalConnectivities, sizeOfPointer, 0, cudaMemcpyHostToDevice) |
			  cudaMemcpyToSymbol(blockedLocalLinks, &deviceBlockedLocalLinks, sizeOfPointer, 0, cudaMemcpyHostToDevice) |
			  cudaMemcpyToSymbol(blockedGlobalCellIDs, &deviceBlockedGlobalCellIDs, sizeOfPointer, 0, cudaMemcpyHostToDevice) |
			  cudaMemcpyToSymbol(activeBlockList, &deviceActiveBlockList, sizeOfPointer, 0, cudaMemcpyHostToDevice) |
			  cudaMemcpyToSymbol(blockOfGroups, &deviceBlockOfGroups, sizeOfPointer, 0, cudaMemcpyHostToDevice) |
			  cudaMemcpyToSymbol(offsetInBlocks, &deviceOffsetInBlocks, sizeOfPointer, 0, cudaMemcpyHostToDevice) |
			  cudaMemcpyToSymbol(stage, &deviceStage, sizeOfPointer, 0, cudaMemcpyHostToDevice) |
			  cudaMemcpyToSymbol(lastPosition, &deviceLastPosition, sizeOfPointer, 0, cudaMemcpyHostToDevice) |
			  cudaMemcpyToSymbol(k1, &deviceK1, sizeOfPointer, 0, cudaMemcpyHostToDevice) |
			  cudaMemcpyToSymbol(k2, &deviceK2, sizeOfPointer, 0, cudaMemcpyHostToDevice) |
			  cudaMemcpyToSymbol(k3, &deviceK3, sizeOfPointer, 0, cudaMemcpyHostToDevice) |
			  cudaMemcpyToSymbol(pastTimes, &devicePastTimes, sizeOfPointer, 0, cudaMemcpyHostToDevice) |
			  cudaMemcpyToSymbol(placesOfInterest, &devicePlacesOfInterest, sizeOfPointer, 0, cudaMemcpyHostToDevice) |
			  cudaMemcpyToSymbol(startOffsetInParticle, &deviceStartOffsetInParticle, sizeOfPointer, 0, cudaMemcpyHostToDevice) |
			  cudaMemcpyToSymbol(blockedActiveParticleIDList, &deviceBlockedActiveParticleIDList, sizeOfPointer, 0, cudaMemcpyHostToDevice) |
			  cudaMemcpyToSymbol(cellLocations, &deviceCellLocations, sizeOfPointer, 0, cudaMemcpyHostToDevice) |
			  cudaMemcpyToSymbol(exitCells, &deviceExitCells, sizeOfPointer, 0, cudaMemcpyHostToDevice) |

			  cudaMemcpyToSymbol(timeStep, &hostTimeStep, sizeof(double), 0, cudaMemcpyHostToDevice) |
			  cudaMemcpyToSymbol(epsilon, &hostEpsilon, sizeof(double), 0, cudaMemcpyHostToDevice));
/*
	err = (cudaError_t)((int)err | cudaMemcpyToSymbol(doubleValues, &startTime, sizeof(double), 0, cudaMemcpyHostToDevice) |
		cudaMemcpyToSymbol(doubleValues, &endTime, sizeof(double), sizeof(double), cudaMemcpyHostToDevice) |
		cudaMemcpyToSymbol(doubleValues, &timeStep, sizeof(double), sizeof(double) * 2, cudaMemcpyHostToDevice) |
		cudaMemcpyToSymbol(doubleValues, &epsilon, sizeof(double), sizeof(double) * 3, cudaMemcpyHostToDevice));
*/
	if (err) {
		printf("Symbol Memcpy Failure\n");
		exit(0);
	}
}

extern "C"
void BlockedTracingOfRK4(/*double *globalVertexPositions,
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

			int *exitCells,*/

			double startTime, double endTime, double timeStep, double epsilon, int numOfActiveBlocks,

			int blockSize, int sharedMemorySize, int multiple) {
	dim3 dimBlock(blockSize, 1, 1);
	dim3 dimGrid(numOfActiveBlocks, 1, 1);

	BlockedTracingKernelOfRK4<<<dimGrid, dimBlock, sharedMemorySize>>>(/*globalVertexPositions,
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
*/
					startTime, endTime, timeStep, epsilon,

					sharedMemorySize, multiple);

	cudaError_t err = cudaDeviceSynchronize();
	if (err) {
		printf("err = %d\n", err);
		cudaGetErrorString(err);
		exit(0);
	}
}
