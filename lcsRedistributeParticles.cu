/******************************************************************
File		:	lcsRedistributeParticles.cu
Author		:	Mingcheng Chen
Last Update	:	January 29th, 2013
*******************************************************************/

#include <stdio.h>

#define BLOCK_SIZE 512

__device__ inline int Sign(double a, double eps) {
	return a < -eps ? -1 : a > eps;
}

__device__ inline int GetLocalTetID(int blockID, int tetID,
			 int *startOffsetsInLocalIDMap,
			 int *blocksOfTets,
			 int *localIDsOfTets) { // blockID and tetID are all global IDs.
	int offset = startOffsetsInLocalIDMap[tetID];
	int endOffset = -1;
	while (1) {
		if (blocksOfTets[offset] == blockID) return localIDsOfTets[offset];
		if (endOffset == -1) endOffset = startOffsetsInLocalIDMap[tetID + 1];

		offset++;
		if (offset >= endOffset) return -1;
	}
}

__device__ inline int GetBlockID(int x, int y, int z, int numOfBlocksInY, int numOfBlocksInZ) {
	return (x * numOfBlocksInY + y) * numOfBlocksInZ + z;
}

__global__ void CollectActiveBlocksKernel(int *activeParticles,
				int *exitCells,
				double *placesOfInterest,

				int *localTetIDs,
				int *blockLocations,

				int *interestingBlockMap,

				int *startOffsetsInLocalIDMap,
				int *blocksOfTets,
				int *localIDsOfTets,

				int *interestingBlockMarks,

				int *activeBlocks,
				int *activeBlockIndices,
				int *numOfActiveBlocks, // Initially 0

				int mark,
				int numOfActiveParticles, //int numOfStages,
				int numOfBlocksInX, int numOfBlocksInY, int numOfBlocksInZ,
				double globalMinX, double globalMinY, double globalMinZ,
				double blockSize,
				double epsilon) {
	int globalID = blockDim.x * blockIdx.x + threadIdx.x;

	if (globalID < numOfActiveParticles) {
		int particleID = activeParticles[globalID];
		double posX = placesOfInterest[particleID * 3];
		double posY = placesOfInterest[particleID * 3 + 1];
		double posZ = placesOfInterest[particleID * 3 + 2];

		int x = (int)((posX - globalMinX) / blockSize);
		int y = (int)((posY - globalMinY) / blockSize);
		int z = (int)((posZ - globalMinZ) / blockSize);

		// Intuitive block ID
		int blockID = GetBlockID(x, y, z, numOfBlocksInY, numOfBlocksInZ);
		int tetID = exitCells[particleID];

		int localTetID = GetLocalTetID(blockID, tetID, startOffsetsInLocalIDMap, blocksOfTets, localIDsOfTets);

		/// DEBUG ///
		/*
		if (particleID == 303)
			printf("x, y, z: %d %d %d\n", x, y, z);
		*/


		if (localTetID == -1) {
			int dx[3], dy[3], dz[3];
			int lx = 1, ly = 1, lz = 1;
			dx[0] = dy[0] = dz[0] = 0;

			double xLower = globalMinX + x * blockSize;
			double yLower = globalMinY + y * blockSize;
			double zLower = globalMinZ + z * blockSize;

			if (!Sign(xLower - posX, 10 * epsilon)) dx[lx++] = -1;
			if (!Sign(yLower - posY, 10 * epsilon)) dy[ly++] = -1;
			if (!Sign(zLower - posZ, 10 * epsilon)) dz[lz++] = -1;

			if (!Sign(xLower + blockSize - posX, 10 * epsilon)) dx[lx++] = 1;
			if (!Sign(yLower + blockSize - posY, 10 * epsilon)) dy[ly++] = 1;
			if (!Sign(zLower + blockSize - posZ, 10 * epsilon)) dz[lz++] = 1;

			// Check every necessary neightbor
			for (int i = 0; localTetID == -1 && i < lx; i++)
				for (int j = 0; localTetID == -1 && j < ly; j++)
					for (int k = 0; k < lz; k++) {
						if (i + j + k == 0) continue;
						int _x = x + dx[i];
						int _y = y + dy[j];
						int _z = z + dz[k];

						if (_x < 0 || _y < 0 || _z < 0 ||
						    _x >= numOfBlocksInX || _y >= numOfBlocksInY || _z >= numOfBlocksInZ)
							continue;

						blockID = GetBlockID(_x, _y, _z, numOfBlocksInY, numOfBlocksInZ);

						/// DEBUG ///
						// if (particleID == 303 && tetID == 6825504) printf("_x = %d, _y = %d, _z = %d, blockID = %d\n", _x, _y, _z, blockID); 

						localTetID = GetLocalTetID(blockID, tetID, startOffsetsInLocalIDMap,
									   blocksOfTets, localIDsOfTets);

						if (localTetID != -1) break;
					}

			/// DEBUG ///
			if (localTetID == -1) {
				/*
				if (particleID == 303) {
					printf("%lf %lf %lf\n", posX, posY, posZ);
					printf("tetID = %d\n", tetID);
					printf("%lf %lf %lf\n", xLower, yLower, zLower);
					for (int i = 0; i < lx; i++)
						printf(" %d", dx[i]);
					printf("\n");
					for (int i = 0; i < ly; i++)
						printf(" %d", dy[i]);
					printf("\n");
					for (int i = 0; i < lz; i++)
						printf(" %d", dz[i]);
					printf("\n");
				}
				return;
				*/
				while (1);
			}
		}

		// localTetID must not be -1 at that point.

		localTetIDs[particleID] = localTetID;

		int interestingBlockID = interestingBlockMap[blockID];
		blockLocations[particleID] = interestingBlockID;

		int oldMark = atomicAdd(interestingBlockMarks + interestingBlockID, 0);

		int index;

		if (oldMark < mark) {
			int delta = mark - oldMark;
			int newMark = atomicAdd(interestingBlockMarks + interestingBlockID, delta);

			if (newMark >= mark)
				atomicAdd(interestingBlockMarks + interestingBlockID, -delta);
			else {
				index = atomicAdd(numOfActiveBlocks, 1);
				activeBlocks[index] = interestingBlockID;
				activeBlockIndices[interestingBlockID] = index;
			}
		}

		// This one cannot be calculated in that kernel
		//activeBlockOfParticles[particleID] = index;
	}
}

__global__ void GetNumOfParticlesByStageInBlocksKernel(int *numOfParticlesByStageInBlocks,
					       int *particleOrders,
					       int *stages,
					       int *activeParticles,
					       
					       int *blockLocations,
					       int *activeBlockIndices,
					       int numOfStages, int numOfActiveParticles) {
	int globalID = blockDim.x * blockIdx.x + threadIdx.x;
	if (globalID < numOfActiveParticles) {
		int particleID = activeParticles[globalID];

		int posi = activeBlockIndices[blockLocations[particleID]] * numOfStages + stages[particleID];
		particleOrders[particleID] = atomicAdd(numOfParticlesByStageInBlocks + posi, 1);
	}
}

__global__ void CollectParticlesToBlocksKernel(int *numOfParticlesByStageInBlocks, // Now it is a prefix sum array.
				       int *particleOrders,
				       int *stages,
				       int *activeParticles,
				       int *blockLocations,
				       int *activeBlockIndices,

				       int *blockedParticleList,
				       int numOfStages, int numOfActiveParticles
				       ) {
	int globalID = blockDim.x * blockIdx.x + threadIdx.x;

	if (globalID < numOfActiveParticles) {
		int particleID = activeParticles[globalID];

		int interestingBlockID = blockLocations[particleID];
		int activeBlockID = activeBlockIndices[interestingBlockID];
		int stage = stages[particleID];

		int position = numOfParticlesByStageInBlocks[activeBlockID * numOfStages + stage] + particleOrders[particleID];

		blockedParticleList[position] = particleID;
	}
}

extern "C"
void CollectActiveBlocks(int *activeParticles,
				int *exitCells,
				double *placesOfInterest,

				int *localTetIDs,
				int *blockLocations,

				int *interestingBlockMap,

				int *startOffsetsInLocalIDMap,
				int *blocksOfTets,
				int *localIDsOfTets,

				int *interestingBlockMarks,

				int *activeBlocks,
				int *activeBlockIndices,
				int *numOfActiveBlocks, // Initially 0

				int mark,
				int numOfActiveParticles, //int numOfStages,
				int numOfBlocksInX, int numOfBlocksInY, int numOfBlocksInZ,
				double globalMinX, double globalMinY, double globalMinZ,
				double blockSize,
				double epsilon) {
	dim3 dimBlock(BLOCK_SIZE, 1, 1);
	dim3 dimGrid((numOfActiveParticles - 1) / dimBlock.x + 1, 1, 1);

	CollectActiveBlocksKernel<<<dimGrid, dimBlock>>>(activeParticles, exitCells, placesOfInterest, localTetIDs, blockLocations, interestingBlockMap,
							startOffsetsInLocalIDMap, blocksOfTets, localIDsOfTets, interestingBlockMarks, activeBlocks,
							activeBlockIndices, numOfActiveBlocks, // Initially 0
							mark, numOfActiveParticles, numOfBlocksInX, numOfBlocksInY, numOfBlocksInZ,
							globalMinX, globalMinY, globalMinZ, blockSize, epsilon);

	cudaError_t err = cudaDeviceSynchronize();
	if (err) {
		cudaGetErrorString(err);
		exit(0);
	}
}

extern "C"
void GetNumOfParticlesByStageInBlocks(int *numOfParticlesByStageInBlocks,
					       int *particleOrders,
					       int *stages,
					       int *activeParticles,
					       
					       int *blockLocations,
					       int *activeBlockIndices,
					       int numOfStages, int numOfActiveParticles) {
	dim3 dimBlock(BLOCK_SIZE, 1, 1);
	dim3 dimGrid((numOfActiveParticles - 1) / dimBlock.x + 1, 1, 1);

	GetNumOfParticlesByStageInBlocksKernel<<<dimGrid, dimBlock>>>(numOfParticlesByStageInBlocks, particleOrders, stages, activeParticles, blockLocations,
									activeBlockIndices, numOfStages, numOfActiveParticles);

	cudaError_t err = cudaDeviceSynchronize();
	if (err) {
		cudaGetErrorString(err);
		exit(0);
	}

}

extern "C"
void CollectParticlesToBlocks(int *numOfParticlesByStageInBlocks, // Now it is a prefix sum array.
				       int *particleOrders,
				       int *stages,
				       int *activeParticles,
				       int *blockLocations,
				       int *activeBlockIndices,

				       int *blockedParticleList,
				       int numOfStages, int numOfActiveParticles) {
	dim3 dimBlock(BLOCK_SIZE, 1, 1);
	dim3 dimGrid((numOfActiveParticles - 1 ) / dimBlock.x + 1, 1, 1);

	CollectParticlesToBlocksKernel<<<dimGrid, dimBlock>>>(numOfParticlesByStageInBlocks, // Now it is a prefix sum array.
								particleOrders, stages, activeParticles, blockLocations, activeBlockIndices, blockedParticleList,
								numOfStages, numOfActiveParticles);

	cudaError_t err = cudaDeviceSynchronize();
	if (err) {
		cudaGetErrorString(err);
		exit(0);
	}
}
