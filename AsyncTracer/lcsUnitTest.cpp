/**********************************************
File			:	lcsUnitTest.cpp
Author			:	Mingcheng Chen
Last Update		:	November 25th, 2012
***********************************************/

#include "lcsUnitTest.h"
#include "lcsUtility.h"

////////////////////////////////////////////////
bool CheckPlane(const lcs::Vector &p1, const lcs::Vector &p2, const lcs::Vector &p3,
				const lcs::Tetrahedron &tetrahedron,
				double localMinX, double localMinY, double localMinZ,
				double blockSize, double epsilon) {
	if (!lcs::Sign(Cross(p1 - p2, p1 - p3).Length(), epsilon)) return false;
	bool tetPos = 0, tetNeg = 0, blkPos = 0, blkNeg = 0;
	for (int i = 0; i < 4; i++) {
		lcs::Vector point = tetrahedron.GetVertex(i);
		double directedVolume = lcs::Mixed(p1 - point, p2 - point, p3 - point);
		if (lcs::Sign(directedVolume, epsilon) < 0) tetNeg = 1;
		if (lcs::Sign(directedVolume, epsilon) > 0) tetPos = 1;
		if (tetNeg * tetPos) return false;
	}
	for (int i = 0; i < 8; i++) {
		lcs::Vector point(localMinX, localMinY, localMinZ);
		if (i & 1) point.SetX(point.GetX() + blockSize);
		if (i & 2) point.SetY(point.GetY() + blockSize);
		if (i & 4) point.SetZ(point.GetZ() + blockSize);
		double directedVolume = lcs::Mixed(p1 - point, p2 - point, p3 - point);
		if (lcs::Sign(directedVolume, epsilon) < 0) blkNeg = 1;
		if (lcs::Sign(directedVolume, epsilon) > 0) blkPos = 1;
		if (blkNeg * blkPos) return false;
	}
	if (tetNeg && blkNeg || tetPos && blkPos) return false;
	return true;
}

bool TetrahedronContainsPoint(const lcs::Tetrahedron &tet, const lcs::Vector &pt, double eps) {
	double coordinates[4];
	tet.CalculateNaturalCoordinates(pt, coordinates);
	for (int i = 0; i < 4; i++)
		if (lcs::Sign(coordinates[i], eps) < 0) {
			printf("coordiantes[%d] = %lf\n", i, coordinates[i]);
			return false;
		}
	return true;
}

////////////////////////////////////////////////
void lcs::UnitTestForTetBlkIntersection(lcs::TetrahedralGrid *grid, double blockSize,
								   double globalMinX, double globalMinY, double globalMinZ,
								   int numOfBlocksInY, int numOfBlocksInZ,
								   int *queryTetrahedron, int *queryBlock,
								   bool *queryResults,
								   int numOfQueries,
								   double epsilon) {
	printf("Unit test for tetrahedron-block intersection ... ");

	for (int i = 0; i < numOfQueries; i++) {
		int tetID = queryTetrahedron[i];
		int blkID = queryBlock[i];
		lcs::Tetrahedron tet = grid->GetTetrahedron(tetID);
		int x, y, z;
		z = blkID % numOfBlocksInZ;
		int temp = blkID / numOfBlocksInZ;
		y = temp % numOfBlocksInY;
		x = temp / numOfBlocksInY;

		double localMinX = globalMinX + x * blockSize;
		double localMinY = globalMinY + y * blockSize;
		double localMinZ = globalMinZ + z * blockSize;

		// Test tetrahedral edge and block point
		bool flag = 0;

		for (int tetEdgeID = 0; !flag && tetEdgeID < 6; tetEdgeID++) {
			Vector p1, p2;
			switch (tetEdgeID) {
			case 0: {
						p1 = tet.GetVertex(0);
						p2 = tet.GetVertex(1);
					} break;
			case 1: {
						p1 = tet.GetVertex(0);
						p2 = tet.GetVertex(2);
					} break;
			case 2: {
						p1 = tet.GetVertex(0);
						p2 = tet.GetVertex(3);
					} break;
			case 3: {
						p1 = tet.GetVertex(1);
						p2 = tet.GetVertex(2);
					} break;
			case 4: {
						p1 = tet.GetVertex(1);
						p2 = tet.GetVertex(3);
					} break;
			case 5: {
						p1 = tet.GetVertex(2);
						p2 = tet.GetVertex(3);
					} break;
			}
			for (int dx = 0; !flag && dx <= 1; dx++)
				for (int dy = 0; !flag && dy <= 1; dy++)
					for (int dz = 0; dz <= 1; dz++) {
						Vector p3 = Vector(localMinX, localMinY, localMinZ) + Vector(dx, dy, dz) * blockSize;
						if (CheckPlane(p1, p2, p3, tet, localMinX, localMinY, localMinZ, blockSize, epsilon)) {
							flag = 1;

							//printf("tetrahedral edge and block point: %d, %d %d %d\n", tetEdgeID, dx, dy, dz);

							break;
						}
					}
		}

		// Test tetrahedral point and block edge
		for (int x1 = 0; !flag && x1 <= 1; x1++)
			for (int y1 = 0; !flag && y1 <= 1; y1++)
				for (int z1 = 0; !flag && z1 <= 1; z1++) {
					Vector p1(localMinX + x1 * blockSize, localMinY + y1 * blockSize, localMinZ + z1 *blockSize);
					for (int k = 0; !flag && k < 3; k++) {
						int x2 = x1, y2 = y1, z2 = z1;
						if (k == 0)
							if (x1 == 0) x2++;
							else continue;
						if (k == 1)
							if (y1 == 0) y2++;
							else continue;
						if (k == 2)
							if (z1 == 0) z2++;
							else continue;
						Vector p2(localMinX + x2 * blockSize, localMinY + y2 * blockSize, localMinZ + z2 * blockSize);
						for (int j = 0; j < 4; j++) {
							Vector p3 = tet.GetVertex(j);
							if (CheckPlane(p1, p2, p3, tet, localMinX, localMinY, localMinZ, blockSize, epsilon)) {
								flag = 1;
								break;
							}
						}
					}
				}

		bool result = !flag;

		if (result != queryResults[i]) {
			char error[100];
			sprintf(error, "Query %d has incorrect result.\ntet = %d, blk = %d\nkernel result: %d, CPU result: %d",
					i + 1, queryTetrahedron[i], queryBlock[i], queryResults[i], result);
			lcs::Error(error);
		}
	}

	printf("Passed\n");
}

void lcs::UnitTestForInitialCellLocations(lcs::TetrahedralGrid *grid,
										  int xRes, int yRes, int zRes,
										  double minX, double minY, double minZ,
										  double dx, double dy, double dz,
										  int *initialCellLocations,
										  double epsilon) {
	printf("Unit test (partial) for initial cell location ");

	int numOfGridPoints = (xRes + 1) * (yRes + 1) * (zRes + 1);
	int numOfMinusOnes = 0, numOfLocations = 0;
	for (int i = 0; i < numOfGridPoints; i++)
		if (initialCellLocations[i] == -1) numOfMinusOnes++;
		else numOfLocations++;
	printf("(outside: %d, inside: %d) ... ", numOfMinusOnes, numOfLocations);

	int idx = -1;
	for (int i = 0; i <= xRes; i++)
		for (int j = 0; j <= yRes; j++)
			for (int k = 0; k <= zRes; k++) {
				idx++;

				lcs::Vector point(minX + dx * i, minY + dy * j, minZ + dz * k);

				if (initialCellLocations[idx] != -1) {
					/// DEBUG ///
					//printf("idx = %d\n", idx);

					lcs::Tetrahedron tet = grid->GetTetrahedron(initialCellLocations[idx]);
					if (!TetrahedronContainsPoint(tet, point, epsilon)) {
						char str[100];
						sprintf(str, "Tetrahedron %d does not contain grid point %d (%d, %d, %d)",
								initialCellLocations[idx] + 1, idx, i, j, k);
						lcs::Error(str);
					}
					continue;
				}

				//for (int tetID = 0; tetID < grid->GetNumOfCells(); tetID++) {
				//	lcs::Tetrahedron tet = grid->GetTetrahedron(tetID);
				//	if (TetrahedronContainsPoint(tet, point, epsilon)) {
				//		char str[100];
				//		sprintf(str, "Tetrahedron %d contains grid point %d (%d, %d, %d)",
				//				tetID, idx, i, j, k);
				//		lcs::Error(str);
				//	}
				//}
			}

	printf("Passed\n");
}

////////////////////// This is a specific test for float blocked tracing kernel
inline float DeterminantThree(float *a) {
	// a[0] a[1] a[2]
	// a[3] a[4] a[5]
	// a[6] a[7] a[8]
	return a[0] * a[4] * a[8] + a[1] * a[5] * a[6] + a[2] * a[3] * a[7] -
		   a[0] * a[5] * a[7] - a[1] * a[3] * a[8] - a[2] * a[4] * a[6];
}

inline void CalculateNaturalCoordinates(float X, float Y, float Z,
					float *tetX, float *tetY, float *tetZ, float *coordinates) {
	X -= tetX[0];
	Y -= tetY[0];
	Z -= tetZ[0];

	float det[9] = {tetX[1] - tetX[0], tetY[1] - tetY[0], tetZ[1] - tetZ[0],
			 tetX[2] - tetX[0], tetY[2] - tetY[0], tetZ[2] - tetZ[0],
			 tetX[3] - tetX[0], tetY[3] - tetY[0], tetZ[3] - tetZ[0]};

	float V = 1 / DeterminantThree(det);

	float z41 = tetZ[3] - tetZ[0];
	float y34 = tetY[2] - tetY[3];
	float z34 = tetZ[2] - tetZ[3];
	float y41 = tetY[3] - tetY[0];
	float a11 = (z41 * y34 - z34 * y41) * V;

	float x41 = tetX[3] - tetX[0];
	float x34 = tetX[2] - tetX[3];
	float a12 = (x41 * z34 - x34 * z41) * V;

	float a13 = (y41 * x34 - y34 * x41) * V;

	coordinates[1] = a11 * X + a12 * Y + a13 * Z;

	float y12 = tetY[0] - tetY[1];
	float z12 = tetZ[0] - tetZ[1];
	float a21 = (z41 * y12 - z12 * y41) * V;

	float x12 = tetX[0] - tetX[1];
	float a22 = (x41 * z12 - x12 * z41) * V;

	float a23 = (y41 * x12 - y12 * x41) * V;

	coordinates[2] = a21 * X + a22 * Y + a23 * Z;

	float z23 = tetZ[1] - tetZ[2];
	float y23 = tetY[1] - tetY[2];
	float a31 = (z23 * y12 - z12 * y23) * V;

	float x23 = tetX[1] - tetX[2];
	float a32 = (x23 * z12 - x12 * z23) * V;

	float a33 = (y23 * x12 - y12 * x23) * V;

	coordinates[3] = a31 * X + a32 * Y + a33 * Z;

	coordinates[0] = 1 - coordinates[1] - coordinates[2] - coordinates[3];
}

inline int gFindCell(float *particle, int *connectivities, int *links,
		     float *vertexPositions,
		     float epsilon, int guess, float *coordinates) {
	float tetX[4], tetY[4], tetZ[4];
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
		if (index >= 0 && index <= 3)
			if (coordinates[index] >= -epsilon) break;
		guess = links[(guess << 2) | index];
		if (guess == -1) break;
	}
	return guess;
}

inline int FindCell(float *particle, /* local */int *connectivities, /*local*/ int *links, /*local*/ float *vertexPositions,
		    float epsilon, int guess, float *coordinates) {
	float tetX[4], tetY[4], tetZ[4];
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
/*
void BlockedTracing(float *globalVertexPositions,
		    float *globalStartVelocities,
		    float *globalEndVelocities,
		    int *globalTetrahedralConnectivities,
		    int *globalTetrahedralLinks,

		    int *startOffsetInCell,
		    int *startOffsetInPoint,

		    int *startOffsetInCellForBig,
		    int *startOffsetInPointForBig,
		    float *vertexPositionsForBig,
		    float *startVelocitiesForBig,
		    float *endVelocitiesForBig,

		    bool *canFitInSharedMemory,

		    int *blockedLocalConnectivities,
		    int *blockedLocalLinks,
		    int *blockedGlobalCellIDs,
		    int *blockedGlobalPointIDs,

		    int *activeBlockList, // Map active block ID to interesting block ID

		    int *stage,
		    float *lastPosition,
		    float *k1,
		    float *k2,
		    float *k3,
		    float *pastTimes,
		    int *startOffsetInParticle,
		    int *blockedActiveParticleIDList,
		    int *blockedCellLocationList,

		    void *sharedMemory,
							 
		    float startTime, float endTime, float timeStep,
		    float epsilon,
							 
		    int *squeezedStage,
		    float *squeezedLastPosition,
		    float *squeezedK1,
		    float *squeezedK2,
		    float *squeezedK3,
		    int *squeezedExitCells


							/// DEBUG ///
							//__global int *counts
							 ) {
	// Get work group ID, which is equal to active block ID
	int activeBlockID = get_group_id(0);
	
	// Get number of threads in a work group
	int numOfThreads = get_local_size(0);

	/// DEBUG ///
	//int numOfBlocks = get_global_size(0) / get_group_size(0);
	//if (numOfBlocks == 1545) return;

	// Get local thread ID
	int localID = get_local_id(0);

	// Get interesting block ID of the current active block ID
	int interestingBlockID = activeBlockList[activeBlockID];

	// Declare some arrays
	/*local float *vertexPositions;
	/*local float *startVelocities;
	/*local float *endVelocities;
	/*local int *connectivities;
	/*local int *links;

	float *gVertexPositions;
	float *gStartVelocities;
	float *gEndVelocities;
	int *gConnectivities;
	int *gLinks;

	bool canFit = canFitInSharedMemory[interestingBlockID];

	int startCell = startOffsetInCell[interestingBlockID];
	int startPoint = startOffsetInPoint[interestingBlockID];

	int numOfCells = startOffsetInCell[interestingBlockID + 1] - startCell;
	int numOfPoints = startOffsetInPoint[interestingBlockID + 1] - startPoint;

	int startCellForBig = startOffsetInCellForBig[interestingBlockID];
	int startPointForBig = startOffsetInPointForBig[interestingBlockID];

	/*if (canFit) { // This branch fills in the shared memory
		// Initialize vertexPositions, startVelocities and endVelocities
		vertexPositions = (float *)sharedMemory;
		startVelocities = vertexPositions + numOfPoints * 3;
		endVelocities = startVelocities + numOfPoints * 3;

		// Initialize connectivities and links
		connectivities = (int *)(endVelocities + numOfPoints * 3);
		links = connectivities + (numOfCells << 2);
	} else { // This branch fills in the global memory
		// Initialize vertexPositions, startVelocities and endVelocities
		gVertexPositions = vertexPositionsForBig + startPointForBig * 3;
		gStartVelocities = startVelocitiesForBig + startPointForBig * 3;
		gEndVelocities = endVelocitiesForBig + startPointForBig * 3;

		// Initialize connectivities and links
		gConnectivities = blockedLocalConnectivities + (startCell << 2);
		gLinks = blockedLocalLinks + (startCell << 2);
	}
/*
	for (int i = localID; i < numOfPoints * 3; i += numOfThreads) {
		int localPointID = i / 3;
		int dimensionID = i % 3;
		int globalPointID = blockedGlobalPointIDs[startPoint + localPointID];

		if (canFit) {
			vertexPositions[i] = globalVertexPositions[globalPointID * 3 + dimensionID];
			startVelocities[i] = globalStartVelocities[globalPointID * 3 + dimensionID];
			endVelocities[i] = globalEndVelocities[globalPointID * 3 + dimensionID];
		} else {
			//gVertexPositions[i] = gliobalVertexPositions[globalPointID * 3 + dimensionID];
			//gStartVelocities[i] = globalStartVelocities[globalPointID * 3 + dimensionID];
			//gEndVelocities[i] = globalEndVelocities[globalPointID * 3 + dimensionID];
		}
	}
*/
/*
	if (canFit)
		for (int i = localID; i < (numOfCells << 2); i += numOfThreads) {
			connectivities[i] = *(blockedLocalConnectivities + (startCell << 2) + i);
			links[i] = *(blockedLocalLinks + (startCell << 2) + i);
		}
*/
/*
	if (canFit)
		barrier(CLK_LOCAL_MEM_FENCE);
	else
		barrier(CLK_GLOBAL_MEM_FENCE);

	int numOfActiveParticles = startOffsetInParticle[activeBlockID + 1] - startOffsetInParticle[activeBlockID];

	for (int idx = localID; idx < numOfActiveParticles; idx += numOfThreads) {
		// activeParticleID here means the initial active particle ID
		int arrayIdx = startOffsetInParticle[activeBlockID] + idx;
		int activeParticleID = blockedActiveParticleIDList[arrayIdx];

		// Initialize the particle status
		int currStage = stage[activeParticleID];
		int currCell = blockedCellLocationList[startOffsetInParticle[activeBlockID] + idx];

		float currTime = pastTimes[activeParticleID];

		float currLastPosition[3];
		currLastPosition[0] = lastPosition[activeParticleID * 3];
		currLastPosition[1] = lastPosition[activeParticleID * 3 + 1];
		currLastPosition[2] = lastPosition[activeParticleID * 3 + 2];
		float currK1[3], currK2[3], currK3[3], currK4[3];
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

		/// DEBUG ///
		//continue;

		int cnt = 0;

		// At least one loop is executed.
		while (true) {

			/// DEBUG ///
			cnt++;
			//if (cnt == 2) break;

			float placeOfInterest[3];
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

			/// DEBUG ///
			//break;

			float coordinates[4];

			int nextCell;
			
			if (canFit)
				nextCell = FindCell(placeOfInterest, connectivities, links, vertexPositions, epsilon, currCell, coordinates);
			else /// DEBUG ///
				nextCell = gFindCell(placeOfInterest, gConnectivities, gLinks, gVertexPositions, epsilon, currCell, coordinates);

			/// DEBUG ///
			//break;

			if (nextCell == -1 || currTime >= endTime) {
				// Find the next cell globally
				int globalCellID = blockedGlobalCellIDs[startCell + currCell];
				int nextGlobalCell;
			
				if (nextCell != -1)
					nextGlobalCell = blockedGlobalCellIDs[startCell + nextCell];
				else
					nextGlobalCell = gFindCell(placeOfInterest, globalTetrahedralConnectivities, globalTetrahedralLinks,
											   globalVertexPositions, epsilon, globalCellID, coordinates);

				if (currTime >= endTime && nextGlobalCell != -1) nextGlobalCell = -2 - nextGlobalCell;

				pastTimes[activeParticleID] = currTime;

				stage[activeParticleID] = currStage;

				lastPosition[activeParticleID * 3] = currLastPosition[0];
				lastPosition[activeParticleID * 3 + 1] = currLastPosition[1];
				lastPosition[activeParticleID * 3 + 2] = currLastPosition[2];
		
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

				// Write squeezed arrays
				squeezedStage[arrayIdx] = currStage;
				squeezedExitCells[arrayIdx] = nextGlobalCell;

				squeezedLastPosition[arrayIdx * 3] = currLastPosition[0];
				squeezedLastPosition[arrayIdx * 3 + 1] = currLastPosition[1];
				squeezedLastPosition[arrayIdx * 3 + 2] = currLastPosition[2];
		
				if (currStage > 0) {
					squeezedK1[arrayIdx * 3] = currK1[0];
					squeezedK1[arrayIdx * 3 + 1] = currK1[1];
					squeezedK1[arrayIdx * 3 + 2] = currK1[2];
				}
				if (currStage > 1) {
					squeezedK2[arrayIdx * 3] = currK2[0];
					squeezedK2[arrayIdx * 3 + 1] = currK2[1];
					squeezedK2[arrayIdx * 3 + 2] = currK2[2];
				}
				if (currStage > 2) {
					squeezedK3[arrayIdx * 3] = currK3[0];
					squeezedK3[arrayIdx * 3 + 1] = currK3[1];
					squeezedK3[arrayIdx * 3 + 2] = currK3[2];
				}

				break;
			}

			currCell = nextCell;

			float alpha = (endTime - currTime) / (endTime - startTime);
			float beta = 1 - alpha;

			float vecX[4], vecY[4], vecZ[4];

			for (int i = 0; i < 4; i++)
				if (canFit) {
					int pointID = connectivities[(nextCell << 2) | i];
					vecX[i] = startVelocities[pointID * 3] * alpha + endVelocities[pointID * 3] * beta;
					vecY[i] = startVelocities[pointID * 3 + 1] * alpha + endVelocities[pointID * 3 + 1] * beta;
					vecZ[i] = startVelocities[pointID * 3 + 2] * alpha + endVelocities[pointID * 3 + 2] * beta;
				} else {
					int pointID = gConnectivities[(nextCell << 2) | i];
					vecX[i] = gStartVelocities[pointID * 3] * alpha + gEndVelocities[pointID * 3] * beta;
					vecY[i] = gStartVelocities[pointID * 3 + 1] * alpha + gEndVelocities[pointID * 3 + 1] * beta;
					vecZ[i] = gStartVelocities[pointID * 3 + 2] * alpha + gEndVelocities[pointID * 3 + 2] * beta;
				}

			float *currK;
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
}*/
