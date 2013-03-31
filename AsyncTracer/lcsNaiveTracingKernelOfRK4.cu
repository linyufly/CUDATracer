/******************************************************************
File			:		lcsNaiveTracingKernelOfRK4.cu
Author			:		Mingcheng Chen
Last Update		:		January 2nd, 2013
*******************************************************************/

#include "device_launch_parameters.h"
#include "CUDAKernels.h"

#include "stdio.h"

__device__ inline double DeterminantThree(double *a) {
	// a[0] a[1] a[2]
	// a[3] a[4] a[5]
	// a[6] a[7] a[8]
	return a[0] * a[4] * a[8] + a[1] * a[5] * a[6] + a[2] * a[3] * a[7] -
		   a[0] * a[5] * a[7] - a[1] * a[3] * a[8] - a[2] * a[4] * a[6];
}

__device__ inline void CheckDoubleCode(double a) {
	long long *test = (long long *)&a;
	printf("code: %lld\n", *test);
}

/// DEBUG ///
__device__ inline void CheckSomeValues(double X, double Y, double Z,
										double *tetX, double *tetY, double *tetZ, double *coordinates) {
	X -= tetX[0];
	Y -= tetY[0];
	Z -= tetZ[0];

	double det[9] = {tetX[1] - tetX[0], tetY[1] - tetY[0], tetZ[1] - tetZ[0],
					 tetX[2] - tetX[0], tetY[2] - tetY[0], tetZ[2] - tetZ[0],
					 tetX[3] - tetX[0], tetY[3] - tetY[0], tetZ[3] - tetZ[0]};

	double V = 1 / DeterminantThree(det);

	double z41 = tetZ[3] - tetZ[0];
	double y34 = tetY[2] - tetY[3];
	double z34 = tetZ[2] - tetZ[3];
	double y41 = tetY[3] - tetY[0];
	double a11 = (z41 * y34 - z34 * y41) * V;

	double x41 = tetX[3] - tetX[0];
	double x34 = tetX[2] - tetX[3];
	double a12 = (x41 * z34 - x34 * z41) * V;

	double a13 = (y41 * x34 - y34 * x41) * V;

	coordinates[1] = a11 * X + a12 * Y + a13 * Z;

	/// DEBUG ///
	printf("X: %.20lf, Y: %.20lf, Z: %.20lf\n", X, Y, Z);
	printf("V: %.20lf, z41: %.20lf, y34: %.20lf, z34: %.20lf, y41: %.20lf, a11: %.20lf\n", V, z41, y34, z34, y41, a11);
	printf("x41: %.20lf, x34: %.20lf, a12: %.20lf, a13: %.20lf\n", x41, x34, a12, a13);
	printf("coordinates[1]: %.20lf\n", coordinates[1]);
	CheckDoubleCode(X);
	CheckDoubleCode(Y);
	CheckDoubleCode(Z);
	CheckDoubleCode(a11);
	CheckDoubleCode(a12);
	CheckDoubleCode(a13);
	CheckDoubleCode(coordinates[1]);
}


__device__ inline void CalculateNaturalCoordinates(double X, double Y, double Z,
										double *tetX, double *tetY, double *tetZ, double *coordinates) {
	X -= tetX[0];
	Y -= tetY[0];
	Z -= tetZ[0];

	double det[9] = {tetX[1] - tetX[0], tetY[1] - tetY[0], tetZ[1] - tetZ[0],
					 tetX[2] - tetX[0], tetY[2] - tetY[0], tetZ[2] - tetZ[0],
					 tetX[3] - tetX[0], tetY[3] - tetY[0], tetZ[3] - tetZ[0]};

	double V = 1 / DeterminantThree(det);

	double z41 = tetZ[3] - tetZ[0];
	double y34 = tetY[2] - tetY[3];
	double z34 = tetZ[2] - tetZ[3];
	double y41 = tetY[3] - tetY[0];
	double a11 = (z41 * y34 - z34 * y41) * V;

	double x41 = tetX[3] - tetX[0];
	double x34 = tetX[2] - tetX[3];
	double a12 = (x41 * z34 - x34 * z41) * V;

	double a13 = (y41 * x34 - y34 * x41) * V;

	coordinates[1] = a11 * X + a12 * Y + a13 * Z;

	double y12 = tetY[0] - tetY[1];
	double z12 = tetZ[0] - tetZ[1];
	double a21 = (z41 * y12 - z12 * y41) * V;

	double x12 = tetX[0] - tetX[1];
	double a22 = (x41 * z12 - x12 * z41) * V;

	double a23 = (y41 * x12 - y12 * x41) * V;

	coordinates[2] = a21 * X + a22 * Y + a23 * Z;

	double z23 = tetZ[1] - tetZ[2];
	double y23 = tetY[1] - tetY[2];
	double a31 = (z23 * y12 - z12 * y23) * V;

	double x23 = tetX[1] - tetX[2];
	double a32 = (x23 * z12 - x12 * z23) * V;

	double a33 = (y23 * x12 - y12 * x23) * V;

	coordinates[3] = a31 * X + a32 * Y + a33 * Z;

	coordinates[0] = 1 - coordinates[1] - coordinates[2] - coordinates[3];
}

///// DEBUG ///
//__device__ int FindCellForSpecialParticle(double *particle, int *connectivities, int *links, double *vertexPositions,
//					double epsilon, int guess, double *coordinates) {
//	double tetX[4], tetY[4], tetZ[4];
//	for (int i = 0; i < 4; i++) {
//		int pointID = connectivities[(guess << 2) | i];
//		tetX[i] = vertexPositions[pointID * 3];
//		tetY[i] = vertexPositions[pointID * 3 + 1];
//		tetZ[i] = vertexPositions[pointID * 3 + 2];
//	}
//	CheckSomeValues(particle[0], particle[1], particle[2], tetX, tetY, tetZ, coordinates);
//}

__device__ int FindCell(double *particle, int *connectivities, int *links, double *vertexPositions,
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

__global__ void NaiveTracing(double *globalVertexPositions,
							 double *globalStartVelocities,
							 double *globalEndVelocities,
							 int *globalTetrahedralConnectivities,
							 int *globalTetrahedralLinks,

							 //int *stage,
							 double *lastPosition,
/*
							 double *k1,
							 double *k2,
							 double *k3,
*/
							 double *pastTimes,

							 int *cellLocations,

							 double startTime, double endTime, double timeStep,
							 double epsilon,

							 int *activeParticles,
							 int numOfActiveParticles
							 ) {
	int globalID = blockIdx.x * blockDim.x + threadIdx.x;

	if (globalID >= numOfActiveParticles) return;

	int activeParticleID = activeParticles[globalID];
	
	// Initialize the particle status
	int currStage = 0; //stage[activeParticleID];
	int currCell = cellLocations[activeParticleID];

	double currTime = pastTimes[activeParticleID];
	double currLastPosition[3];
	currLastPosition[0] = lastPosition[activeParticleID * 3];
	currLastPosition[1] = lastPosition[activeParticleID * 3 + 1];
	currLastPosition[2] = lastPosition[activeParticleID * 3 + 2];
/*
	if (!activeParticleID) {
		printf("hahaha: %lf %lf %lf\n", currLastPosition[0], currLastPosition[1], currLastPosition[2]);
		printf("curr cell: %d\n", currCell);
	}
*/
	//double currK1[3], currK2[3], currK3[3], currK4[3];
	double currK[3], NX[3];
	for (int i = 0; i < 3; i++)
		NX[i] = currLastPosition[i];
/*
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
*/
	int cnt = 0;

	// At least one loop is executed.
	while (true) {
		/// DEBUG ///
		cnt++;

		double placeOfInterest[3];
		placeOfInterest[0] = currLastPosition[0];
		placeOfInterest[1] = currLastPosition[1];
		placeOfInterest[2] = currLastPosition[2];
		switch (currStage) {
			case 1: {
				placeOfInterest[0] += 0.5 * currK[0];//1
				placeOfInterest[1] += 0.5 * currK[1];
				placeOfInterest[2] += 0.5 * currK[2];
					} break;
			case 2: {
				placeOfInterest[0] += 0.5 * currK[0];//2
				placeOfInterest[1] += 0.5 * currK[1];
				placeOfInterest[2] += 0.5 * currK[2];
					} break;
			case 3: {
				placeOfInterest[0] += currK[0];//3
				placeOfInterest[1] += currK[1];
				placeOfInterest[2] += currK[2];
				} break;
		}

		double coordinates[4];

		int nextCell = FindCell(placeOfInterest, globalTetrahedralConnectivities, globalTetrahedralLinks, globalVertexPositions, epsilon, currCell, coordinates);

		///// DEBUG ///
		//if (currTime < 0.01 && activeParticleID == 1269494) {
		//	FindCellForSpecialParticle(placeOfInterest, globalTetrahedralConnectivities, globalTetrahedralLinks, globalVertexPositions, epsilon, nextCell, coordinates);
		//	for (int i = 0; i < 4; i++)
		//		printf(" %.20lf", coordinates[i]);
		//	printf("\n");
		//}

		if (nextCell == -1 || currTime >= endTime) {
			int nextGlobalCell = nextCell;

			//if (currTime >= endTime && nextGlobalCell != -1) nextGlobalCell = -2 - nextGlobalCell;

			pastTimes[activeParticleID] = currTime;

//			stage[activeParticleID] = currStage;
			cellLocations[activeParticleID] = nextGlobalCell;

			lastPosition[activeParticleID * 3] = currLastPosition[0];
			lastPosition[activeParticleID * 3 + 1] = currLastPosition[1];
			lastPosition[activeParticleID * 3 + 2] = currLastPosition[2];
			
/*		
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
*/
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
			int pointID = globalTetrahedralConnectivities[(nextCell << 2) | i];
			vecX[i] = globalStartVelocities[pointID * 3] * alpha + globalEndVelocities[pointID * 3] * beta;
			vecY[i] = globalStartVelocities[pointID * 3 + 1] * alpha + globalEndVelocities[pointID * 3 + 1] * beta;
			vecZ[i] = globalStartVelocities[pointID * 3 + 2] * alpha + globalEndVelocities[pointID * 3 + 2] * beta;
		}
/*
		double *currK;
		switch (currStage) {
			case 0: currK = currK1; break;
			case 1: currK = currK2; break;
			case 2: currK = currK3; break;
			case 3: currK = currK4; break;
		}
*/
		currK[0] = currK[1] = currK[2] = 0;

		for (int i = 0; i < 4; i++) {
			currK[0] += vecX[i] * coordinates[i];
			currK[1] += vecY[i] * coordinates[i];
			currK[2] += vecZ[i] * coordinates[i];
		}
/*
		if (!activeParticleID) {
			printf("currK: %lf %lf %lf\n", currK[0], currK[1], currK[2]);
		}
*/
		currK[0] *= timeStep;
		currK[1] *= timeStep;
		currK[2] *= timeStep;

		double coefficient = currStage == 1 || currStage == 2 ? (1.0 / 3) : (1.0 / 6);

		
		for (int i = 0; i < 3; i++)
			NX[i] += currK[i] * coefficient;
/*
		if (!activeParticleID) {
			printf("coefficient: %lf\n", coefficient);
			printf("timeStep: %lf\n", timeStep);
			printf("currK: %lf %lf %lf\n", currK[0], currK[1], currK[2]);
			printf("NX: %lf %lf %lf\n", NX[0], NX[1], NX[2]);
		}
*/
		if (currStage == 3) {
			/// DEBUG ///
			//double oldTime = currTime;

			currTime += timeStep;
			for (int i = 0; i < 3; i++)
				currLastPosition[i] = NX[i];
			//for (int i = 0; i < 3; i++)
			//	currLastPosition[i] += (currK1[i] + currK2[i] * 2 + currK3[i] * 2 + currK4[i]) / 6;

			currStage = 0;

			///// DEBUG ///
			//if (oldTime < 0.319 && activeParticleID == 1269494) {
			//	printf("time step: %.20lf\n", oldTime);
			//	printf("k1: %.20lf %.20lf %.20lf\n", currK1[0], currK1[1], currK1[2]);
			//	printf("k2: %.20lf %.20lf %.20lf\n", currK2[0], currK2[1], currK2[2]);
			//	printf("k3: %.20lf %.20lf %.20lf\n", currK3[0], currK3[1], currK3[2]);
			//	printf("currPosi: %.20lf %.20lf %.20lf\n", currLastPosition[0], currLastPosition[1], currLastPosition[2]);
			//	printf("Special code of k1.x: ");
			//	long long *test = (long long *)currK1;
			//	printf("%lld\n", *test);
			//	printf("\n");
			//}
		} else
			currStage++;
	}
}	
