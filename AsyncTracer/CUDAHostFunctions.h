#ifndef __CUDA_HOST_FUNCTIONS_H
#define __CUDA_HOST_FUNCTIONS_H

#include "lcsUtility.h"

#define BLOCK_SIZE 256

void LaunchGPUForInitialCellLocation(double minX, double maxX, double minY, double maxY, double minZ, double maxZ,
									 int xRes, int yRes, int zRes,
									 int *&initialCellLocations,
									 int *&gridCounts,
									 int *&d_cellLocations,
									 int *&d_gridCounts,
									 int globalNumOfCells,
									 double *d_vertexPositions,
									 int *d_tetrahedralConnectivities,
									 double epsilon);

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
								 );

#endif
