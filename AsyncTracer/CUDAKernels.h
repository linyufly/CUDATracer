#ifndef __CUDA_KERNELS_H
#define __CUDA_KERNELS_H

__global__ void InitialCellLocation(double *vertexPositions,
								    int *tetrahedralConnectivities,
									int *cellLocations,
									int xRes, int yRes, int zRes,
									double minX, double minY, double minZ,
									double dx, double dy, double dz,
									double epsilon,
									int numOfCells,

									int *gridCounts
									);

__global__ void NaiveTracing(double *globalVertexPositions,
							 double *globalStartVelocities,
							 double *globalEndVelocities,
							 int *globalTetrahedralConnectivities,
							 int *globalTetrahedralLinks,

							 //int *stage,
							 double *lastPosition,
							 //double *k1,
							 //double *k2,
							 //double *k3,
							 double *pastTimes,

							 int *cellLocations,

							 double startTime, double endTime, double timeStep,
							 double epsilon,

							 int *activeParticles,
							 int numOfActiveParticles
							 );
#endif
