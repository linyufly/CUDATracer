/**********************************************
File			:	main.cpp
Author			:	Mingcheng Chen
Last Update		:	November 8th, 2012
***********************************************/

#include "lcs.h"
#include "lcsUtility.h"
#include "lcsUnitTest.h"
#include "lcsGeometry.h"

#include <boost/unordered_map.hpp>
#include "cuda_runtime.h"

#include <ctime>
#include <string>
#include <algorithm>
#include <set>

#include "CUDAHostFunctions.h"

//const char *configurationFile = "RungeKutta4.conf";
//const char *configurationFile = "RungeKutta4ForTCPC.conf";
const char *configurationFile = "RungeKutta4ForUpperVasc.conf";
const char *tetrahedronBlockIntersectionKernel = "lcsTetrahedronBlockIntersectionKernel.cl";
const char *initialCellLocationKernel = "lcsInitialCellLocationKernel.cl";
const char *bigBlockInitializationKernel = "lcsBigBlockInitializationKernel.cl";
const char *blockedTracingKernelPrefix = "lcsBlockedTracingKernelOf";
const char *blockedTracingKernelSuffix = ".cl";
const char *lastPositionFile = "lcsLastPositions.txt";

lcs::Configure *configure;

lcs::Frame **frames;
int numOfFrames;

int *tetrahedralConnectivities, *tetrahedralLinks;
double *vertexPositions;
int globalNumOfCells, globalNumOfPoints;
double globalMinX, globalMaxX, globalMinY, globalMaxY, globalMinZ, globalMaxZ;

double blockSize;
int numOfBlocksInX, numOfBlocksInY, numOfBlocksInZ;

// For tetrahedron-block intersection
int *xLeftBound, *xRightBound, *yLeftBound, *yRightBound, *zLeftBound, *zRightBound;
int numOfQueries;
int *queryTetrahedron, *queryBlock;
bool *queryResults; // Whether certain tetrahedron intersects with certain block

// For blocks
boost::unordered_map<int, int> interestingBlockMap;
boost::unordered_map<lcs::BlockTetrahedronPair, int, lcs::HashForBlockTetrahedronPair> localCellIDMap;
int numOfBlocks, numOfInterestingBlocks, numOfBigBlocks;
lcs::BlockRecord **blocks;
bool *canFitInSharedMemory;
int *startOffsetInCell, *startOffsetInPoint;
int *startOffsetInCellForBig, *startOffsetInPointForBig;

// For initial cell location
int *initialCellLocations;

// For tracing
lcs::ParticleRecord **particleRecords;
int *exitCells;
int numOfInitialActiveParticles;

// CUDA C variables
cudaError_t err;
int *d_tetrahedralConnectivities, *d_tetrahedralLinks;
double *d_vertexPositions;
int *d_queryTetrahedron, *d_queryBlock;
bool *d_queryResults;

int *d_bigBlocks;
int *d_startOffsetInCellForBig, *d_startOffsetInPointForBig;
double *d_vertexPositionsForBig, *d_startVelocitiesForBig, *d_endVelocitiesForBig;

int *d_startOffsetInCell, *d_startOffsetInPoint;
int *d_localConnectivities, *d_localLinks;
int *d_globalCellIDs, *d_globalPointIDs;

bool *d_canFitInSharedMemory;

int *d_cellLocations;
int *d_gridCounts;

int *d_squeezedStage;
int *d_squeezedExitCells;
double *d_squeezedLastPositionForRK4;
double *d_squeezedK1ForRK4, *d_squeezedK2ForRK4, *d_squeezedK3ForRK4;

int *d_stage;
double *d_pastTimes;
double *d_lastPositionForRK4;
double *d_k1ForRK4, *d_k2ForRK4, *d_k3ForRK4;

double *d_velocities[2];

int *d_activeBlockList;
int *d_startOffsetInParticles;

int *d_blockedActiveParticles;
int *d_blockedCellLocations;

int *d_activeParticles;
int *d_exitCells;

void SystemTest() {
	printf("sizeof(double) = %d\n", sizeof(double));
	printf("sizeof(float) = %d\n", sizeof(float));
	printf("sizeof(int) = %d\n", sizeof(int));
	printf("sizeof(int *) = %d\n", sizeof(int *));
	printf("sizeof(char) = %d\n", sizeof(char));

	//printf("sizeof(cl_float) = %d\n", sizeof(cl_float));
	//printf("sizeof(cl_double) = %d\n", sizeof(cl_double));
	//printf("sizeof(cl_mem) = %d\n", sizeof(cl_mem));

	printf("\n");
}

void ReadConfFile() {
	configure = new lcs::Configure(configurationFile);
	if (configure->GetIntegration() == "FE") lcs::ParticleRecord::SetDataType(lcs::ParticleRecord::FE);
	if (configure->GetIntegration() == "RK4") lcs::ParticleRecord::SetDataType(lcs::ParticleRecord::RK4);
	if (configure->GetIntegration() == "RK45") lcs::ParticleRecord::SetDataType(lcs::ParticleRecord::RK45);
	printf("\n");
}

void LoadFrames() {
	numOfFrames = configure->GetNumOfFrames();
	frames = new lcs::Frame *[numOfFrames];

	for (int i = 0; i < numOfFrames; i++) {
		double timePoint = configure->GetTimePoints()[i];
		std::string veloFileName = configure->GetDataFilePrefix() + configure->GetDataFileIndices()[i] + "." + configure->GetDataFileSuffix();
		printf("Loading frame %d (file = %s) ... ", i, veloFileName.c_str());
		frames[i] = new lcs::Frame(timePoint, "./UpperVasc/geometry.txt", veloFileName.c_str());
		printf("Done.\n");

		if (i) frames[i]->GetTetrahedralGrid()->CleanAllButVelocities();

	}
	printf("\n");
}

void GetTopologyAndGeometry() {
	globalNumOfCells = frames[0]->GetTetrahedralGrid()->GetNumOfCells();
	globalNumOfPoints = frames[0]->GetTetrahedralGrid()->GetNumOfVertices();

	tetrahedralConnectivities = new int [globalNumOfCells * 4];
	tetrahedralLinks = new int [globalNumOfCells * 4];

	vertexPositions = new double [globalNumOfPoints * 3];

	frames[0]->GetTetrahedralGrid()->ReadConnectivities(tetrahedralConnectivities);
	frames[0]->GetTetrahedralGrid()->ReadLinks(tetrahedralLinks);

	if (configure->UseDouble())
		frames[0]->GetTetrahedralGrid()->ReadPositions((double *)vertexPositions);
	else
		frames[0]->GetTetrahedralGrid()->ReadPositions((float *)vertexPositions);
}

void GetGlobalBoundingBox() {
	lcs::Vector firstPoint = frames[0]->GetTetrahedralGrid()->GetVertex(0);

	globalMaxX = globalMinX = firstPoint.GetX();
	globalMaxY = globalMinY = firstPoint.GetY();
	globalMaxZ = globalMinZ = firstPoint.GetZ();

	for (int i = 1; i < globalNumOfPoints; i++) {
		lcs::Vector point = frames[0]->GetTetrahedralGrid()->GetVertex(i);

		globalMaxX = std::max(globalMaxX, point.GetX());
		globalMinX = std::min(globalMinX, point.GetX());
		
		globalMaxY = std::max(globalMaxY, point.GetY());
		globalMinY = std::min(globalMinY, point.GetY());

		globalMaxZ = std::max(globalMaxZ, point.GetZ());
		globalMinZ = std::min(globalMinZ, point.GetZ());
	}

	printf("Global Bounding Box\n");
	printf("X: [%lf, %lf], length = %lf\n", globalMinX, globalMaxX, globalMaxX - globalMinX);
	printf("Y: [%lf, %lf], length = %lf\n", globalMinY, globalMaxY, globalMaxY - globalMinY);
	printf("Z: [%lf, %lf], length = %lf\n", globalMinZ, globalMaxZ, globalMaxZ - globalMinZ);
	printf("\n");
}

void CalculateNumOfBlocksInXYZ() {
	blockSize = configure->GetBlockSize();

	numOfBlocksInX = (int)((globalMaxX - globalMinX) / blockSize) + 1;
	numOfBlocksInY = (int)((globalMaxY - globalMinY) / blockSize) + 1;
	numOfBlocksInZ = (int)((globalMaxZ - globalMinZ) / blockSize) + 1;
}

void InitialCellLocation() {
	printf("Start to use GPU to process initial cell location ...\n");
	printf("\n");

	int startTime = clock();

	double minX = configure->GetBoundingBoxMinX();
	double maxX = configure->GetBoundingBoxMaxX();
	double minY = configure->GetBoundingBoxMinY();
	double maxY = configure->GetBoundingBoxMaxY();
	double minZ = configure->GetBoundingBoxMinZ();
	double maxZ = configure->GetBoundingBoxMaxZ();

	int xRes = configure->GetBoundingBoxXRes();
	int yRes = configure->GetBoundingBoxYRes();
	int zRes = configure->GetBoundingBoxZRes();

	double dx = (maxX - minX) / xRes;
	double dy = (maxY - minY) / yRes;
	double dz = (maxZ - minZ) / zRes;

	int *gridCounts;

	/// DEBUG ///
	//printf("ASDFASDFASDFA   epsi = %lf\n", configure->GetEpsilon());
	
	LaunchGPUForInitialCellLocation(minX, maxX, minY, maxY, minZ, maxZ,
									xRes, yRes, zRes,
									initialCellLocations,
									gridCounts,
									d_cellLocations,
									d_gridCounts,
									globalNumOfCells,
									d_vertexPositions,
									d_tetrahedralConnectivities,
									configure->GetEpsilon());

	/// DEBUG ///
	FILE *locationFile = fopen("lcsInitialLocations.txt", "w");
	int numOfGridPoints = (xRes + 1) * (yRes + 1) * (zRes + 1);
	for (int i = 0; i < numOfGridPoints; i++)
		if (initialCellLocations[i] != -1) {
			int Z = i % (zRes + 1);
			int temp = i / (zRes + 1);
			int Y = temp % (yRes + 1);
			int X = temp / (yRes + 1);
			fprintf(locationFile, "%lf %lf %lf %d %d\n", minX + X * dx, minY + Y * dy, minZ + Z * dz, i, initialCellLocations[i]);
		}
	fclose(locationFile);

	locationFile = fopen("lcsGridCounts.txt", "w");
	for (int i = 0; i < numOfGridPoints; i++)
		fprintf(locationFile, "%d\n", gridCounts[i]);
	fclose(locationFile);

	int endTime = clock();

	printf("First 10 results: ");
	for (int i = 0; i < 10; i++) {
		if (i) printf(" ");
		printf("%d", initialCellLocations[i]);
	}
	printf("\n\n");

	printf("The GPU Kernel for initial cell locations cost %lf sec.\n", (endTime - startTime) * 1.0 / CLOCKS_PER_SEC);
	printf("\n");

	// Unit Test for Initial Cell Location Kernel
	startTime = clock();

	if (configure->UseUnitTestForInitialCellLocation()) {
		lcs::UnitTestForInitialCellLocations(frames[0]->GetTetrahedralGrid(),
											 xRes, yRes, zRes,
											 minX, minY, minZ,
											 dx, dy, dz,
											 initialCellLocations,
											 configure->GetEpsilon());
		printf("\n");
	}

	endTime = clock();

	printf("The unit test cost %lf sec.\n", (endTime - startTime) * 1.0 / CLOCKS_PER_SEC);
	printf("\n");
}

void InitializeParticleRecordsInDevice() {
	// Initialize squeezed exitCells
	err = cudaMalloc((void **)&d_squeezedExitCells, sizeof(int) * numOfInitialActiveParticles);
	if (err) lcs::Error("Fail to create a device squeezedExitCells");

	// Initialize squeezed stage
	err = cudaMalloc((void **)&d_squeezedStage, sizeof(int) * numOfInitialActiveParticles);
	if (err) lcs::Error("Fail to create a device squeezedStage");

	// Initialize d_stage
	err = cudaMalloc((void **)&d_stage, sizeof(int) * numOfInitialActiveParticles);
	if (err) lcs::Error("Fail to create a buffer for device stage");

	err = cudaMemset(d_stage, 0, sizeof(int) * numOfInitialActiveParticles);
	if (err) lcs::Error("Fail to initialize d_stage");

	// Initialize d_pastTimes
	err = cudaMalloc((void **)&d_pastTimes, sizeof(double) * numOfInitialActiveParticles);
	if (err) lcs::Error("Fail to create a buffer for device pastTimes");

	err = cudaMemset(d_pastTimes, 0, sizeof(double) * numOfInitialActiveParticles);
	if (err) lcs::Error("Fail to enqueue initialize d_pastTimes");

	// Initialize some integration-specific device arrays
	switch (lcs::ParticleRecord::GetDataType()) {
	case lcs::ParticleRecord::RK4: {
		// Initialize squeezed arrays for RK4
		err = cudaMalloc((void **)&d_squeezedLastPositionForRK4, sizeof(double) * 3 * numOfInitialActiveParticles);
		if (err) lcs::Error("Fail to create a device squeezedLastPositionForRK4");

		err = cudaMalloc((void **)&d_squeezedK1ForRK4, sizeof(double) * 3 * numOfInitialActiveParticles);
		if (err) lcs::Error("Fail to create a device squeezedK1ForRK4");
		
		err = cudaMalloc((void **)&d_squeezedK2ForRK4, sizeof(double) * 3 * numOfInitialActiveParticles);
		if (err) lcs::Error("Fail to create a device squeezedK2ForRK4");

		err = cudaMalloc((void **)&d_squeezedK3ForRK4, sizeof(double) * 3 * numOfInitialActiveParticles);
		if (err) lcs::Error("Fail to create a device squeezedK3ForRK4");

		
		// Initialize d_lastPositionForRK4
		double *lastPosition;
		lastPosition = new double [numOfInitialActiveParticles * 3];

		for (int i = 0; i < numOfInitialActiveParticles; i++) {
			lcs::ParticleRecordDataForRK4 *data = (lcs::ParticleRecordDataForRK4 *)particleRecords[i]->GetData();
			lcs::Vector point = data->GetLastPosition();
			double x = point.GetX();
			double y = point.GetY();
			double z = point.GetZ();
			lastPosition[i * 3] = x;
			lastPosition[i * 3 + 1] = y;
			lastPosition[i * 3 + 2] = z;
		}

		err = cudaMalloc((void **)&d_lastPositionForRK4, sizeof(double) * 3 * numOfInitialActiveParticles);
		if (err) lcs::Error("Fail to create a buffer for device lastPosition for RK4");

		err = cudaMemcpy(d_lastPositionForRK4, lastPosition, sizeof(double) * 3 * numOfInitialActiveParticles, cudaMemcpyHostToDevice);
		if (err) lcs::Error("Fail to initialize d_lastPositionForRK4");

		// Initialize d_k1ForRK4
		err = cudaMalloc((void **)&d_k1ForRK4, sizeof(double) * 3 * numOfInitialActiveParticles);
		if (err) lcs::Error("Fail to create a buffer for device k1 for RK4");

		// Initialize d_k2ForRK4
		err = cudaMalloc((void **)&d_k2ForRK4, sizeof(double) * 3 * numOfInitialActiveParticles);
		if (err) lcs::Error("Fail to create a buffer for device k2 for RK4");

		// Initialize d_k3ForRK4
		err = cudaMalloc((void **)&d_k3ForRK4, sizeof(double) * 3 * numOfInitialActiveParticles);
		if (err) lcs::Error("Fail to create a buffer for device k3 for RK4");
								   } break;
	}
}

/// DEBUG ///
double kernelSum;


void InitializeInitialActiveParticles() {
	// Initialize particleRecord
	double minX = configure->GetBoundingBoxMinX();
	double maxX = configure->GetBoundingBoxMaxX();
	double minY = configure->GetBoundingBoxMinY();
	double maxY = configure->GetBoundingBoxMaxY();
	double minZ = configure->GetBoundingBoxMinZ();
	double maxZ = configure->GetBoundingBoxMaxZ();

	int xRes = configure->GetBoundingBoxXRes();
	int yRes = configure->GetBoundingBoxYRes();
	int zRes = configure->GetBoundingBoxZRes();

	double dx = (maxX - minX) / xRes;
	double dy = (maxY - minY) / yRes;
	double dz = (maxZ - minZ) / zRes;

	int numOfGridPoints = (xRes + 1) * (yRes + 1) * (zRes + 1);

	// Get numOfInitialActiveParticles
	numOfInitialActiveParticles = 0;
	for (int i = 0; i < numOfGridPoints; i++)
		if (initialCellLocations[i] != -1) numOfInitialActiveParticles++;

	if (!numOfInitialActiveParticles)
		lcs::Error("There is no initial active particle for tracing.");

	// Initialize particleRecords
	particleRecords = new lcs::ParticleRecord * [numOfInitialActiveParticles];

	int idx = -1, activeIdx = -1;
	for (int i = 0; i <= xRes; i++)
		for (int j = 0; j <= yRes; j++)
			for (int k = 0; k <= zRes; k++) {
				idx++;

				if (initialCellLocations[idx] == -1) continue;

				activeIdx++;

				switch (lcs::ParticleRecord::GetDataType()) {
				case lcs::ParticleRecord::RK4: {
					lcs::ParticleRecordDataForRK4 *data = new lcs::ParticleRecordDataForRK4();
					data->SetLastPosition(lcs::Vector(minX + i * dx, minY + j * dy, minZ + k * dz));
					particleRecords[activeIdx] = new lcs::ParticleRecord(lcs::ParticleRecordDataForRK4::COMPUTING_K1, idx, data);
											   } break;
				}
			}

	// Initialize exitCells
	exitCells = new int [numOfInitialActiveParticles];
	for (int i = 0; i < numOfInitialActiveParticles; i++)
		exitCells[i] = initialCellLocations[particleRecords[i]->GetGridPointID()];

	// Initialize particle records in device
	InitializeParticleRecordsInDevice();
}

void InitializeVelocityData(double **velocities) {
	// Initialize velocity data
	for (int i = 0; i < 2; i++)
		velocities[i] = new double [globalNumOfPoints * 3];

	// Read velocities[0]
	frames[0]->GetTetrahedralGrid()->ReadVelocities(velocities[0]);

	// Create d_velocities[2]
	for (int i = 0; i < 2; i++) {
		err = cudaMalloc((void **)&d_velocities[i], sizeof(double) * 3 * globalNumOfPoints);
		if (err) lcs::Error("Fail to create buffers for d_velocities[2]");
	}

	// Initialize d_velocities[0]
	err = cudaMemcpy(d_velocities[0], velocities[0], sizeof(double) * 3 * globalNumOfPoints, cudaMemcpyHostToDevice);
	if (err) lcs::Error("Fail to enqueue copy for d_velocities[0]");
}

void LoadVelocities(double *velocities, double *d_velocities, int frameIdx) {
	// Read velocities
	frames[frameIdx]->GetTetrahedralGrid()->ReadVelocities(velocities);

	// Write for d_velocities[frameIdx]
	err = cudaMemcpy(d_velocities, velocities, sizeof(double) * 3 * globalNumOfPoints, cudaMemcpyHostToDevice);
	if (err) lcs::Error("Fail to enqueue copy for d_velocities");
}

void UpdateActiveParticleDataForRK4(int *activeParticles, double *lastPositions, double *k1, double *k2, double *k3, int *squeezedStages, int *squeezedExitCells, int numOfActiveParticles) {
	int i;
	for (int i = 0; i < numOfActiveParticles; i++) {
		int particleID = activeParticles[i];

		exitCells[particleID] = squeezedExitCells[i];
		if (exitCells[particleID] == -1) continue;

		particleRecords[particleID]->SetStage(squeezedStages[i]);

		((lcs::ParticleRecordDataForRK4 *)particleRecords[particleID]->GetData())->SetLastPosition(((double *)lastPositions) + i * 3);
		((lcs::ParticleRecordDataForRK4 *)particleRecords[particleID]->GetData())->SetK1(((double *)k1) + i * 3);
		((lcs::ParticleRecordDataForRK4 *)particleRecords[particleID]->GetData())->SetK2(((double *)k2) + i * 3);
		((lcs::ParticleRecordDataForRK4 *)particleRecords[particleID]->GetData())->SetK3(((double *)k3) + i * 3);
	}
}

void UpdateSqueezedArraysForRK4(double *squeezedLastPositionForRK4, double *squeezedK1ForRK4, double *squeezedK2ForRK4, double *squeezedK3ForRK4,
								int *squeezedStage, int *squeezedExitCells, int numOfActiveParticles) {
	err = cudaMemcpy(squeezedExitCells, d_squeezedExitCells, sizeof(int) * numOfActiveParticles, cudaMemcpyDeviceToHost);

	/// DEBUG ///
	printf("err = %d\n", err);
	if (err) lcs::Error("Fail to read d_squeezedExitCells");

	err = cudaMemcpy(squeezedStage, d_squeezedStage, sizeof(int) * numOfActiveParticles, cudaMemcpyDeviceToHost);
	if (err) lcs::Error("Fail to read d_squeezedStage");

	err = cudaMemcpy(squeezedLastPositionForRK4, d_squeezedLastPositionForRK4, sizeof(double) * 3 * numOfActiveParticles, cudaMemcpyDeviceToHost);
	if (err) lcs::Error("Fail to read d_squeezedLastPositionForRK4");

	err = cudaMemcpy(squeezedK1ForRK4, d_squeezedK1ForRK4, sizeof(double) * 3 * numOfActiveParticles, cudaMemcpyDeviceToHost);
	if (err) lcs::Error("Fail to read d_squeezedK1ForRK4");

	err = cudaMemcpy(squeezedK2ForRK4, d_squeezedK2ForRK4, sizeof(double) * 3 * numOfActiveParticles, cudaMemcpyDeviceToHost);
	if (err) lcs::Error("Fail to read d_squeezedK2ForRK4");

	err = cudaMemcpy(squeezedK3ForRK4, d_squeezedK3ForRK4, sizeof(double) * 3 * numOfActiveParticles, cudaMemcpyDeviceToHost);
	if (err) lcs::Error("Fail to read d_squeezedK3ForRK4");
}

void InitializationForNaiveTracing() {
	err = cudaMalloc((void **)&d_vertexPositions, sizeof(double) * globalNumOfPoints * 3);
	if (err) lcs::Error("Fail to create d_vertexPositions");
	err = cudaMemcpy(d_vertexPositions, vertexPositions, sizeof(double) * globalNumOfPoints * 3, cudaMemcpyHostToDevice);
	if (err) lcs::Error("Fail to initialize d_vertexPositions");

	err = cudaMalloc((void **)&d_tetrahedralConnectivities, sizeof(int) * globalNumOfCells * 4);
	if (err) lcs::Error("Fail to create d_tetrahedralConnectivities");
	err = cudaMemcpy(d_tetrahedralConnectivities, tetrahedralConnectivities, sizeof(int) * globalNumOfCells * 4, cudaMemcpyHostToDevice);
	if (err) lcs::Error("Fail to initialize d_tetrahedralConnectivities");
}

void Tracing() {
	// Initialize d_tetrahedralLinks
	err = cudaMalloc((void **)&d_tetrahedralLinks, sizeof(int) * globalNumOfCells * 4);
	if (err) lcs::Error("Fail to create a buffer for device tetrahedralLinks");

	err = cudaMemcpy(d_tetrahedralLinks, tetrahedralLinks, sizeof(int) * globalNumOfCells * 4, cudaMemcpyHostToDevice);
	if (err) lcs::Error("Fail to enqueue copy for d_tetrahedralLinks");
	
	// Initialize initial active particle data
	InitializeInitialActiveParticles();
	
	err = cudaMalloc((void **)&d_activeParticles, sizeof(int) * numOfInitialActiveParticles);
	if (err) lcs::Error("Fail to create d_activeParticles");

	err = cudaMalloc((void **)&d_exitCells, sizeof(int) * numOfInitialActiveParticles);
	if (err) lcs::Error("Fail to create d_exitCells");
	err = cudaMemcpy(d_exitCells, exitCells, sizeof(int) * numOfInitialActiveParticles, cudaMemcpyHostToDevice);
	if (err) lcs::Error("Fail to initialize d_exitCells");

	// Initialize velocity data
	double *velocities[2];
	int currStartVIndex = 1;
	InitializeVelocityData(velocities);

	// Create some dynamic device arrays
	err = cudaMalloc((void **)&d_activeBlockList, sizeof(int) * numOfInterestingBlocks);
	if (err) lcs::Error("Fail to create a buffer for device activeBlockList");

	err = cudaMalloc((void **)&d_startOffsetInParticles, sizeof(int) * (numOfInterestingBlocks + 1));
	if (err) lcs::Error("Fail to create a buffer for device startOffsetInParticles");

	err = cudaMalloc((void **)&d_blockedActiveParticles, sizeof(int) * numOfInitialActiveParticles);
	if (err) lcs::Error("Fail to create a buffer for device blockedAciveParticles");

	err = cudaMalloc((void **)&d_blockedCellLocations, sizeof(int) * numOfInitialActiveParticles);
	if (err) lcs::Error("Fail to create a buffer for device blockedCellLocations");

	// Initialize activeParticles
	int *activeParticles = new int [numOfInitialActiveParticles];
	for (int i = 0; i < numOfInitialActiveParticles; i++)
		activeParticles[i] = i;

	// Create cellLocations
	int *cellLocations = new int [numOfInitialActiveParticles];

	// Create blockLocations
	int *blockLocations = new int [numOfInitialActiveParticles];

	// Create activeBlockIDs and activeBlockList
	int *activeBlockIDs = new int [numOfInterestingBlocks]; // An interesting block has the ID if and only if it has particles
	int *activeBlockIDList = new int [numOfInterestingBlocks];

	// Create startOffsetInParticle	
	int *startOffsetInParticle = new int [numOfInterestingBlocks + 1];

	// Create blockedActiveParticleIDList and topOfActiveBlocks
	int *blockedActiveParticleIDList = new int [numOfInitialActiveParticles];
	int *topOfActiveBlocks = new int [numOfInterestingBlocks];

	// Create blockedCellLocations
	int *blockedCellLocations = new int [numOfInitialActiveParticles];

	// Create work arrays
	int *countParticlesInInterestingBlocks = new int [numOfInterestingBlocks];
	int *marks = new int [numOfInterestingBlocks];
	memset(marks, 255, sizeof(int) * numOfInterestingBlocks);
	int markCount = 0;

	// Some start setting
	double currTime = 0;
	double interval = configure->GetTimeInterval();

	// Create general squeezed arrays
	int *squeezedStage = new int [numOfInitialActiveParticles];
	int *squeezedExitCells = new int [numOfInitialActiveParticles];

	// Create RK4-specific squeezed arrays
	double *squeezedLastPositionForRK4, *squeezedK1ForRK4, *squeezedK2ForRK4, *squeezedK3ForRK4;

	switch (lcs::ParticleRecord::GetDataType()) {
	case lcs::ParticleRecord::RK4: {
		squeezedLastPositionForRK4 = new double [numOfInitialActiveParticles * 3];
		squeezedK1ForRK4 = new double [numOfInitialActiveParticles * 3];
		squeezedK2ForRK4 = new double [numOfInitialActiveParticles * 3];
		squeezedK3ForRK4 = new double [numOfInitialActiveParticles * 3];
								   }break;
	}

	// Main loop for blocked tracing
	int startTime = clock();

	/// DEBUG ///
	kernelSum = 0;
	int numOfKernelCalls = 0;

	/// DEBUG ///
	FILE *tracer = fopen("tracer.txt", "w");

	for (int frameIdx = 0; frameIdx + 1 < numOfFrames; frameIdx++, currTime += interval) {
		printf("*********Tracing between frame %d and frame %d*********\n", frameIdx, frameIdx + 1);
		printf("\n");

		/// DEBUG ///
		int startTime;
		startTime = clock();

		currStartVIndex = 1 - currStartVIndex;

		int lastNumOfActiveParticles = 0;
		for (int i = 0; i < numOfInitialActiveParticles; i++) {
			if (exitCells[i] < -1) exitCells[i] = -(exitCells[i] + 2);
			if (exitCells[i] != -1) activeParticles[lastNumOfActiveParticles++] = i;
		}

		/// DEBUG ///
		printf("lastNumOfActiveParticles = %d\n", lastNumOfActiveParticles);
		//std::random_shuffle(activeParticles, activeParticles + lastNumOfActiveParticles);

		err = cudaMemcpy(d_activeParticles, activeParticles, sizeof(int) * lastNumOfActiveParticles, cudaMemcpyHostToDevice);
		if (err) lcs::Error("Fail to initialize d_activeParticles");

		// Load end velocities
		LoadVelocities(velocities[1 - currStartVIndex], d_velocities[1 - currStartVIndex], frameIdx + 1);

		// Naive tracing
		int kernelStart = clock();

		LaunchGPUForNaiveTracing(d_vertexPositions,
								 d_velocities[currStartVIndex],
								 d_velocities[1 - currStartVIndex],
								 d_tetrahedralConnectivities,
								 d_tetrahedralLinks,

								 d_stage,
								 d_lastPositionForRK4,
								 d_k1ForRK4,
								 d_k2ForRK4,
								 d_k3ForRK4,
								 d_pastTimes,

								 d_exitCells,

								 currTime, currTime + interval, configure->GetTimeStep(),
								 configure->GetEpsilon(),

								 d_activeParticles,
								 lastNumOfActiveParticles
								 );

		int kernelEnd = clock();
		kernelSum += (kernelEnd - kernelStart) * 1.0 / CLOCKS_PER_SEC;
		numOfKernelCalls++;

		cudaMemcpy(exitCells, d_exitCells, sizeof(int) * numOfInitialActiveParticles, cudaMemcpyDeviceToHost);

		int endTime = clock();
		printf("This interval cost %lf sec.\n", (double)(endTime - startTime) / CLOCKS_PER_SEC);
		printf("\n");
	}

	fclose(tracer);

	/// DEBUG ///
	printf("kernelSum = %lf\n", kernelSum);
	printf("numOfKernelCalls = %d\n", numOfKernelCalls);

	/// DEBUG ///
	int endTime = clock();
	printf("The total tracing time is %lf sec.\n", (double)(endTime - startTime) / CLOCKS_PER_SEC);
	printf("\n");

	// Release work arrays
	delete [] activeParticles;
	delete [] cellLocations;
	delete [] blockLocations;
	delete [] countParticlesInInterestingBlocks;
	delete [] marks;
	delete [] activeBlockIDs;
	delete [] activeBlockIDList;
	delete [] startOffsetInParticle;
}

void GetFinalPositions() {
	void *finalPositions;

	finalPositions = new double [numOfInitialActiveParticles * 3];

	err = cudaMemcpy(finalPositions, d_lastPositionForRK4, sizeof(double) * 3 * numOfInitialActiveParticles, cudaMemcpyDeviceToHost);
	if (err) lcs::Error("Fail to read d_lastPositionForRK4");

	FILE *fout = fopen(lastPositionFile, "w");
	for (int i = 0; i < numOfInitialActiveParticles; i++) {

		/// DEBUG ///
		//if (i != 1269494) continue;

		int gridPointID = particleRecords[i]->GetGridPointID();
		int z = gridPointID % (configure->GetBoundingBoxZRes() + 1);
		int temp = gridPointID / (configure->GetBoundingBoxZRes() + 1);
		int y = temp % (configure->GetBoundingBoxYRes() + 1);
		int x = temp / (configure->GetBoundingBoxYRes() + 1);
		fprintf(fout, "%d %d %d:", x, y, z);
		for (int j = 0; j < 3; j++)
			if (configure->UseDouble())
				fprintf(fout, " %lf", ((double *)finalPositions)[i * 3 + j]);
			else
				fprintf(fout, " %lf", ((float *)finalPositions)[i * 3 + j]);
		fprintf(fout, "\n");
	}
	fclose(fout);

	if (configure->UseDouble())
		delete [] (double *)finalPositions;
	else
		delete [] (float *)finalPositions;
}
//
///// DEBUG ///
//void CheckNaturalCoordinates() {	
//	double minX = configure->GetBoundingBoxMinX();
//	double maxX = configure->GetBoundingBoxMaxX();
//	double minY = configure->GetBoundingBoxMinY();
//	double maxY = configure->GetBoundingBoxMaxY();
//	double minZ = configure->GetBoundingBoxMinZ();
//	double maxZ = configure->GetBoundingBoxMaxZ();
//
//	int xRes = configure->GetBoundingBoxXRes();
//	int yRes = configure->GetBoundingBoxYRes();
//	int zRes = configure->GetBoundingBoxZRes();
//
//	double dx = (maxX - minX) / xRes;
//	double dy = (maxY - minY) / yRes;
//	double dz = (maxZ - minZ) / zRes;
//
//	int numOfGridPoints = (xRes + 1) * (yRes + 1) * (zRes + 1);
//
//	int node = 113979;
//
//	int Z = node % (zRes + 1);
//	node /= zRes + 1;
//	int Y = node % (yRes + 1);
//	int X = node / (yRes + 1);
//
//	lcs::Vector point(minX + X * dx, minY + Y * dy, minZ + Z * dz);
//
//	printf("%lf %lf %lf\n", point.GetX(), point.GetY(), point.GetZ());
//
//	lcs::TetrahedralGrid *grid = frames[0]->GetTetrahedralGrid();
//
//	for (int i = 0; i < globalNumOfCells; i++) {
//		lcs::Tetrahedron tet = grid->GetTetrahedron(i);
//		double coordinates[4];
//		tet.CalculateNaturalCoordinates(point, coordinates);
//		double minima = 1e100;
//		for (int j = 0; j < 4; j++)
//			if (coordinates[j] < minima) minima = coordinates[j];
//		if (minima > -1e-6) {
//			printf("i = %d\n", i);
//			printf("minima = %lf\n", minima);
//		}
//	}
//}
//

int main() {
	// Test the system
	SystemTest();

	// Read the configure file
	ReadConfFile();

	// Load all the frames
	LoadFrames();

	// Put both topological and geometrical data into arrays
	GetTopologyAndGeometry();

	// Get the global bounding box
	GetGlobalBoundingBox();

	// Calculate the number of blocks in X, Y and Z
	CalculateNumOfBlocksInXYZ();

	// Divide the flow domain into blocks
	//Division();

	InitializationForNaiveTracing();

	// Initially locate global tetrahedral cells for interesting Cartesian grid points
	InitialCellLocation();

	//int xRes = configure->GetBoundingBoxXRes();
	//int yRes = configure->GetBoundingBoxYRes();
	//int zRes = configure->GetBoundingBoxZRes();
	//int numOfGridPoints = (xRes + 1) * (yRes + 1) * (zRes + 1);

	//// Evaluate the device usage
	//int need = SizeOfDeviceArrays(globalNumOfPoints, globalNumOfCells, numOfGridPoints, numOfInitialActiveParticles, numOfQueries,
	//	       	   startOffsetInPoint[numOfInterestingBlocks], startOffsetInCell[numOfInterestingBlocks],
	//		   numOfInterestingBlocks);

	//printf("need = %d bytes\n", need);

	///// Check the natural coordinates of some points
	//CheckNaturalCoordinates();

	///// DEBUG ///
	////return 0;

	// Main Tracing Process
	Tracing();

	// Get final positions for initial active particles
	GetFinalPositions();

	return 0;
}
