/**********************************************
File			:		lcs.cpp
Author			:		Mingcheng Chen
Last Update		:		February 25th, 2013
***********************************************/

#include "lcs.h"
#include "lcsUtility.h"
//#include <vtkXMLUnstructuredGridReader.h>
#include <cstring>

////////////////////////////////////////////////
lcs::Frame::Frame(double timePoint, const char *geometryFile, const char *velocityFile) {
	//vtkXMLUnstructuredGridReader *reader = vtkXMLUnstructuredGridReader::New();
	//reader->SetFileName(dataFile);
	//reader->Update();

	//	FILE *fin = fopen("data/geometry.txt", "rb");
	//int numOfCells, numOfPoints;
	//fread(&numOfCells, sizeof(int), 1, fin);
	//fread(&numOfPoints, sizeof(int), 1, fin);
	//printf("%d %d\n", numOfCells, numOfPoints);
	//int *conn = new int [numOfCells * 4];
	//int *link = new int [numOfCells * 4];
	//double *posi = new double [numOfPoints * 3];
	////fread(conn, sizeof(int), numOfCells * 4, fin);
	//for (int i = 0; i < numOfCells * 4; i++)
	//	fread(conn + i, sizeof(int), 1, fin);
	////fread(link, sizeof(int), numOfCells * 4, fin);
	////fread(posi, sizeof(double), numOfPoints * 3, fin);
	//for (int i = 0; i < 10; i++)
	//	printf(" %d", conn[i + 640]);
	//printf("\n");
	//fclose(fin);

	FILE *fin = fopen(geometryFile, "rb");
	if (!fin) lcs::Error("Fail to open file");
	int numOfPoints, numOfCells;
	fread(&numOfCells, sizeof(int), 1, fin);
	fread(&numOfPoints, sizeof(int), 1, fin);

	/// DEBUG ///
	//printf("numOfPoints = %d, numOfCells = %d\n", numOfPoints, numOfCells);

	int *conn = new int [numOfCells << 2];
	int *link = new int [numOfCells << 2];
	double *posi = new double [numOfPoints * 3];
	fread(conn, sizeof(int), numOfCells << 2, fin);
	fread(link, sizeof(int), numOfCells << 2, fin);

	fread(posi, sizeof(double), numOfPoints * 3, fin);
	fclose(fin);

	/// DEBUG ///
	//for (int i = 0; i < 30; i++)
	//	printf("%lf ", posi[i]);
	//printf("\n");

	fin = fopen(velocityFile, "rb");
	double *velo = new double [numOfPoints * 3];
	fread(velo, sizeof(double), numOfPoints * 3, fin);
	fclose(fin);
	this->tetrahedralGrid = new TetrahedralGrid(numOfCells, numOfPoints, conn, link, posi, velo);
	this->timePoint = timePoint;
	delete [] conn;
	delete [] link;
	delete [] posi;
	delete [] velo;
}

lcs::Frame::~Frame() {
	delete this->tetrahedralGrid;
}

////////////////////////////////////////////////
lcs::ParticleRecord::ParticleRecord() {
	this->stage = -1;
	this->data = NULL;
}

lcs::ParticleRecord::ParticleRecord(int stage, int gridPointID, void *data) {
	this->stage = stage;
	this->gridPointID = gridPointID;
	this->data = data;
}

lcs::ParticleRecord::~ParticleRecord() {
	delete (ParticleRecordDataForRK4 *)this->data;
}

int lcs::ParticleRecord::dataType = 0;

void lcs::ParticleRecord::SetDataType(int type) {
	lcs::ParticleRecord::dataType = type;
}

int lcs::ParticleRecord::GetDataType() {
	return lcs::ParticleRecord::dataType;
}

lcs::Vector lcs::ParticleRecord::GetPositionInInterest() const {
	switch (lcs::ParticleRecord::dataType) {
	case lcs::ParticleRecord::RK4: {
		lcs::ParticleRecordDataForRK4 *data = (lcs::ParticleRecordDataForRK4 *)this->data;
		switch (this->stage) {
		case lcs::ParticleRecordDataForRK4::COMPUTING_K1: {
			return data->GetLastPosition();
														  } break;
		case lcs::ParticleRecordDataForRK4::COMPUTING_K2: {
			return data->GetLastPosition() + data->GetK1() * 0.5;
														  } break;
		case lcs::ParticleRecordDataForRK4::COMPUTING_K3: {
			return data->GetLastPosition() + data->GetK2() * 0.5;
														  } break;
		case lcs::ParticleRecordDataForRK4::COMPUTING_K4: {
			return data->GetLastPosition() + data->GetK3();
														  } break;
		}
								   } break;
	}
}

int lcs::ParticleRecord::GetStage() const {
	return this->stage;
}

int lcs::ParticleRecord::GetGridPointID() const {
	return this->gridPointID;
}

void *lcs::ParticleRecord::GetData() const {
	return this->data;
}

void lcs::ParticleRecord::SetStage(int stage) {
	this->stage = stage;
}

////////////////////////////////////////////////
void lcs::ParticleRecordDataForRK4::SetLastPosition(const lcs::Vector &lastPosition) {
	this->lastPosition = lastPosition;
}

void lcs::ParticleRecordDataForRK4::SetK1(const lcs::Vector &k1) {
	this->k1 = k1;
}

void lcs::ParticleRecordDataForRK4::SetK2(const lcs::Vector &k2) {
	this->k2 = k2;
}

void lcs::ParticleRecordDataForRK4::SetK3(const lcs::Vector &k3) {
	this->k3 = k3;
}

void lcs::ParticleRecordDataForRK4::SetLastPosition(const double *arr) {
	this->lastPosition.SetX(arr[0]);
	this->lastPosition.SetY(arr[1]);
	this->lastPosition.SetZ(arr[2]);
}

void lcs::ParticleRecordDataForRK4::SetK1(const double *arr) {
	this->k1.SetX(arr[0]);
	this->k1.SetY(arr[1]);
	this->k1.SetZ(arr[2]);
}

void lcs::ParticleRecordDataForRK4::SetK2(const double *arr) {
	this->k2.SetX(arr[0]);
	this->k2.SetY(arr[1]);
	this->k2.SetZ(arr[2]);
}

void lcs::ParticleRecordDataForRK4::SetK3(const double *arr) {
	this->k3.SetX(arr[0]);
	this->k3.SetY(arr[1]);
	this->k3.SetZ(arr[2]);
}

void lcs::ParticleRecordDataForRK4::SetLastPosition(const float *arr) {
	this->lastPosition.SetX(arr[0]);
	this->lastPosition.SetY(arr[1]);
	this->lastPosition.SetZ(arr[2]);
}

void lcs::ParticleRecordDataForRK4::SetK1(const float *arr) {
	this->k1.SetX(arr[0]);
	this->k1.SetY(arr[1]);
	this->k1.SetZ(arr[2]);
}

void lcs::ParticleRecordDataForRK4::SetK2(const float *arr) {
	this->k2.SetX(arr[0]);
	this->k2.SetY(arr[1]);
	this->k2.SetZ(arr[2]);
}

void lcs::ParticleRecordDataForRK4::SetK3(const float *arr) {
	this->k3.SetX(arr[0]);
	this->k3.SetY(arr[1]);
	this->k3.SetZ(arr[2]);
}


lcs::Vector lcs::ParticleRecordDataForRK4::GetLastPosition() const {
	return this->lastPosition;
}

lcs::Vector lcs::ParticleRecordDataForRK4::GetK1() const {
	return this->k1;
}

lcs::Vector lcs::ParticleRecordDataForRK4::GetK2() const {
	return this->k2;
}

lcs::Vector lcs::ParticleRecordDataForRK4::GetK3() const {
	return this->k3;
}

////////////////////////////////////////////////
lcs::BlockRecord::BlockRecord() {
	this->globalCellIDs = NULL;
	this->globalPointIDs = NULL;
	this->localConnectivities = NULL;
	this->localLinks = NULL;
	this->localNumOfCells = -1;
	this->localNumOfPoints = -1;
}

lcs::BlockRecord::~BlockRecord() {
	delete [] this->globalCellIDs;
	delete [] this->globalPointIDs;
	delete [] this->localConnectivities;
	delete [] this->localLinks;
}

int lcs::BlockRecord::GetLocalNumOfCells() const {
	return this->localNumOfCells;
}

int lcs::BlockRecord::GetLocalNumOfPoints() const {
	return this->localNumOfPoints;
}

void lcs::BlockRecord::SetLocalNumOfCells(int localNumOfCells) {
	this->localNumOfCells = localNumOfCells;
}

void lcs::BlockRecord::SetLocalNumOfPoints(int localNumOfPoints) {
	this->localNumOfPoints = localNumOfPoints;
}

void lcs::BlockRecord::CreateGlobalCellIDs(int *cellIDs) {
	if (this->localNumOfCells == -1) lcs::Error("Error on lcs::BlockRecord::CreateGlobalCellIDs(int *cellIDs): this->localNumOfCells is not set.\n");
	this->globalCellIDs = new int [this->localNumOfCells];
	memcpy(this->globalCellIDs, cellIDs, sizeof(int) * this->localNumOfCells);
}

void lcs::BlockRecord::CreateGlobalPointIDs(int *pointIDs) {
	if (this->localNumOfPoints == -1) lcs::Error("Error on lcs::BlockRecord::CreateGlobalPointIDs(int *pointIDs): this->localNumOfPoints is not set.\n");
	this->globalPointIDs = new int [this->localNumOfPoints];
	memcpy(this->globalPointIDs, pointIDs, sizeof(int) * this->localNumOfPoints);
}

void lcs::BlockRecord::CreateLocalConnectivities(int *connectivities) {
	if (this->localNumOfCells == -1) lcs::Error("Error on lcs::BlockRecord::CreateLocalConnectivities(int *connectivities): this->localNumOfCells is not set.\n");
	this->localConnectivities = new int [this->localNumOfCells * 4];
	memcpy(this->localConnectivities, connectivities, sizeof(int) * this->localNumOfCells * 4);
}

void lcs::BlockRecord::CreateLocalLinks(int *links) {
	if (this->localNumOfCells == -1) lcs::Error("Error on lcs::BlockRecord::CreateLocalLinks(int *links): this->localNumOfCells is not set.\n");
	this->localLinks = new int [this->localNumOfCells * 4];
	memcpy(this->localLinks, links, sizeof(int) * this->localNumOfCells * 4);
}

int lcs::BlockRecord::EvaluateNumOfBytes(int numOfIntervals) const {
	return this->localNumOfCells * sizeof(int) * 4 +				// this->localConnectivities
		   this->localNumOfCells * sizeof(int) * 4 +				// this->localLinks
		   this->localNumOfPoints * sizeof(double) * 3 +			// point positions
		   this->localNumOfPoints * sizeof(double) * 3 * (numOfIntervals + 1);	// point velocities (start and end)
}

int lcs::BlockRecord::GetGlobalCellID(int localCellID) const {
	if (localCellID >= this->localNumOfCells) lcs::Error("Error on lcs::BlockRecord::GetGlobalCellID(int localCellID): Out of bound.\n");
	return this->globalCellIDs[localCellID];
}

int lcs::BlockRecord::GetGlobalPointID(int localPointID) const {
	if (localPointID >= this->localNumOfPoints) lcs::Error("Error on lcs::BlockRecord::GetGlobalPointID(int localPointID): Out of bound.\n");
	return this->globalPointIDs[localPointID];
}

int *lcs::BlockRecord::GetGlobalCellIDs() const {
	return this->globalCellIDs;
}

int *lcs::BlockRecord::GetGlobalPointIDs() const {
	return this->globalPointIDs;
}

int *lcs::BlockRecord::GetLocalConnectivities() const {
	return this->localConnectivities;
}

int *lcs::BlockRecord::GetLocalLinks() const {
	return this->localLinks;
}

////////////////////////////////////////////////
lcs::BlockTetrahedronPair::BlockTetrahedronPair(int blockID, int tetrahedronID) {
	this->blockID = blockID;
	this->tetrahedronID = tetrahedronID;
}

int lcs::BlockTetrahedronPair::GetBlockID() const {
	return this->blockID;
}

int lcs::BlockTetrahedronPair::GetTetrahedronID() const {
	return this->tetrahedronID;
}

bool lcs::BlockTetrahedronPair::operator == (const BlockTetrahedronPair &anotherPair) const {
	return this->blockID == anotherPair.blockID && this->tetrahedronID == anotherPair.tetrahedronID;
}

////////////////////////////////////////////////
int lcs::HashForBlockTetrahedronPair::operator () (const lcs::BlockTetrahedronPair &btPair) const {
	return btPair.GetBlockID() * 2147483647 + btPair.GetTetrahedronID();
}

////////////////////////////////////////////////
lcs::ExecutionBlock::ExecutionBlock() {
	this->blockRecord = NULL;
	this->numOfParticles = 0;
	this->particleIDs = NULL;
}

lcs::ExecutionBlock::~ExecutionBlock() {
	if (this->blockRecord) delete [] this->blockRecord;
	if (this->particleIDs) delete [] this->particleIDs;
}

void lcs::ExecutionBlock::SetNumOfParticles(int numOfParticles) {
	this->numOfParticles = numOfParticles;
	this->particleIDs = new int [numOfParticles];
}

void lcs::ExecutionBlock::SetParticleID(int index, int particleID) {
	this->particleIDs[index] = particleID;
}

void lcs::ExecutionBlock::SetBlockRecord(const lcs::BlockRecord *blockRecord) {
	this->blockRecord = blockRecord;
}
