/**********************************************
File		:	lcsUtility.cpp
Author		:	Mingcheng Chen
Last Update	:	March 24th, 2013
***********************************************/

#include "lcsUtility.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cctype>
#include <sys/time.h>
#include <vector>
#include <algorithm>

////////////////////////////////////////////////
void lcs::Error(const char *str) {
	printf("Error: %s\n", str);
	exit(0);
}

void lcs::ConsumeChar(char goal, FILE *fin) {
	char ch = fgetc(fin);
	for (; ch != goal && ch != EOF; ch = fgetc(fin));
	if (ch == EOF) lcs::Error("The configure file is defective.");
}

bool lcs::IsFloatChar(char ch) {
	return isdigit(ch) || ch == '.' || ch == '-' || ch == '+';
}

double lcs::GetCurrentTimeInSeconds() {
	timeval currTime;
	gettimeofday(&currTime, 0);
	return currTime.tv_sec + currTime.tv_usec * 1e-6;
}

////////////////////////////////////////////////
lcs::Configure::Configure(const char *fileName) {
	this->DefaultSetting();

	this->fileName = std::string(fileName);
	FILE *fin = fopen(this->fileName.c_str(), "r");
	if (fin == NULL) {
		char *err = new char [this->fileName.length() + 100];
		sprintf(err, "Configure file \"%s\" cannot be opened.", this->fileName.c_str());
		lcs::Error(err);
	}
	char ch;
	while ((ch = fgetc(fin)) != EOF) {
		// Skip the comments
		if (ch == '#') {
			for (; ch != '\n' && ch != '\r' && ch != EOF; ch = fgetc(fin));
			continue;
		}

		// Get the variable name
		if (isalpha(ch)) {
			char name[50];
			int len = 0;
			for (; isalpha(ch); name[len++] = ch, ch = fgetc(fin));
			name[len] = 0;

			// Find the equation mark
			lcs::ConsumeChar('=', fin);
			if (!strcmp(name, "numOfTimePoints")) {
				printf("read numOfTimePoints ... ");
				int value;
				if (fscanf(fin, "%d", &value) != 1) lcs::Error("Fail to read \"numOfTimePoints\"");
				this->numOfTimePoints = value;
				printf("Done. numOfTimePoints = %d\n", numOfTimePoints);
				continue;
			}
			if (!strcmp(name, "numOfFrames")) {
				printf("read numOfFrames ... ");
				int numOfFrames;
				if (fscanf(fin, "%d", &numOfFrames) != 1) lcs::Error("Fail to read \"numOfFrames\"");
				this->numOfFrames = numOfFrames;
				printf("Done. numOfFrames = %d\n", numOfFrames);
				continue;
			}
			if (!strcmp(name, "sharedMemoryKilobytes")) {
				printf("read sharedMemoryKilobytes ... ");
				int sharedMemoryKilobytes;
				if (fscanf(fin, "%d", &sharedMemoryKilobytes) != 1) lcs::Error("Fail to read \"sharedMemoryKilobytes\"");
				this->sharedMemoryKilobytes = sharedMemoryKilobytes;
				printf("Done. sharedMemoryKilobytes = %d\n", sharedMemoryKilobytes);
				continue;
			}
			if (!strcmp(name, "boundingBoxXRes")) {
				printf("read boundingBoxXRes ... ");
				int value;
				if (fscanf(fin, "%d", &value) != 1) lcs::Error("Fail to read \"boundingBoxXRes\"");
				this->boundingBoxXRes = value;
				printf("Done. boundingBoxXRes = %d\n", value);
				continue;
			}
			if (!strcmp(name, "boundingBoxYRes")) {
				printf("read boundingBoxYRes ... ");
				int value;
				if (fscanf(fin, "%d", &value) != 1) lcs::Error("Fail to read \"boundingBoxYRes\"");
				this->boundingBoxYRes = value;
				printf("Done. boundingBoxYRes = %d\n", value);
				continue;
			}
			if (!strcmp(name, "boundingBoxZRes")) {
				printf("read boundingBoxZRes ... ");
				int value;
				if (fscanf(fin, "%d", &value) != 1) lcs::Error("Fail to read \"boundingBoxZRes\"");
				this->boundingBoxZRes = value;
				printf("Done. boundingBoxZRes = %d\n", value);
				continue;
			}
			if (!strcmp(name, "numOfBanks")) {
				printf("read numOfBanks ... ");
				int value;
				if (fscanf(fin, "%d", &value) != 1) lcs::Error("Fail to read \"numOfBanks\"");
				this->numOfBanks = value;
				printf("Done. numOfBanks = %d\n", value);
				continue;
			}
			if (!strcmp(name, "timePoints")) {
				printf("read timePoints ... ");
				this->timePoints.clear();
				lcs::ConsumeChar('[', fin);
				while (1) {
					ch = fgetc(fin);
					if (ch == EOF) lcs::Error("The configure file is defective.");

					// Get a float number
					if (lcs::IsFloatChar(ch)) {
						char number[50];
						int len = 0;
						for (; lcs::IsFloatChar(ch); number[len++] = ch, ch = fgetc(fin));
						number[len] = 0;
						double floatNum;
						sscanf(number, "%lf", &floatNum);
						this->timePoints.push_back(floatNum);
						if (ch == ']') break;
						continue;
					}

					// Get the ending character ']'
					if (ch == ']') break;
				}
				printf("Done. %d time points are read.\n", timePoints.size());
				continue;
			}
			if (!strcmp(name, "dataFilePrefix")) {
				printf("read dataFilePrefix ... ");
				lcs::ConsumeChar('\"', fin);
				this->dataFilePrefix = "";
				while (1) {
					ch = fgetc(fin);
					if (ch == EOF) lcs::Error("The configure file is defective.");
					if (ch == '\"') break;
					this->dataFilePrefix += ch;
				}
				printf("Done. dataFilePrefix = %s\n", dataFilePrefix.c_str());
				continue;
			}
			if (!strcmp(name, "dataFileSuffix")) {
				printf("read dataFileSuffix ... ");
				lcs::ConsumeChar('\"', fin);
				this->dataFileSuffix = "";
				while (1) {
					ch = fgetc(fin);
					if (ch == EOF) lcs::Error("The configure file is defective.");
					if (ch == '\"') break;
					this->dataFileSuffix += ch;
				}
				printf("Done. dataFileSuffix = %s\n", dataFileSuffix.c_str());
				continue;
			}
			if (!strcmp(name, "dataFileIndices")) {
				printf("read dataFileIndices ... ");
				this->dataFileIndices.clear();
				lcs::ConsumeChar('[', fin);
				while (1) {
					ch = fgetc(fin);
					if (ch == EOF) lcs::Error("The configure file is defective.");

					// Get an integer
					if (isdigit(ch)) {
						char number[50];
						int len = 0;
						for (; isdigit(ch); number[len++] = ch, ch = fgetc(fin));
						number[len] = 0;
						this->dataFileIndices.push_back(number);
						if (ch == ']') break;
						continue;
					}

					// Get the ending character ']'
					if (ch == ']') break;
				}
				printf("Done. %d data file indices are read.\n", dataFileIndices.size());
				continue;
			}
			if (!strcmp(name, "integration")) {
				printf("read integration ... ");
				lcs::ConsumeChar('\"', fin);
				this->integration = "";
				while (1) {
					ch = fgetc(fin);
					if (ch == EOF) lcs::Error("The configure file is defective.");
					if (ch == '\"') break;
					this->integration += ch;
				}
				printf("Done. integration = %s\n", integration.c_str());
				continue;
			}
			if (!strcmp(name, "timeStep")) {
				printf("read timeStep ... ");
				double timeStep;
				if (fscanf(fin, "%lf", &timeStep) != 1) lcs::Error("Fail to read \"timeStep\"");
				this->timeStep = timeStep;
				printf("Done. timeStep = %lf\n", timeStep);
				continue;
			}
			if (!strcmp(name, "blockSize")) {
				printf("read blockSize ... ");
				double blockSize;
				if (fscanf(fin, "%lf", &blockSize) != 1) lcs::Error("Fail to read \"blockSize\"");
				this->blockSize = blockSize;
				printf("Done. blockSize = %lf\n", blockSize);
				continue;
			}
			if (!strcmp(name, "timeInterval")) {
				printf("read timeInterval ... ");
				double timeInterval;
				if (fscanf(fin, "%lf", &timeInterval) != 1) lcs::Error("Fail to read \"timeInterval\"");
				this->timeInterval = timeInterval;
				printf("Done. timeInterval = %lf\n", timeInterval);
				continue;
			}
			if (!strcmp(name, "epsilonForTetBlkIntersection")) {
				printf("read epsilonForTetBlkIntersection ... ");
				double epsilonForTetBlkIntersection;
				if (fscanf(fin, "%lf", &epsilonForTetBlkIntersection) != 1) lcs::Error("Fail to read \"epsilonForTetBlkIntersection\"");
				this->epsilonForTetBlkIntersection = epsilonForTetBlkIntersection;
				printf("Done. epsilonForTetBlkIntersection = %le\n", epsilonForTetBlkIntersection);
				continue;
			}
			if (!strcmp(name, "epsilon")) {
				printf("read epsilon ... ");
				double epsilon;
				if (fscanf(fin, "%lf", &epsilon) != 1) lcs::Error("Fail to read \"epsilon\"");
				this->epsilon = epsilon;
				printf("Done. epsilon = %le\n", epsilon);
				continue;
			}
			if (!strcmp(name, "marginRatio")) {
				printf("read marginRatio ... ");
				double value;
				if (fscanf(fin, "%lf", &value) != 1) lcs::Error("Fail to read \"marginRatio\"");
				this->marginRatio = value;
				printf("Done. marginRatio = %lf\n", value);
				continue;
			}
			if (!strcmp(name, "boundingBoxMinX")) {
				printf("read boundingBoxMinX ... ");
				double value;
				if (fscanf(fin, "%lf", &value) != 1) lcs::Error("Fail to read \"boundingBoxMinX\"");
				this->boundingBoxMinX = value;
				printf("Done. boundingBoxMinX = %lf\n", value);
				continue;
			}
			if (!strcmp(name, "boundingBoxMaxX")) {
				printf("read boundingBoxMaxX ... ");
				double value;
				if (fscanf(fin, "%lf", &value) != 1) lcs::Error("Fail to read \"boundingBoxMaxX\"");
				this->boundingBoxMaxX = value;
				printf("Done. boundingBoxMaxX = %lf\n", value);
				continue;
			}
			if (!strcmp(name, "boundingBoxMinY")) {
				printf("read boundingBoxMinY ... ");
				double value;
				if (fscanf(fin, "%lf", &value) != 1) lcs::Error("Fail to read \"boundingBoxMinY\"");
				this->boundingBoxMinY = value;
				printf("Done. boundingBoxMinY = %lf\n", value);
				continue;
			}
			if (!strcmp(name, "boundingBoxMaxY")) {
				printf("read boundingBoxMaxY ... ");
				double value;
				if (fscanf(fin, "%lf", &value) != 1) lcs::Error("Fail to read \"boundingBoxMaxY\"");
				this->boundingBoxMaxY = value;
				printf("Done. boundingBoxMaxY = %lf\n", value);
				continue;
			}
			if (!strcmp(name, "boundingBoxMinZ")) {
				printf("read boundingBoxMinZ ... ");
				double value;
				if (fscanf(fin, "%lf", &value) != 1) lcs::Error("Fail to read \"boundingBoxMinZ\"");
				this->boundingBoxMinZ = value;
				printf("Done. boundingBoxMinZ = %lf\n", value);
				continue;
			}
			if (!strcmp(name, "boundingBoxMaxZ")) {
				printf("read boundingBoxMaxZ ... ");
				double value;
				if (fscanf(fin, "%lf", &value) != 1) lcs::Error("Fail to read \"boundingBoxMaxZ\"");
				this->boundingBoxMaxZ = value;
				printf("Done. boundingBoxMaxZ = %lf\n", value);
				continue;
			}
			if (!strcmp(name, "double")) {
				printf("read double ... ");
				char status[50];
				if (fscanf(fin, "%s", status) != 1) lcs::Error("Fail to read \"double\"");
				this->useDouble = tolower(status[0]) == 'e';
				printf("Done. double = %s\n", status);
				continue;
			}
			if (!strcmp(name, "unitTestForTetBlkIntersection")) {
				printf("read unitTestForTetBlkIntersection ... ");
				char status[50];
				if (fscanf(fin, "%s", status) != 1) lcs::Error("Fail to read \"unitTestForTetBlkIntersection\"");
				this->unitTestForTetBlkIntersection = tolower(status[0]) == 'e';
				printf("Done. unitTestForTetBlkIntersection = %s\n", status);
				continue;
			}
			if (!strcmp(name, "unitTestForInitialCellLocation")) {
				printf("read unitTestForInitialCellLocation ... ");
				char status[50];
				if (fscanf(fin, "%s", status) != 1) lcs::Error("Fail to read \"unitTestForInitialCellLocation\"");
				this->unitTestForInitialCellLocation = tolower(status[0]) == 'e';
				printf("Done. unitTestForInitialCellLocation = %s\n", status);
				continue;
			}	
		}
	}
}

void lcs::Configure::DefaultSetting() {
	this->fileName = "";
	this->dataFilePrefix = "";
	this->dataFileSuffix = "";
	this->integration = "RK4";
	this->numOfFrames = 0;
	this->timePoints.clear();
	this->dataFileIndices.clear();
	this->timeStep = 0.1;
	this->blockSize = 1.0;
	this->epsilon = 1e-8;
	// TODO: May add more default settings
}

int lcs::Configure::GetNumOfTimePoints() const {
	return this->numOfTimePoints;
}

int lcs::Configure::GetNumOfFrames() const {
	return this->numOfFrames;
}

int lcs::Configure::GetSharedMemoryKilobytes() const {
	return this->sharedMemoryKilobytes;
}

int lcs::Configure::GetBoundingBoxXRes() const {
	return this->boundingBoxXRes;
}

int lcs::Configure::GetBoundingBoxYRes() const {
	return this->boundingBoxYRes;
}

int lcs::Configure::GetBoundingBoxZRes() const {
	return this->boundingBoxZRes;
}

int lcs::Configure::GetNumOfBanks() const {
	return this->numOfBanks;
}

double lcs::Configure::GetTimeStep() const {
	return this->timeStep;
}

double lcs::Configure::GetBlockSize() const {
	return this->blockSize;
}

double lcs::Configure::GetTimeInterval() const {
	return this->timeInterval;
}

double lcs::Configure::GetEpsilonForTetBlkIntersection() const {
	return this->epsilonForTetBlkIntersection;
}

double lcs::Configure::GetEpsilon() const {
	return this->epsilon;
}

double lcs::Configure::GetMarginRatio() const {
	return this->marginRatio;
}

double lcs::Configure::GetBoundingBoxMinX() const {
	return this->boundingBoxMinX;
}

double lcs::Configure::GetBoundingBoxMaxX() const {
	return this->boundingBoxMaxX;
}

double lcs::Configure::GetBoundingBoxMinY() const {
	return this->boundingBoxMinY;
}

double lcs::Configure::GetBoundingBoxMaxY() const {
	return this->boundingBoxMaxY;
}

double lcs::Configure::GetBoundingBoxMinZ() const {
	return this->boundingBoxMinZ;
}

double lcs::Configure::GetBoundingBoxMaxZ() const {
	return this->boundingBoxMaxZ;
}

std::string lcs::Configure::GetFileName() const {
	return this->fileName;
}

std::string lcs::Configure::GetDataFilePrefix() const {
	return this->dataFilePrefix;
}

std::string lcs::Configure::GetDataFileSuffix() const {
	return this->dataFileSuffix;
}

std::string lcs::Configure::GetIntegration() const {
	return this->integration;
}

std::vector<double> lcs::Configure::GetTimePoints() const {
	return this->timePoints;
}

std::vector<std::string> lcs::Configure::GetDataFileIndices() const {
	return this->dataFileIndices;
}
/*
bool lcs::Configure::UseDouble() const {
	return this->useDouble;
}
*/
bool lcs::Configure::UseUnitTestForTetBlkIntersection() const {
	return this->unitTestForTetBlkIntersection;
}

bool lcs::Configure::UseUnitTestForInitialCellLocation() const {
	return this->unitTestForInitialCellLocation;
}

