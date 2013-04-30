project(Fastvection)

cmake_minimum_required(VERSION 2.8)

find_package(CUDA)
#include(${CUDA_USE_FILE})

find_package(VTK)

include_directories(${VTK_INCLUDE_DIRS})

#if (VTK_FOUND)
#	include(${VTK_USE_FILE})
#else (VTK_FOUND)
#	message(FATAL_ERROR, "Cannot build the executable without VTK. Please set the VTK_DIR")
#endif (VTK_FOUND)

set (CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -gencode arch=compute_35,code=sm_35)

cuda_compile(lcsBigBlockInitializationForPositions lcsBigBlockInitializationForPositions.cu)
cuda_compile(lcsBigBlockInitializationForVelocities lcsBigBlockInitializationForVelocities.cu)
cuda_compile(lcsBlockedTracingOfRK4 lcsBlockedTracingOfRK4.cu)
cuda_compile(lcsCollectActiveParticlesForNewInterval lcsCollectActiveParticlesForNewInterval.cu)
cuda_compile(lcsCollectActiveParticlesForNewRun lcsCollectActiveParticlesForNewRun.cu)
cuda_compile(lcsExclusiveScanForInt lcsExclusiveScanForInt.cu)
cuda_compile(lcsGetGroupsForBlocks lcsGetGroupsForBlocks.cu)
cuda_compile(lcsGetStartOffsetInParticles lcsGetStartOffsetInParticles.cu)
cuda_compile(lcsInitialCellLocation lcsInitialCellLocation.cu)
cuda_compile(lcsRedistributeParticles lcsRedistributeParticles.cu)
cuda_compile(lcsTetrahedronBlockIntersection lcsTetrahedronBlockIntersection.cu)

cuda_add_executable(fastvection lcs.cpp lcsGeometry.cpp lcsUnitTest.cpp lcsUtility.cpp main.cpp
	${lcsBigBlockInitializationForPositions}
	${lcsBigBlockInitializationForVelocities}
	${lcsBlockedTracingOfRK4}
	${lcsCollectActiveParticlesForNewInterval}
	${lcsCollectActiveParticlesForNewRun}
	${lcsExclusiveScanForInt}
	${lcsGetGroupsForBlocks}
	${lcsGetStartOffsetInParticles}
	${lcsInitialCellLocation}
	${lcsRedistributeParticles}
	${lcsTetrahedronBlockIntersection}
)

target_link_libraries(fastvection ${VTK_LIBRARIES})
