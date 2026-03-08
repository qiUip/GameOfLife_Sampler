cmake_minimum_required(VERSION 3.16)
project(GameOfLife)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# On macOS with homebrew LLVM, use homebrew's libomp
if(APPLE)
  set(OpenMP_CXX_FLAGS "-fopenmp")
  set(OpenMP_CXX_LIB_NAMES "omp")
  set(OpenMP_omp_LIBRARY "/opt/homebrew/opt/libomp/lib/libomp.dylib")
endif()

find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(MPI REQUIRED)

add_executable(golSimulator
    golSimulator.cpp
    grid.cpp
    gol_simple.cpp
    gol_simd.cpp
    gol_bitpack.cpp
    gol_utils.cpp
    gol_mpi.cpp
)
# -march=native enables platform SIMD (NEON on Apple Silicon, AVX2 on x86).
# For full optimisation build with: cmake -B build -DCMAKE_BUILD_TYPE=Release
target_compile_options(golSimulator PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-march=native>)
target_link_libraries(golSimulator PRIVATE ${MPI_LIBRARIES} OpenMP::OpenMP_CXX)
target_include_directories(golSimulator PRIVATE ${MPI_INCLUDE_PATH})

# Optional GPU support
set(GPU_SOURCES_ADDED FALSE)

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  target_sources(golSimulator PRIVATE gol_cuda.cu gol_gpu.cpp)
  target_compile_definitions(golSimulator PRIVATE GOL_CUDA=1)
  set(GPU_SOURCES_ADDED TRUE)
  set_target_properties(golSimulator PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON)
  target_compile_options(golSimulator PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-arch=native -lineinfo>)
endif()

if(NOT CMAKE_HIP_ARCHITECTURES)
  set(CMAKE_HIP_ARCHITECTURES "gfx942")
endif()
check_language(HIP)
if(CMAKE_HIP_COMPILER)
  enable_language(HIP)
  target_sources(golSimulator PRIVATE gol_hip.cpp)
  if(NOT GPU_SOURCES_ADDED)
    target_sources(golSimulator PRIVATE gol_gpu.cpp)
  endif()
  target_compile_definitions(golSimulator PRIVATE GOL_HIP=1)
  set_target_properties(golSimulator PROPERTIES HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")
  set_source_files_properties(gol_hip.cpp PROPERTIES LANGUAGE HIP)
endif()
