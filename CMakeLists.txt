cmake_minimum_required(VERSION 3.24)
project(GameOfLife)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(MPI REQUIRED)

# ── Shared library (CPU engines + utils, no MPI main) ────────────────────────

add_library(gol_lib STATIC
    src/grid.cpp
    src/gol_simple.cpp
    src/gol_simd.cpp
    src/gol_bitpack.cpp
    src/gol_utils.cpp
    src/gol_mpi.cpp
)
target_include_directories(gol_lib PUBLIC src ${MPI_INCLUDE_PATH})
# -march=native enables platform SIMD (NEON on Apple Silicon, AVX2 on x86).
# For full optimisation build with: cmake -B build -DCMAKE_BUILD_TYPE=Release
target_compile_options(gol_lib PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-march=native>)
target_link_libraries(gol_lib PUBLIC ${MPI_LIBRARIES} OpenMP::OpenMP_CXX)

# ── Main executable ──────────────────────────────────────────────────────────

add_executable(golSimulator src/golSimulator.cpp)
target_link_libraries(golSimulator PRIVATE gol_lib)
# CUDA_ARCHITECTURES set conditionally below if CUDA is found.

# ── Optional GPU support ─────────────────────────────────────────────────────

set(GPU_SOURCES_ADDED FALSE)

find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
  set(CMAKE_CUDA_COMPILER "${CUDAToolkit_NVCC_EXECUTABLE}")
  set(CMAKE_CUDA_ARCHITECTURES native)
  enable_language(CUDA)
  target_sources(gol_lib PRIVATE src/gol_cuda.cu src/gol_gpu.cpp)
  target_compile_definitions(gol_lib PUBLIC GOL_CUDA=1)
  set(GPU_SOURCES_ADDED TRUE)
  set_target_properties(gol_lib PROPERTIES
    CUDA_ARCHITECTURES native
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON)
  target_compile_options(gol_lib PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-arch=native -lineinfo>)
  set_target_properties(golSimulator PROPERTIES CUDA_ARCHITECTURES native)
endif()

set(CMAKE_HIP_ARCHITECTURES native)
find_package(hip QUIET CONFIG)
if(hip_FOUND)
  enable_language(HIP)
  target_sources(gol_lib PRIVATE src/gol_hip.cpp)
  if(NOT GPU_SOURCES_ADDED)
    target_sources(gol_lib PRIVATE src/gol_gpu.cpp)
  endif()
  target_compile_definitions(gol_lib PUBLIC GOL_HIP=1)
  set_target_properties(gol_lib PROPERTIES HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")
  set_source_files_properties(src/gol_hip.cpp PROPERTIES LANGUAGE HIP)
endif()

# ── Tests (GTest + CTest) ────────────────────────────────────────────────────

include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        v1.15.2
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()
include(GoogleTest)

function(add_gol_test name)
  add_executable(${name} tests/${name}.cpp)
  target_link_libraries(${name} PRIVATE gol_lib GTest::gtest_main)
  target_compile_definitions(${name} PRIVATE
    TEST_DATA_DIR="${CMAKE_SOURCE_DIR}/data")
  gtest_discover_tests(${name})
endfunction()

add_gol_test(test_grid)
add_gol_test(test_engine)
add_gol_test(test_utils)
add_gol_test(test_integration)
