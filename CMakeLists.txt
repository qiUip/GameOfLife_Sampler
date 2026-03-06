cmake_minimum_required(VERSION 3.16)
project(GameOfLife)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(MPI REQUIRED)

add_executable(golSimulator
    golSimulator.cpp
    gol.cpp
    gol_simple.cpp
    gol_simd.cpp
    gol_bitpack.cpp
    gol_utils.cpp
)
# -march=native enables platform SIMD (NEON on Apple Silicon, AVX2 on x86).
# For full optimisation build with: cmake -B build -DCMAKE_BUILD_TYPE=Release
target_compile_options(golSimulator PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-march=native>)
target_link_libraries(golSimulator ${MPI_LIBRARIES} OpenMP::OpenMP_CXX)
target_include_directories(golSimulator PRIVATE ${MPI_INCLUDE_PATH})

# Optional CUDA support
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  target_sources(golSimulator PRIVATE gol_cuda.cu)
  target_compile_definitions(golSimulator PRIVATE GOL_CUDA=1)
  set_target_properties(golSimulator PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON)
  target_compile_options(golSimulator PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-arch=native -lineinfo>)
endif()
