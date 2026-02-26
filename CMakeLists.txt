cmake_minimum_required(VERSION 3.16)
project(GameOfLife)
set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(MPI REQUIRED)

add_executable(golSimulator
    golSimulator.cpp
    gol.cpp
    gol_utils.cpp
)
# -march=native enables platform SIMD (NEON on Apple Silicon, AVX2 on x86).
# For full optimisation build with: cmake -B build -DCMAKE_BUILD_TYPE=Release
target_compile_options(golSimulator PRIVATE -march=native)
target_link_libraries(golSimulator ${MPI_LIBRARIES} OpenMP::OpenMP_CXX)
target_include_directories(golSimulator PRIVATE ${MPI_INCLUDE_PATH})
