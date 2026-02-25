cmake_minimum_required(VERSION 3.16)
project(GameOfLife)
set(CMAKE_CXX_STANDARD 17)

find_package(OpenMP REQUIRED)
find_package(MPI REQUIRED)

add_executable(golSimulator
    golSimulator.cpp
    gol.cpp
    gol_utils.cpp
)
target_link_libraries(golSimulator ${MPI_LIBRARIES} OpenMP::OpenMP_CXX)
target_include_directories(golSimulator PRIVATE ${MPI_INCLUDE_PATH})
