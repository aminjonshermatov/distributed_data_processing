find_package(MPI REQUIRED)

set(PROJECT mpi)

add_executable(${PROJECT}_floyd floyd.cpp)
target_link_libraries(${PROJECT}_floyd PUBLIC MPI::MPI_CXX)