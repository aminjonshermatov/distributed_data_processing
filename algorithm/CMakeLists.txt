find_package(MPI REQUIRED)

add_executable(floyd_mpi floyd_mpi.cpp)
target_link_libraries(floyd_mpi PUBLIC MPI::MPI_CXX)