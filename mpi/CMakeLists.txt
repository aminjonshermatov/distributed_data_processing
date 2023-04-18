find_package(MPI REQUIRED)

set(PROJECT mpi)

set(LAST 5)

foreach (prog RANGE 1 ${LAST})
    add_executable(${PROJECT}_${prog} ${prog}.cpp)
    target_link_libraries(${PROJECT}_${prog} PUBLIC MPI::MPI_CXX)
endforeach ()