//
// Created by aminjon on 4/18/23.
//
#include <iostream>
#include <mpi.h>

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);

  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  int proc_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);

  std::cout << "Hello world from processor rank " << proc_rank << " out of " << world_size << " processors\n";

  MPI_Finalize();
}