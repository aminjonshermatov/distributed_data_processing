//
// Created by aminjon on 4/19/23.
//
#include <iostream>
#include <chrono>
#include <mpi.h>

inline constexpr int MASTER_PROC = 0;
inline constexpr int SEND_TAG = 0;

inline int next_proc(int proc_id, int n_procs) { return (proc_id + 1) % n_procs; }
inline int prev_proc(int proc_id, int n_procs) { return (proc_id - 1 + n_procs) % n_procs; }
inline int functor(int x) { return x + 7; }

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);

  int n_procs;
  MPI_Comm_size(MPI_COMM_WORLD, &n_procs);
  int cur_proc;
  MPI_Comm_rank(MPI_COMM_WORLD, &cur_proc);

  if (cur_proc == MASTER_PROC) {
    int tmp = 0;
    std::cout << "At beginning: " << tmp << std::endl;
    MPI_Send(&tmp,
             1,
             MPI_INT,
             next_proc(cur_proc, n_procs),
             SEND_TAG,
             MPI_COMM_WORLD);
    MPI_Status status;
    MPI_Recv(&tmp,
             1,
             MPI_INT,
             prev_proc(cur_proc, n_procs),
             SEND_TAG,
             MPI_COMM_WORLD,
             &status);
    std::cout << "Final result: " << tmp << std::endl;
  } else {
    int tmp;
    MPI_Status status;
    MPI_Recv(&tmp,
             1,
             MPI_INT,
             prev_proc(cur_proc, n_procs),
             SEND_TAG,
             MPI_COMM_WORLD,
             &status);
    tmp = functor(tmp);
    MPI_Send(&tmp,
             1,
             MPI_INT,
             next_proc(cur_proc, n_procs),
             SEND_TAG,
             MPI_COMM_WORLD);
  }

  MPI_Finalize();
}