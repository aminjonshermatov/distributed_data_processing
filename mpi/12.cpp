//
// Created by aminjon on 4/19/23.
//
#include <iostream>
#include <mpi.h>

inline constexpr int MASTER_PROC = 0;
inline constexpr int SEND_TAG = 0;

inline int next_proc(int proc_id, int n_procs) { return (proc_id + 1) % n_procs; }
inline int prev_proc(int proc_id, int n_procs) { return (proc_id - 1 + n_procs) % n_procs; }
inline int functor(int x) { return x + 7; }

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);

  const auto rec = [](auto &&self, const MPI_Comm &comm) -> void {
    int n_procs;
    MPI_Comm_size(comm, &n_procs);
    int cur_proc;
    MPI_Comm_rank(comm, &cur_proc);

    if (cur_proc == MASTER_PROC) {
      int tmp = 0;
      std::cout << "At beginning: " << tmp << std::endl;
      if (next_proc(cur_proc, n_procs) == cur_proc) return;
      MPI_Send(&tmp,
               1,
               MPI_INT,
               next_proc(cur_proc, n_procs),
               SEND_TAG,
               comm);
    } else {
      int tmp;
      MPI_Status status;
      MPI_Recv(&tmp,
               1,
               MPI_INT,
               prev_proc(cur_proc, n_procs),
               SEND_TAG,
               comm,
               &status);
      tmp = functor(tmp);
      MPI_Send(&tmp,
               1,
               MPI_INT,
               next_proc(cur_proc, n_procs),
               SEND_TAG,
               comm);
    }

    if (cur_proc == MASTER_PROC) {
      int tmp;
      MPI_Status status;
      MPI_Recv(&tmp,
               1,
               MPI_INT,
               prev_proc(cur_proc, n_procs),
               SEND_TAG,
               comm,
               &status);
      std::cout << "Size: " << n_procs << ", result: " << tmp << std::endl;
    }

    MPI_Group old_group, new_group;
    MPI_Comm_group(comm, &old_group);
    int blocked_procs[1] = {n_procs - 1};
    MPI_Group_excl(old_group, 1, blocked_procs, &new_group);
    MPI_Comm new_comm;
    MPI_Comm_create(comm, new_group, &new_comm);
    if (new_comm != MPI_COMM_NULL) {
      self(self, new_comm);
      MPI_Group_free(&new_group);
      MPI_Comm_free(&new_comm);
    }
  };

  rec(rec, MPI_COMM_WORLD);

  MPI_Finalize();
}