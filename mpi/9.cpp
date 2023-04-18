//
// Created by aminjon on 4/19/23.
//
#include <chrono>
#include <iostream>
#include <mpi.h>
#include <random>

inline constexpr std::size_t N = 10u;
inline constexpr auto MASTER_PROC = 0;
inline constexpr int SEND_LEN_TAG = 0;
inline constexpr int SEND_ARRAY_TAG = 1;

inline constexpr int LB = 1;
inline constexpr int RB = 100;

inline std::mt19937_64 rnd_device(std::chrono::steady_clock::now().time_since_epoch().count());
inline std::uniform_int_distribution<int64_t> rnd_range(LB, RB);

inline std::array<int64_t, N> array{};

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);

  int n_procs;
  MPI_Comm_size(MPI_COMM_WORLD, &n_procs);
  int cur_proc;
  MPI_Comm_rank(MPI_COMM_WORLD, &cur_proc);

  if (cur_proc == MASTER_PROC) {
    [&]() {
      for (std::size_t i = 0u; i < N; ++i) {
        array[i] = rnd_range(rnd_device);
      }
      for (std::size_t i = 0u; i < N; ++i) {
        std::cout << array[i] << (i + 1 < N ? ' ' : '\n');
      }
    }();

    const int per_proc = (N + n_procs - 1) / n_procs;
    int used = 0;
    for (int proc_id = 1; proc_id < n_procs; ++proc_id) {
      MPI_Send(&per_proc,
               1,
               MPI_INT,
               proc_id,
               SEND_LEN_TAG,
               MPI_COMM_WORLD);
      MPI_Send(array.data() + per_proc * (proc_id - 1),
               per_proc,
               MPI_INT64_T,
               proc_id,
               SEND_ARRAY_TAG,
               MPI_COMM_WORLD);
      used += per_proc;
    }

    auto rem = N - used;
    std::array<int64_t, N> reversed{};
    for (int i = N - 1; i >= used; --i) {
      reversed[N - 1 - i] = array[i];
    }

    for (int proc_id = 1; proc_id < n_procs; ++proc_id) {
      int len;
      MPI_Status status;
      MPI_Recv(&len,
               1,
               MPI_INT,
               proc_id,
               SEND_LEN_TAG,
               MPI_COMM_WORLD,
               &status);
      std::vector<int64_t> tmp(len);
      MPI_Recv(tmp.data(),
               len,
               MPI_INT64_T,
               proc_id,
               SEND_ARRAY_TAG,
               MPI_COMM_WORLD,
               &status);

      std::copy(tmp.begin(), tmp.end(), &reversed[rem + (n_procs - status.MPI_SOURCE - 1) * per_proc]);
    }

    std::cout << "Reversed: " << std::endl;
    for (std::size_t i = 0u; i < N; ++i) {
      std::cout << reversed[i] << (i + 1 < N ? ' ' : '\n');
    }
  } else {
    int len;
    MPI_Status status;
    MPI_Recv(&len,
             1,
             MPI_INT,
             MASTER_PROC,
             SEND_LEN_TAG,
             MPI_COMM_WORLD,
             &status);
    std::vector<int64_t> tmp(len);
    MPI_Recv(tmp.data(),
             len,
             MPI_INT64_T,
             MASTER_PROC,
             SEND_ARRAY_TAG,
             MPI_COMM_WORLD,
             &status);
    std::reverse(tmp.begin(), tmp.end());
    MPI_Send(&len,
             1,
             MPI_INT,
             MASTER_PROC,
             SEND_LEN_TAG,
             MPI_COMM_WORLD);
    MPI_Send(tmp.data(),
             len,
             MPI_INT64_T,
             MASTER_PROC,
             SEND_ARRAY_TAG,
             MPI_COMM_WORLD);
  }

  MPI_Finalize();
}