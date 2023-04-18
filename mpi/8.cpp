//
// Created by aminjon on 4/19/23.
//
#include <chrono>
#include <functional>
#include <iostream>
#include <mpi.h>
#include <random>

inline constexpr int N = 13;

inline constexpr int MASTER_PROC = 0;
inline constexpr int SEND_LEN_TAG = 0;
inline constexpr int SEND_ARRAY_TAG = 0;

inline constexpr int LB = 1;
inline constexpr int RB = 100;

inline std::mt19937_64 rnd(std::chrono::steady_clock::now().time_since_epoch().count());
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
        array[i] = rnd_range(rnd);
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

    std::cout << "# " << cur_proc << std::endl;
    for (std::size_t i = used; i < N; ++i) {
      std::cout << array[i] << (i + 1 < N ? ' ' : '\n');
    }

    std::array<int64_t, N> res{};
    auto ptr = res.data();
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

      for (auto x: tmp) {
        *ptr = x;
        ++ptr;
      }
    }

    for (std::size_t i = used; i < N; ++i) {
      *ptr = array[i];
      ++ptr;
    }

    std::cout << "Result: " << std::endl;
    for (std::size_t i = 0u; i < N; ++i) {
      std::cout << res[i] << (i + 1 < N ? ' ' : '\n');
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

    std::cout << "# " << cur_proc << std::endl;
    for (std::size_t i = 0u; i < len; ++i) {
      std::cout << tmp[i] << (i + 1 < len ? ' ' : '\n');
    }
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