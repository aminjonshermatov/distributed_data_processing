//
// Created by aminjon on 4/18/23.
//
#include <iostream>
#include <random>
#include <chrono>
#include <functional>
#include <mpi.h>

inline constexpr int N = 2;
inline constexpr int M = 5;

inline constexpr int MASTER_PROC = 0;
inline constexpr int SEND_LEN_TAG = 0;
inline constexpr int SEND_ARRAY_PTR_TAG = 1;
inline constexpr int SEND_PARTIAL_TAG = 2;

inline constexpr int LB = 1;
inline constexpr int RB = 10;

inline constexpr auto ninf = std::numeric_limits<int64_t>::min();

inline std::mt19937_64 rnd(std::chrono::steady_clock::now().time_since_epoch().count());
inline std::uniform_int_distribution<int64_t> rnd_range(LB, RB);

template <typename T, std::size_t nRows, std::size_t nCols> class matrix {
  T _data[nRows * nCols];
public:

  T& at(std::size_t i, std::size_t j) noexcept { return _data[i * M + j]; }
  T at(std::size_t i, std::size_t j) const noexcept { return _data[i * M + j]; }
  T& at(std::size_t i) noexcept { return _data[i]; }
  T at(std::size_t i) const noexcept { return _data[i]; }

  T* data() noexcept { return _data; }
};

int main(int argc, char *argv[]) {
  matrix<int64_t, N, M> mat{};

  MPI_Init(&argc, &argv);

  int n_procs;
  MPI_Comm_size(MPI_COMM_WORLD, &n_procs);
  int cur_proc;
  MPI_Comm_rank(MPI_COMM_WORLD, &cur_proc);

  if (cur_proc == MASTER_PROC) {
    [&](){
      for (std::size_t i = 0u; i < N; ++i) {
        for (std::size_t j = 0u; j < M; ++j) {
          mat.at(i, j) = rnd_range(rnd);
        }
      }
      for (std::size_t i = 0u; i < N; ++i) {
        for (std::size_t j = 0u; j < M; ++j) {
          std::cout << mat.at(i, j) << (j + 1 < M ? ' ' : '\n');
        }
      }
    }();

    const std::uint32_t per_proc = (N * M + n_procs - 1) / n_procs;
    std::uint32_t used = 0;
    for (int proc_id = 1; proc_id < n_procs; ++proc_id) {
      MPI_Send(&per_proc,
               1,
               MPI_UINT32_T,
               proc_id,
               SEND_LEN_TAG,
               MPI_COMM_WORLD);
      MPI_Send(mat.data() + per_proc * (proc_id - 1),
               per_proc,
               MPI_INT64_T,
               proc_id,
               SEND_ARRAY_PTR_TAG,
               MPI_COMM_WORLD);
      used += per_proc;
    }

    int64_t mx = ninf;
    for (std::size_t i = used; i < N * M; ++i) {
      mx = std::max(mx, mat.at(i));
    }

    for (int proc_id = 1; proc_id < n_procs; ++proc_id) {
      int64_t partial;
      MPI_Status status;
      MPI_Recv(&partial,
               1,
               MPI_INT64_T,
               MPI_ANY_SOURCE,
               SEND_PARTIAL_TAG,
               MPI_COMM_WORLD,
               &status);
      mx = std::max(mx, partial);
    }

    std::cout << "Max: " << mx << std::endl;
  } else {
    uint32_t len;
    MPI_Status status;
    MPI_Recv(&len,
             1,
             MPI_UINT32_T,
             MASTER_PROC,
             SEND_LEN_TAG,
             MPI_COMM_WORLD,
             &status);
    std::vector<int64_t> tmp(len);
    MPI_Recv(tmp.data(),
             len,
             MPI_INT64_T,
             MASTER_PROC,
             SEND_ARRAY_PTR_TAG,
             MPI_COMM_WORLD,
             &status);

    int64_t partial = ninf;
    for (int i = 0; i < len; ++i) {
      partial = std::max(partial, tmp[i]);
    }

    MPI_Send(&partial,
             1,
             MPI_INT64_T,
             MASTER_PROC,
             SEND_PARTIAL_TAG,
             MPI_COMM_WORLD);
  }

  MPI_Finalize();
}