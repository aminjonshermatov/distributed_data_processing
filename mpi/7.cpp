//
// Created by aminjon on 4/18/23.
//
#include <iostream>
#include <random>
#include <chrono>
#include <functional>
#include <mpi.h>

inline constexpr int N = 13;
inline constexpr int M = 37;

inline constexpr int MASTER_PROC = 0;
inline constexpr int SEND_RESULT_TAG = 0;

inline constexpr int LB = 1;
inline constexpr int RB = 100;

inline std::mt19937_64 rnd(std::chrono::steady_clock::now().time_since_epoch().count());
inline std::uniform_int_distribution<int64_t> rnd_range(LB, RB);

/*
[[0,3,6]
,[1,4,7],
,[2,5,8]]
*/
template <typename T, std::size_t _nRows, std::size_t _nCols> class matrix {
  T _data[_nRows * _nCols];
public:
  inline static constexpr auto nRows = _nRows;
  inline static constexpr auto nCols = _nCols;

  T& at(std::size_t i, std::size_t j) noexcept { return _data[j * N + i]; }
  T at(std::size_t i, std::size_t j) const noexcept { return _data[j * N + i]; }
  T& at(std::size_t i) noexcept { return _data[i]; }
  T at(std::size_t i) const noexcept { return _data[i]; }

  T* data() noexcept { return _data; }
};
template <typename T, std::size_t nCols> using vector = std::array<T, nCols>;

int main(int argc, char *argv[]) {
  matrix<int64_t, N, M> mat{}, recv_mat{};
  vector<int64_t, M> vec{}, recv_vec{};
  static_assert(vec.size() == mat.nCols);

  MPI_Init(&argc, &argv);

  int n_procs;
  MPI_Comm_size(MPI_COMM_WORLD, &n_procs);
  int cur_proc;
  MPI_Comm_rank(MPI_COMM_WORLD, &cur_proc);

  const int per_proc = M / n_procs;
  int rem = M % n_procs;
  std::vector<int> mat_cnt(n_procs), mat_offset(n_procs), vec_cnt(n_procs), vec_offset(n_procs);

  for (int proc_id = 0, offset = 0; proc_id < n_procs; ++proc_id) {
    mat_offset[proc_id] = offset;
    vec_offset[proc_id] = offset;
    mat_cnt[proc_id] = per_proc;
    vec_cnt[proc_id] = per_proc;
    if (rem > 0) {
      ++mat_cnt[proc_id];
      ++vec_cnt[proc_id];
      --rem;
    }
    offset += mat_cnt[proc_id];
  }
  for (int proc_id = 0; proc_id < n_procs; ++proc_id) {
    mat_cnt[proc_id] *= N;
    mat_offset[proc_id] *= N;
  }
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
      std::cout << std::endl;
      for (std::size_t i = 0u; i < M; ++i) {
        vec.at(i) = rnd_range(rnd);
      }
      for (std::size_t i = 0u; i < M; ++i) {
        std::cout << vec.at(i) << (i + 1 < M ? ' ' : '\n');
      }
    }();
  }

  MPI_Scatterv(mat.data(),
               mat_cnt.data(),
               mat_offset.data(),
               MPI_INT64_T,
               recv_mat.data(),
               N * M,
               MPI_INT64_T,
               MASTER_PROC,
               MPI_COMM_WORLD);
  MPI_Scatterv(vec.data(),
               vec_cnt.data(),
               vec_offset.data(),
               MPI_INT64_T,
               recv_vec.data(),
               M,
               MPI_INT64_T,
               MASTER_PROC,
               MPI_COMM_WORLD);

  vector<int64_t, N> res{};
  std::fill(res.begin(), res.end(), 0);
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < vec_cnt[cur_proc]; ++j) {
      res[i] += recv_mat.at(i, j) * recv_vec.at(j);
    }
  }

  if (cur_proc == MASTER_PROC) {
    for (int other_proc = 1; other_proc < n_procs; ++other_proc) {
      vector<int64_t, N> tmp{};
      std::fill(tmp.begin(), tmp.end(), 0);
      MPI_Status status;

      MPI_Recv(tmp.data(),
               N,
               MPI_INT64_T,
               MPI_ANY_SOURCE,
               SEND_RESULT_TAG,
               MPI_COMM_WORLD,
               &status);

      for (size_t i = 0; i < N; ++i) {
        res[i] += tmp[i];
      }
    }

    for (size_t i = 0; i < N; ++i) {
      std::cout << res[i] << (i + 1 < N ? ' ' : '\n');
    }
  } else {
    MPI_Send(res.begin(),
             N,
             MPI_INT64_T,
             MASTER_PROC,
             SEND_RESULT_TAG,
             MPI_COMM_WORLD);
  }

  MPI_Finalize();
}