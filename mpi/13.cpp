//
// Created by aminjon on 4/19/23.
//
#include <cassert>
#include <chrono>
#include <functional>
#include <iostream>
#include <mpi.h>
#include <random>
#include <vector>

inline constexpr std::size_t N = 10u;
inline constexpr int MASTER_PROC = 0;

inline constexpr int64_t LB = 1;
inline constexpr int64_t UB = 1;

inline std::mt19937_64 rnd_device(std::chrono::steady_clock::now().time_since_epoch().count());
inline std::uniform_int_distribution<int64_t> rnd_range(LB, UB);

template<typename T, std::size_t _nRows, std::size_t _nCols>
class matrix {
  T _data[_nRows * _nCols];

public:
  inline static constexpr auto nRows = _nRows;
  inline static constexpr auto nCols = _nCols;

  T &at(std::size_t i, std::size_t j) noexcept { return _data[i * nCols + j]; }
  T at(std::size_t i, std::size_t j) const noexcept { return _data[i * nCols + i]; }
  T &at(std::size_t i) noexcept { return _data[i]; }
  T at(std::size_t i) const noexcept { return _data[i]; }

  T *begin() noexcept { return _data; }
  T *end() noexcept { return _data + _nCols * _nRows; }

  T *data() noexcept { return _data; }
};

int main(int argc, char *argv[]) {
  static_assert(N % 2 == 0);

  MPI_Init(&argc, &argv);

  int n_procs;
  MPI_Comm_size(MPI_COMM_WORLD, &n_procs);
  int cur_proc;
  MPI_Comm_rank(MPI_COMM_WORLD, &cur_proc);

  matrix<int64_t, N, N> mat{}, pairwise{}, recv_buf{};
  std::vector<int> send_cnt(n_procs), offsets(n_procs);

  if (cur_proc == MASTER_PROC) {
    [&]() {
      std::generate(mat.begin(), mat.end(), std::bind(rnd_range, std::ref(rnd_device)));
      for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t j = 0; j < N; ++j) {
          std::cout << mat.at(i, j) << (j + 1 < N ? ' ' : '\n');
        }
      }
    }();

    const int per_proc = N * (N - 1) / 2 / n_procs;
    int rem = N * (N - 1) % (2 * n_procs);
    for (std::size_t proc_id = 0, offset = 0; proc_id < n_procs; ++proc_id) {
      send_cnt[proc_id] = per_proc * 2;
      offsets[proc_id] = offset;
      if (rem > 0) {
        assert(rem % 2 == 0);
        send_cnt[proc_id] += 2;
        rem -= 2;
      }
      offset += send_cnt[proc_id];
    }

    auto ptr = pairwise.data();
    for (std::size_t i = 0; i < N; ++i) {
      for (std::size_t j = i + 1; j < N; ++j) {
        *ptr = mat.at(i, j); ++ptr;
        *ptr = mat.at(j, i); ++ptr;
      }
    }
  }

  MPI_Scatterv(mat.data(),
               send_cnt.data(),
               offsets.data(),
               MPI_INT64_T,
               recv_buf.data(),
               N * (N - 1),
               MPI_INT64_T,
               MASTER_PROC,
               MPI_COMM_WORLD);

  int64_t loc_cnt = 0, global_cnt;
  for (std::size_t i = 0; i < send_cnt[cur_proc]; i += 2) {
    loc_cnt += recv_buf.at(i) == recv_buf.at(i + 1);
  }

  MPI_Reduce(&loc_cnt,
             &global_cnt,
             1,
             MPI_INT64_T,
             MPI_SUM,
             MASTER_PROC,
             MPI_COMM_WORLD);

  if (cur_proc == MASTER_PROC) {
    if (2 * global_cnt == N * (N - 1)) {
      std::cout << "Symmetrical" << std::endl;
    } else {
      std::cout << "Unsymmetrical" << std::endl;
    }
  }

  MPI_Finalize();
}