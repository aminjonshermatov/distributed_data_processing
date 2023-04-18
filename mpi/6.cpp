//
// Created by aminjon on 4/18/23.
//
#include <iostream>
#include <random>
#include <chrono>
#include <functional>
#include <mpi.h>

inline constexpr int N = 9;
inline constexpr int M = 3;

inline constexpr int MASTER_PROC = 0;
inline constexpr int SEND_LEN_TAG = 0;
inline constexpr int SEND_ARRAY_PTR_TAG = 1;
inline constexpr int SEND_PARTIAL_MAX_TAG = 2;
inline constexpr int SEND_PARTIAL_MIN_TAG = 3;

inline constexpr int LB = 1;
inline constexpr int RB = 10;

inline constexpr auto  inf = std::numeric_limits<int64_t>::max();
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
  matrix<int64_t, N, M> mat{}, recv_buf{};

  MPI_Init(&argc, &argv);

  int n_procs;
  MPI_Comm_size(MPI_COMM_WORLD, &n_procs);
  int cur_proc;
  MPI_Comm_rank(MPI_COMM_WORLD, &cur_proc);

  const int per_proc = N / n_procs;
  int rem = N % n_procs;
  std::vector<int> send_cnt(n_procs, per_proc), offsets(n_procs);

  for (int proc_id = 0, offset = 0; proc_id < n_procs; ++proc_id) {
    offsets[proc_id] = offset;
    send_cnt[proc_id] = per_proc;
    if (rem > 0) {
      ++send_cnt[proc_id];
      --rem;
    }
    offset += send_cnt[proc_id];
  }
  for (int proc_id = 0; proc_id < n_procs; ++proc_id) {
    send_cnt[proc_id] *= M;
    offsets[proc_id] *= M;
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
    }();
  }

  MPI_Scatterv(mat.data(),
               send_cnt.data(),
               offsets.data(),
               MPI_INT64_T,
               recv_buf.data(),
               N * M,
               MPI_INT64_T,
               MASTER_PROC,
               MPI_COMM_WORLD);

  auto row_mx = ninf, row_mn = inf;
  for (int k = 0; k < send_cnt[cur_proc]; k += M) {
    auto loc_mx = ninf, loc_mn = inf;
    for (int j = 0; j < M && k + j < send_cnt[cur_proc]; ++j) {
      loc_mx = std::max(loc_mx, recv_buf.at(k / M, j));
      loc_mn = std::min(loc_mn, recv_buf.at(k / M, j));
    }
    row_mx = std::max(row_mx, loc_mn);
    row_mn = std::min(row_mn, loc_mx);
  }

  if (cur_proc == MASTER_PROC) {
    for (int other_proc = 1; other_proc < n_procs; ++other_proc) {
      int64_t tmp_mx, tmp_mn;
      MPI_Status status;

      MPI_Recv(&tmp_mx,
               1,
               MPI_INT64_T,
               MPI_ANY_SOURCE,
               SEND_PARTIAL_MAX_TAG,
               MPI_COMM_WORLD,
               &status);
      MPI_Recv(&tmp_mn,
               1,
               MPI_INT64_T,
               MPI_ANY_SOURCE,
               SEND_PARTIAL_MIN_TAG,
               MPI_COMM_WORLD,
               &status);
      row_mx = std::max(row_mx, tmp_mx);
      row_mn = std::max(row_mn, tmp_mn);
    }

    std::cout << "Max from min: " << row_mx << std::endl
              << "Min from max: " << row_mn << std::endl;
  } else {
    MPI_Send(&row_mx,
             1,
             MPI_INT64_T,
             MASTER_PROC,
             SEND_PARTIAL_MAX_TAG,
             MPI_COMM_WORLD);
    MPI_Send(&row_mn,
             1,
             MPI_INT64_T,
             MASTER_PROC,
             SEND_PARTIAL_MIN_TAG,
             MPI_COMM_WORLD);
  }

  MPI_Finalize();
}