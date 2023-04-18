//
// Created by aminjon on 4/18/23.
//
#include <chrono>
#include <functional>
#include <iomanip>
#include <iostream>
#include <mpi.h>
#include <random>

inline constexpr int N = 30;
inline constexpr int PRECISION = 15;

inline constexpr int MASTER_PROC = 0;
inline constexpr int SEND_LEN_TAG = 0;
inline constexpr int SEND_ARRAY_PTR_TAG = 1;
inline constexpr int SEND_PARTIAL_TAG = 2;

inline constexpr int LB = 1;
inline constexpr int RB = 10;

inline constexpr auto ninf = std::numeric_limits<int>::min();

inline std::mt19937 rnd(std::chrono::steady_clock::now().time_since_epoch().count());
inline std::uniform_int_distribution<int> rnd_range(LB, RB);

inline std::array<int, N> array{}, recv_buf{};

typedef long double ld;

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);

  int n_procs;
  MPI_Comm_size(MPI_COMM_WORLD, &n_procs);
  int cur_proc;
  MPI_Comm_rank(MPI_COMM_WORLD, &cur_proc);

  const int per_proc = (N + n_procs - 1) / n_procs;
  std::vector<int> send_cnt(n_procs, per_proc), offsets(n_procs);

  if (cur_proc == MASTER_PROC) {
    for (int proc_id = 1; proc_id < n_procs; ++proc_id) {
      offsets[proc_id] = per_proc * (proc_id - 1);
    }
    send_cnt[0] = N - (n_procs - 1) * per_proc;
    offsets[0] = (n_procs - 1) * per_proc;

    [&]() {
      std::generate(array.begin(), array.end(), std::bind(rnd_range, std::ref(rnd)));
      std::cout << '[';
      for (std::size_t i = 0u; i < N; ++i) {
        std::cout << array[i];
        if (i + 1 < N) std::cout << ',';
      }
      std::cout << ']' << std::endl;
    }();
  }

  MPI_Scatterv(array.data(),
               send_cnt.data(),
               offsets.data(),
               MPI_INT,
               recv_buf.data(),
               N,
               MPI_INT,
               MASTER_PROC,
               MPI_COMM_WORLD);

  int loc = 0;
  for (int i = 0; i < send_cnt[cur_proc]; ++i) {
    loc += recv_buf[i];
  }

  int ans;
  if (cur_proc == MASTER_PROC) {
    ans = loc;
    for (int i = 1; i < n_procs; ++i) {
      int tmp;
      MPI_Status status;
      MPI_Recv(&tmp,
               1,
               MPI_INT,
               MPI_ANY_SOURCE,
               SEND_PARTIAL_TAG,
               MPI_COMM_WORLD,
               &status);
      ans += tmp;
    }
    std::cout << "Average: " << std::setprecision(PRECISION) << ld(ans) / N << std::endl;
  } else {
    MPI_Send(&loc,
             1,
             MPI_INT,
             MASTER_PROC,
             SEND_PARTIAL_TAG,
             MPI_COMM_WORLD);
  }

  MPI_Finalize();
}