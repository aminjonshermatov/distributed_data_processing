//
// Created by aminjon on 4/18/23.
//
#include <iostream>
#include <random>
#include <chrono>
#include <functional>
#include <mpi.h>

inline constexpr int N = 30;

inline constexpr int MASTER_PROC = 0;
inline constexpr int SEND_LEN_TAG = 0;
inline constexpr int SEND_ARRAY_PTR_TAG = 1;
inline constexpr int SEND_PARTIAL_TAG = 2;

inline constexpr int LB = 1;
inline constexpr int RB = 10;

inline constexpr auto ninf = std::numeric_limits<int>::min();

inline std::mt19937 rnd(std::chrono::steady_clock::now().time_since_epoch().count());
inline std::uniform_int_distribution<int> rnd_range(LB, RB);

inline std::array<int, N> A{}, B{};

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);

  int n_procs;
  MPI_Comm_size(MPI_COMM_WORLD, &n_procs);
  int cur_proc;
  MPI_Comm_rank(MPI_COMM_WORLD, &cur_proc);

  if (cur_proc == MASTER_PROC) {
    [&](){
      const auto gen_print = [](std::array<int, N> &arr) {
        std::generate(arr.begin(), arr.end(), std::bind(rnd_range, std::ref(rnd)));
        std::cout << '[';
        for (std::size_t i = 0u; i < N; ++i) {
          std::cout << arr[i];
          if (i + 1 < N) std::cout << ',';
        }
        std::cout << ']' << std::endl;
      };
      gen_print(A); gen_print(B);
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
      MPI_Send(A.data() + per_proc * (proc_id - 1),
               per_proc,
               MPI_INT,
               proc_id,
               SEND_ARRAY_PTR_TAG,
               MPI_COMM_WORLD);
      MPI_Send(B.data() + per_proc * (proc_id - 1),
               per_proc,
               MPI_INT,
               proc_id,
               SEND_ARRAY_PTR_TAG,
               MPI_COMM_WORLD);
      used += per_proc;
    }

    int64_t s = 0;
    for (int i = used; i < N; ++i) {
      s += A[i] * B[i];
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
      s += partial;
    }

    std::cout << "Scalar: " << s << std::endl;
  } else {
    int len;
    std::array<int, N> tmpA{}, tmpB{};
    MPI_Status status;
    MPI_Recv(&len,
             1,
             MPI_INT,
             MASTER_PROC,
             SEND_LEN_TAG,
             MPI_COMM_WORLD,
             &status);
    MPI_Recv(tmpA.data(),
             len,
             MPI_INT,
             MASTER_PROC,
             SEND_ARRAY_PTR_TAG,
             MPI_COMM_WORLD,
             &status);
    MPI_Recv(tmpB.data(),
             len,
             MPI_INT,
             MASTER_PROC,
             SEND_ARRAY_PTR_TAG,
             MPI_COMM_WORLD,
             &status);

    int64_t partial = 0;
    for (int i = 0; i < len; ++i) {
      partial += tmpA[i] * tmpB[i];
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