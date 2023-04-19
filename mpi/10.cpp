//
// Created by aminjon on 4/19/23.
//
#include <cassert>
#include <chrono>
#include <functional>
#include <iostream>
#include <mpi.h>
#include <random>

inline constexpr std::size_t N = 1'000u;
inline constexpr int SRC_PROC = 0;
inline constexpr int DST_PROC = 1;
inline constexpr int SEND_ARRAY_TO_DST_TAG = 0;
inline constexpr int SEND_ARRAY_TO_SRC_TAG = 1;

inline constexpr int LB = -100;
inline constexpr int UB =  100;

inline std::mt19937 rnd_device(std::chrono::steady_clock::now().time_since_epoch().count());
inline std::uniform_int_distribution<int> rnd_range(LB, UB);

inline auto fill_rnd(std::array<int, N> &arr) {
  std::generate(arr.begin(), arr.end(), std::bind(rnd_range, rnd_device));
}

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);

  int n_procs;
  MPI_Comm_size(MPI_COMM_WORLD, &n_procs);
  int cur_proc;
  MPI_Comm_rank(MPI_COMM_WORLD, &cur_proc);

  assert(n_procs == 2);

  const auto measureTime = [&cur_proc](std::string_view label, auto &&sendFunc) {
    std::array<int, N> A{}, recv_buf{};
    if (label == "MPI_Bsend") {
      MPI_Buffer_attach(A.data(), N * sizeof(int));
      MPI_Buffer_attach(recv_buf  .data(), N * sizeof(int));
    }
    fill_rnd(A);
    double start;
    if (cur_proc == SRC_PROC) {
      start = MPI_Wtime();
      sendFunc(A.data(),
               N,
               MPI_INT,
               DST_PROC,
               SEND_ARRAY_TO_DST_TAG,
               MPI_COMM_WORLD);
    } else {
      MPI_Status status;
      MPI_Recv(recv_buf.data(),
               N,
               MPI_INT,
               SRC_PROC,
               SEND_ARRAY_TO_DST_TAG,
               MPI_COMM_WORLD,
               &status);
      sendFunc(recv_buf.data(),
               N,
               MPI_INT,
               SRC_PROC,
               SEND_ARRAY_TO_SRC_TAG,
               MPI_COMM_WORLD);
    }
    if (cur_proc == SRC_PROC) {
      MPI_Status status;
      MPI_Recv(recv_buf.data(),
               N,
               MPI_INT,
               DST_PROC,
               SEND_ARRAY_TO_SRC_TAG,
               MPI_COMM_WORLD,
               &status);
      std::cout << label << ' ' << MPI_Wtime() - start << " seconds" << std::endl;
    }
  };

#define FUNC_NAME_TO_STR(f) #f
  measureTime(FUNC_NAME_TO_STR(MPI_Send), MPI_Send);
  measureTime(FUNC_NAME_TO_STR(MPI_Ssend), MPI_Ssend);
  measureTime(FUNC_NAME_TO_STR(MPI_Bsend), MPI_Bsend);
  measureTime(FUNC_NAME_TO_STR(MPI_Rsend), MPI_Rsend);

  MPI_Finalize();
}