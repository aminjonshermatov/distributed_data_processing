//
// Created by aminjon on 4/26/23.
//
#include <iostream>
#include <mpi.h>
#include <cassert>
#include <vector>
#include <random>

constexpr int MASTER_PROC = 0;
constexpr int N = 16;
constexpr int LB = 1;
constexpr int UB = 10;
constexpr auto INF = std::numeric_limits<int>::max();

std::random_device dev{};
std::mt19937 rnd(dev());
std::uniform_int_distribution<std::mt19937::result_type> dist(LB, UB);

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);

  int n_procs;
  MPI_Comm_size(MPI_COMM_WORLD, &n_procs);
  int cur_proc;
  MPI_Comm_rank(MPI_COMM_WORLD, &cur_proc);

  assert(N * N % n_procs == 0);

  const auto per_proc = N * N / n_procs;

  std::vector<int> mat, loc(per_proc);

  auto print_mat = [&mat](){
    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < N; ++j) {
        std::cout << (mat[i * N + j] == INF ? -1 : mat[i * N + j]) << (j + 1 < N ? ' ' : '\n');
      }
    }
  };

  if (cur_proc == MASTER_PROC) {
    mat.resize(N * N);
    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < N; ++j) {
        if (i == j) {
          mat[i * N + j] = 0;
        } else {
          auto d = dist(rnd);
          if (d % 2 == rnd() % 2) {
            mat[i * N + j] = d;
          } else {
            mat[i * N + j] = INF;
          }
        }
      }
    }
    print_mat();
  }
  MPI_Scatter(mat.data(), per_proc, MPI_INT, loc.data(), per_proc, MPI_INT, MASTER_PROC, MPI_COMM_WORLD);

  std::vector<int> kRow(N);
  auto section = N / n_procs;

  auto cast_row = [&](int k) -> void {
    auto [head, locK] = div(k, section);
    if (cur_proc == head) {
      for (int j = 0; j < N; ++j) {
        kRow[j] = loc[locK * N + j];
      }
    }
    MPI_Bcast(kRow.data(), N, MPI_INT, cur_proc, MPI_COMM_WORLD);
  };

  for (int k = 0; k < N; ++k) {
    cast_row(k);
    for (int i = 0; i < section; ++i) {
      for (int j = 0; j < N; ++j) {
        if (loc[i * N + k] == INF || kRow[j] == INF) continue;
        auto tmp = loc[i * N + k] + kRow[j];
        if (tmp < loc[i * N + j]) {
          loc[i * N + j] = tmp;
        }
      }
    }
  }

  MPI_Gather(loc.data(), per_proc, MPI_INT, mat.data(), per_proc, MPI_INT, MASTER_PROC, MPI_COMM_WORLD);

  if (cur_proc == MASTER_PROC) {
    std::cout << "Result:" << std::endl;
    print_mat();
  }

  MPI_Finalize();
}