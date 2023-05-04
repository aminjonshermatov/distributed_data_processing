//
// Created by aminjon on 4/27/23.
//
#include <array>
#include <chrono>
#include <functional>
#include <iostream>
#include <random>

// #define PRINT_INITIAL_MATRIX
// #define PRINT_FINAL_MATRIX

constexpr int LB = 1;
constexpr int UB = 10;

std::random_device dev{};
std::mt19937 rnd(dev());
std::uniform_int_distribution<int> distribution(LB, UB);

constexpr int N = 2880;
constexpr int THREADS_PER_BLOCK = 16;// Each block have 16 * 16 = 256 threads
constexpr int BLOCKS_PER_SIDE = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
constexpr auto INF = std::numeric_limits<int>::max();

__global__ void work_gpu(int *adj, int k) {
  auto i = blockDim.y * blockIdx.y + threadIdx.y;
  auto j = blockDim.x * blockIdx.x + threadIdx.x;
  if (adj[i * N + k] == INF || adj[k * N + j] == INF) return;
  auto nDist = adj[i * N + k] + adj[k * N + j];
  if (i < N && j < N && (nDist < adj[i * N + j])) {
    adj[i * N + j] = nDist;
  }
}

int main() {
  cudaError_t err = cudaSuccess;

  int *hGraph = new int[N * N];
  [&]() {// init
    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < N; ++j) {
        hGraph[i * N + j] = i == j
                                ? 0
                            : rnd() % 2 == 0
                                ? INF
                                : distribution(rnd);
      }
    }

#ifdef PRINT_INITIAL_MATRIX
    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < N; ++j) {
        std::cout << (hGraph[i * N + j] == INF ? -1 : hGraph[i * N + j]) << (j + 1 < N ? ' ' : '\n');
      }
    }
#endif
  }();

  static_assert(N <= INF / N + INF % N);// overflow
  int *dGraph = nullptr;
  if ((err = cudaMalloc(&dGraph, N * N * sizeof(int))) != cudaSuccess) { std::cerr << "Failed while malloc dGraph_data: " << cudaGetErrorString(err) << std::endl; std::exit(EXIT_FAILURE); }
  if ((err = cudaMemcpy(dGraph, hGraph, N * N * sizeof(int), cudaMemcpyHostToDevice)) != cudaSuccess) { std::cerr << "Failed while memcpy hGraph: " << cudaGetErrorString(err) << std::endl; std::exit(EXIT_FAILURE); }

  dim3 blocks(BLOCKS_PER_SIDE, BLOCKS_PER_SIDE, 1);
  dim3 threadsPerBlock(THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1);

  auto start = std::chrono::steady_clock::now();
  for (int k = 0; k < N; ++k) {
    work_gpu<<<blocks, threadsPerBlock>>>(dGraph, k);
  }
  cudaDeviceSynchronize();
  auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();

  if ((err = cudaGetLastError()) != cudaSuccess) { std::cerr << "Failed while launch kernel: " << cudaGetErrorString(err) << std::endl; std::exit(EXIT_FAILURE); }

  if ((err = cudaMemcpy(hGraph, dGraph, N * N * sizeof(int), cudaMemcpyDeviceToHost)) != cudaSuccess) { std::cerr << "Failed while memcpy dGraph: " << cudaGetErrorString(err) << std::endl; std::exit(EXIT_FAILURE); }

#ifdef PRINT_FINAL_MATRIX
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      std::cout << (hGraph[i * N + j] == INF ? -1 : hGraph[i * N + j]) << (j + 1 < N ? ' ' : '\n');
    }
  }
#endif

  if ((err = cudaFree(dGraph)) != cudaSuccess) { std::cerr << "Failed while free dGraph: " << cudaGetErrorString(err) << std::endl; std::exit(EXIT_FAILURE); }

  std::cout << "Duration: " << dur << std::endl;
}