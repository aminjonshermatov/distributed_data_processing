//
// Created by aminjon on 4/27/23.
//
#include <iostream>
#include <random>
#include <array>
#include <iomanip>
#include <functional>

__global__ void vecAdd(const int *A, const int *B, int *C, int len) {
  auto i =blockDim.x * blockIdx.x + threadIdx.x;
  if (i < len) {
    C[i] = A[i] + B[i];
  }
}

constexpr int LB = 1;
constexpr int UB = 10;

std::random_device dev{};
std::mt19937 rnd(dev());
std::uniform_int_distribution<int> distribution(LB, UB);

constexpr int N = 50000;

int main() {
  cudaError_t err = cudaSuccess;

  std::array<int, N> hA{}, hB{}, hC{};
  [&](){ // init
    std::generate(hA.begin(), hA.end(), std::bind(distribution, std::ref(rnd)));
    std::generate(hB.begin(), hB.end(), std::bind(distribution, std::ref(rnd)));

    std::cout << "N: " << N << std::endl;
    for (int i = 0; i < N; ++i) {
      std::cout << std::setw(2) << std::setfill('0') << hA[i] << (i + 1 < N ? ' ' : '\n');
    }
    for (int i = 0; i < N; ++i) {
      std::cout << std::setw(2) << std::setfill('0') << hB[i] << (i + 1 < N ? ' ' : '\n');
    }
  }();

  int *dA = nullptr, *dB = nullptr, *dC = nullptr;
  if ((err = cudaMalloc(&dA, N * sizeof(int))) != cudaSuccess) { std::cerr << "Failed while malloc A: " << cudaGetErrorString(err) << std::endl; }
  if ((err = cudaMalloc(&dB, N * sizeof(int))) != cudaSuccess) { std::cerr << "Failed while malloc B: " << cudaGetErrorString(err) << std::endl; }
  if ((err = cudaMalloc(&dC, N * sizeof(int))) != cudaSuccess) { std::cerr << "Failed while malloc C: " << cudaGetErrorString(err) << std::endl; }

  if ((err = cudaMemcpy(dA, hA.data(), N * sizeof(int), cudaMemcpyHostToDevice)) != cudaSuccess) { std::cerr << "Failed while memcpy A: " << cudaGetErrorString(err) << std::endl; }
  if ((err = cudaMemcpy(dB, hB.data(), N * sizeof(int), cudaMemcpyHostToDevice)) != cudaSuccess) { std::cerr << "Failed while memcpy B: " << cudaGetErrorString(err) << std::endl; }

  int threadsPerBlock = 256;
  int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
  std::cout << "threadsPerBlock: " << threadsPerBlock << " blocksPerGrid: " << blocksPerGrid << std::endl;

  vecAdd<<<blocksPerGrid, threadsPerBlock>>>(dA, dB, dC, N);
  if ((err = cudaGetLastError()) != cudaSuccess) { std::cerr << "Failed while launch kernel: " << cudaGetErrorString(err) << std::endl; }

  if ((err = cudaMemcpy(hC.data(), dC, N * sizeof(int), cudaMemcpyDeviceToHost)) != cudaSuccess) { std::cerr << "Failed while memcpy C: " << cudaGetErrorString(err) << std::endl; }

  for (int i = 0; i < N; ++i) {
    std::cout << std::setw(2) << std::setfill('0') << hC[i] << (i + 1 < N ? ' ' : '\n');
  }

  if ((err = cudaFree(dA)) != cudaSuccess) { std::cerr << "Failed while free A: " << cudaGetErrorString(err) << std::endl; }
  if ((err = cudaFree(dB)) != cudaSuccess) { std::cerr << "Failed while free B: " << cudaGetErrorString(err) << std::endl; }
  if ((err = cudaFree(dC)) != cudaSuccess) { std::cerr << "Failed while free C: " << cudaGetErrorString(err) << std::endl; }

}