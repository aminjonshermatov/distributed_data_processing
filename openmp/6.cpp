//
// Created by aminjon on 3/22/23.
//
#include <iostream>
#include <array>
#include <random>
#include <chrono>

inline constexpr std::size_t N = 100u;

std::mt19937 rnd(std::chrono::steady_clock::now().time_since_epoch().count());

int main() {
  std::array<std::int32_t, N> A{};
  //for (int i = 0; i < N; i += 2) A[i] = 1;
  std::generate(A.begin(), A.end(), rnd);

  std::int64_t sumWithReduction = 0, sumWithoutReduction = 0;
#pragma omp parallel for reduction(+: sumWithReduction)
  for (const auto num : A) sumWithReduction += num;

#pragma omp parallel for
  for (std::size_t i = 0u; i < N; ++i) {
//#pragma omp atomic
    sumWithoutReduction += A[i];
  }

  std::cout << sumWithReduction << std::endl
            << sumWithoutReduction << std::endl;
}