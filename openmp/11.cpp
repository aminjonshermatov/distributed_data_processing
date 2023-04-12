//
// Created by aminjon on 4/12/23.
//
#include <iostream>
#include <array>
#include <random>
#include <chrono>

std::mt19937 rnd(std::chrono::steady_clock::now().time_since_epoch().count());

inline constexpr std::size_t N = 30u;
inline constexpr std::int32_t DIVIDER = 9;

using array_t = std::array<std::int32_t, N>;

int main() {
  auto array = array_t{};
  std::generate(array.begin(), array.end(), rnd);

  std::uint32_t countMultiples = 0u;
#pragma omp parallel for num_threads(8) reduction(+: countMultiples)
  for (std::size_t i = 0u; i < N; ++i) {
    if (array[i] % DIVIDER == 0) {
#pragma omp critical
      ++countMultiples;
    }
  }

  std::cout << "Count of multiples of " << DIVIDER << ": " << countMultiples << std::endl;
}