//
// Created by aminjon on 4/12/23.
//
#include <iostream>
#include <random>
#include <chrono>
#include <array>

std::mt19937 rnd(std::chrono::steady_clock::now().time_since_epoch().count());

inline constexpr std::size_t N = 6u;
inline constexpr std::size_t M = 8u;

inline constexpr auto  inf = std::numeric_limits<std::int32_t>::max();
inline constexpr auto ninf = std::numeric_limits<std::int32_t>::min();

using array_t = std::array<std::array<std::int32_t, M>, N>;

int main() {
  auto array = array_t{};
  std::for_each(array.begin(), array.end(), [](auto &row) -> void {
    std::generate(row.begin(), row.end(), rnd);
  });

  auto mn = inf;
  auto mx = ninf;

#pragma omp parallel num_threads(8) reduction(min:mn) reduction(max:mx)
  for (std::size_t i = 0u; i < N; ++i) {
    for (std::size_t j = 0u; j < M; ++j) {
      if (mn > array[i][j]) {
#pragma omp critical
        mn = array[i][j];
      }

      if (mx < array[i][j]) {
#pragma omp critical
        mx = array[i][j];
      }
    }
  }

  std::cout << "Minimum value: " << mn << std::endl
            << "Maximum value: " << mx << std::endl;
}