//
// Created by aminjon on 4/12/23.
//
#include <iostream>
#include <array>
#include <random>
#include <chrono>
#include <omp.h>

std::mt19937 rnd(std::chrono::steady_clock::now().time_since_epoch().count());

inline constexpr std::size_t N = 10u;
inline constexpr std::int32_t DIVIDER = 7;

using array_t = std::array<std::int32_t, N>;

inline constexpr auto ninf = std::numeric_limits<std::int32_t>::min();

int main() {
  omp_lock_t lock{};
  omp_init_lock(&lock);

  auto array = array_t{};
  std::generate(array.begin(), array.end(), rnd);

  std::int32_t mx = ninf;
#pragma omp parallel for num_threads(8) shared(mx)
  for (std::size_t i = 0u; i < N; ++i) {
    if (array[i] % DIVIDER != 0) continue ;
    omp_set_lock(&lock);
    if (array[i] > mx) {
      mx = array[i];
    }
    omp_unset_lock(&lock);
  }

  if (mx == ninf) {
    std::cout << "There is no element that multiples of " << DIVIDER << std::endl;
  } else {
    std::cout << "Maximum element that multiples of " << DIVIDER << ": " << mx << std::endl;
  }
  omp_destroy_lock(&lock);
}