//
// Created by aminjon on 3/22/23.
//
#include <array>
#include <cassert>
#include <chrono>
#include <iostream>
#include <random>

inline constexpr std::size_t ARRAY_LENGTH = 10u;

std::mt19937 rnd(std::chrono::steady_clock::now().time_since_epoch().count());

int main() {
  std::array<std::int32_t, ARRAY_LENGTH> A{}, B{};
  std::generate(A.begin(), A.end(), rnd);
  std::generate(B.begin(), B.end(), rnd);

  auto maxFromA = std::numeric_limits<std::int32_t>::min();
  auto minFromB = std::numeric_limits<std::int32_t>::max();

#pragma omp parallel sections num_threads(2)
  {
#pragma omp section
    {
      for (const auto num : A) {
        maxFromA = std::max(maxFromA, num);
      }
    }
#pragma omp section
    {
      for (const auto num : B) {
        minFromB = std::min(minFromB, num);
      }
    }
  }

  auto expectedMaxFromA = *std::max_element(A.begin(), A.end());
  assert(expectedMaxFromA == maxFromA);

  auto expectedMinFromB = *std::min_element(B.begin(), B.end());
  assert(expectedMinFromB == minFromB);

  std::cout << "Max from A: " << maxFromA << std::endl
            << "Min from B: " << minFromB << std::endl;
}