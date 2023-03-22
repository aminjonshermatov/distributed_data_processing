//
// Created by aminjon on 3/22/23.
//
#include <iostream>
#include <random>
#include <chrono>
#include <iomanip>
#include <array>
#include <omp.h>

inline constexpr std::size_t N = 6u;
inline constexpr std::size_t M = 8u;
inline constexpr std::int32_t MULTIPLE_OF = 3;
inline constexpr std::size_t PRECISION = 10u;

std::mt19937 rnd(std::chrono::steady_clock::now().time_since_epoch().count());

int main() {
  std::array<std::array<std::int32_t, M>, N> matrix{};
  for (auto &row : matrix) {
    for (auto &element : row) {
      element = static_cast<std::int32_t>(rnd());
    }
  }

#pragma omp parallel sections
  {
#pragma omp section
    {
      auto sum = std::accumulate(matrix.cbegin(), matrix.cend(), 0, [](auto acc, const auto& row) {
        return std::accumulate(row.cbegin(), row.cend(), acc);
      });
      std::ostringstream toPrint;
      toPrint << "Current thread ID: " << omp_get_thread_num() << std::endl
              << "Average: " << std::fixed << std::setprecision(PRECISION) << sum / double(N * M) << std::endl;
      std::cout << toPrint.str();
    }
#pragma omp section
    {
      auto minElement = std::numeric_limits<std::int32_t>::max();
      auto maxElement = std::numeric_limits<std::int32_t>::min();

      for (const auto &row : matrix) {
        for (const auto &element : row) {
          minElement = std::min(minElement, element);
          maxElement = std::max(maxElement, element);
        }
      }
      std::ostringstream toPrint;
      toPrint << "Current thread ID: " << omp_get_thread_num() << std::endl
              << "Min: " << minElement << std::endl
              << "Max: " << maxElement << std::endl;
      std::cout << toPrint.str();
    }
#pragma omp section
    {
      std::size_t countOfMultiple = 0u;
      for (const auto &row : matrix) {
        for (const auto &element : row) {
          if (element % MULTIPLE_OF == 0) {
            ++countOfMultiple;
          }
        }
      }
      std::ostringstream toPrint;
      toPrint << "Current thread ID: " << omp_get_thread_num() << std::endl
              << "Count of element than multiple of " << MULTIPLE_OF << ": " << countOfMultiple << std::endl;
      std::cout << toPrint.str();
    }
  }
}