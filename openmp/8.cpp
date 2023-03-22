//
// Created by aminjon on 3/23/23.
//
#include <iostream>
#include <numeric>
#include <array>
#include <omp.h>

inline constexpr std::size_t N = 16'000u;

int main() {
  std::array<std::size_t, N> A{};
  std::array<double, N> B{};

  std::iota(A.begin(), A.end(), 0u);

  const auto withMeasureTime = [](std::string_view label, const auto &&func) -> void {
    auto start = omp_get_wtime();
    func();
    std::cout << std::endl;
    auto overallTime = omp_get_wtime() - start;
    std::cout << label << ' ' << "time: " << overallTime << std::endl;
  };

  withMeasureTime("static", [&]() {
#pragma omp parallel for schedule(static) num_threads(8) shared(B)
    for (std::size_t i = 1u; i < N - 1; ++i) {
      B[i] = double(A[i - 1] + A[i] + A[i + 1]) / 3;
      std::cout << B[i] << ' ';
    }
  });

  withMeasureTime("dynamic", [&]() {
#pragma omp parallel for schedule(dynamic) num_threads(8) shared(B)
    for (std::size_t i = 1u; i < N - 1; ++i) {
      B[i] = double(A[i - 1] + A[i] + A[i + 1]) / 3;
      std::cout << B[i] << ' ';
    }
  });

  withMeasureTime("guided", [&]() {
#pragma omp parallel for schedule(guided) num_threads(8) shared(B)
    for (std::size_t i = 1u; i < N - 1; ++i) {
      B[i] = double(A[i - 1] + A[i] + A[i + 1]) / 3;
      std::cout << B[i] << ' ';
    }
  });

  withMeasureTime("auto", [&]() {
#pragma omp parallel for schedule(auto) num_threads(8) shared(B)
    for (std::size_t i = 1u; i < N - 1; ++i) {
      B[i] = double(A[i - 1] + A[i] + A[i + 1]) / 3;
      std::cout << B[i] << ' ';
    }
  });
}