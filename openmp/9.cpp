//
// Created by aminjon on 3/23/23.
//
#include <array>
#include <random>
#include <chrono>
#include <iostream>
#include <omp.h>

inline constexpr std::size_t N = 100u;

template<std::size_t N = N, std::size_t M = N>
using matrix_t = std::array<std::array<std::int32_t, M>, N>;
template<std::size_t N = N>
using vector_t = std::array<std::int32_t, N>;

std::mt19937 rnd(std::chrono::steady_clock::now().time_since_epoch().count());

int main() {
  const auto generateMatrix = []() -> matrix_t<> {
    matrix_t<> mat{};
    for (auto &row : mat) {
      for (auto &element : row) {
        element = static_cast<std::int32_t>(rnd());
      }
    }
    return mat;
  };
  const auto generateVector = []() -> vector_t<> {
    vector_t<> vec{};
    for (auto &element : vec) {
      element = static_cast<std::int32_t>(rnd());
    }
    return vec;
  };

  const auto withMeasureTime = [](std::string_view label, const auto &&func) -> void {
    auto start = omp_get_wtime();
    func();
    std::cout << std::endl;
    auto overallTime = omp_get_wtime() - start;
    std::cout << label << ' ' << "time: " << overallTime << std::endl;
  };

  auto mat = generateMatrix();
  auto vec = generateVector();

  withMeasureTime("sequentially", [&]() {
    vector_t<> multiplied_vector{};
    std::fill(vec.begin(), vec.end(), 0);

    for (std::size_t i = 0u; i < N; ++i) {
      for (std::size_t j = 0u; j < N; ++j) {
        multiplied_vector[i] += mat[i][j] * vec[j];
      }
    }
  });

  withMeasureTime("static", [&]() {
    vector_t<> multiplied_vector{};
    std::fill(vec.begin(), vec.end(), 0);
#pragma omp parallel for schedule(static) num_threads(8)
    for (std::size_t i = 0u; i < N; ++i) {
      for (std::size_t j = 0u; j < N; ++j) {
        multiplied_vector[i] += mat[i][j] * vec[j];
      }
    }
  });

  withMeasureTime("dynamic", [&]() {
    vector_t<> multiplied_vector{};
    std::fill(vec.begin(), vec.end(), 0);
#pragma omp parallel for schedule(dynamic) num_threads(8)
    for (std::size_t i = 0u; i < N; ++i) {
      for (std::size_t j = 0u; j < N; ++j) {
        multiplied_vector[i] += mat[i][j] * vec[j];
      }
    }
  });

  withMeasureTime("guided", [&]() {
    vector_t<> multiplied_vector{};
    std::fill(vec.begin(), vec.end(), 0);
#pragma omp parallel for schedule(guided) num_threads(8)
    for (std::size_t i = 0u; i < N; ++i) {
      for (std::size_t j = 0u; j < N; ++j) {
        multiplied_vector[i] += mat[i][j] * vec[j];
      }
    }
  });

  withMeasureTime("auto", [&]() {
    vector_t<> multiplied_vector{};
    std::fill(vec.begin(), vec.end(), 0);
#pragma omp parallel for schedule(auto) num_threads(8)
    for (std::size_t i = 0u; i < N; ++i) {
      for (std::size_t j = 0u; j < N; ++j) {
        multiplied_vector[i] += mat[i][j] * vec[j];
      }
    }
  });
}