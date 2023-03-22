//
// Created by aminjon on 3/23/23.
//
#include <iostream>
#include <sstream>
#include <array>
#include <omp.h>

inline constexpr std::size_t N = 12u;

int main() {
  std::array<std::size_t, N> A{}, B{}, C{};

#pragma omp parallel num_threads(3)
  {
#pragma omp for schedule(static)
    for (std::size_t i = 0u; i < N; ++i) {
      A[i] = i;
      B[i] = 2 * i;
      std::ostringstream toPrint;
      toPrint << "Current thread: " << omp_get_thread_num() << std::endl
              << "Overall count: " << omp_get_num_threads() << std::endl;
      std::cout << toPrint.str();
    }
  }

  omp_set_num_threads(4);
#pragma parallel omp for schedule(dynamic, 3)
  for (std::size_t i = 0u; i < N; ++i) {
    C[i] = A[i] + B[i];
    std::ostringstream toPrint;
    toPrint << "Current thread: " << omp_get_thread_num() << std::endl
            << "Overall count: " << omp_get_num_threads() << std::endl
            << "A[" << i << ']' << '+' << "B[" << i << "]=" << C[i] << std::endl;
    std::cout << toPrint.str();
  }

}