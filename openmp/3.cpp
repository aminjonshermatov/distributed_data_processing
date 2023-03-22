//
// Created by aminjon on 3/22/23.
//
#include <iostream>
#include <omp.h>

int main() {
  std::int32_t a = 0, b = 0;
  std::cout << "Before, a: " << a << ' ' << "b: " << b << std::endl;

#pragma omp parallel num_threads(2) private(a) firstprivate(b)
  {
    a = 0;
    a += omp_get_thread_num();
    b += omp_get_thread_num();
    std::cout << "In parallel block, a: " << a << ' ' << "b: " << b << std::endl;
  }

  std::cout << "After, a: " << a << ' ' << "b: " << b << std::endl;

#pragma omp parallel num_threads(4) shared(a) private(b)
  {
    b = 0;
    a -= omp_get_thread_num();
    b -= omp_get_thread_num();
  }

  std::cout << "After, a: " << a << ' ' << "b: " << b << std::endl;
}