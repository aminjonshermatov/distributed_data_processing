//
// Created by aminjon on 3/22/23.
//
#include <iostream>
#include <omp.h>
#include <sstream>

inline constexpr std::size_t THREAD_NUM = 8u;

int main() {
  omp_set_num_threads(THREAD_NUM);
#pragma omp parallel
  {
    std::ostringstream toPrint;
    toPrint << "Threads count: " << omp_get_num_threads() << std::endl
            << "Current thread ID: " << omp_get_thread_num() << std::endl
            << "Hello World" << std::endl;

    std::cout << toPrint.str();
  }
}