//
// Created by aminjon on 3/22/23.
//
#include <iostream>
#include <omp.h>
#include <sstream>

int main() {
  for (const size_t threadCount : {3u, 2u}) {
    omp_set_num_threads(threadCount);
#pragma omp parallel if(omp_get_max_threads() > 2)
    {
      std::ostringstream toPrint;
      toPrint << "Threads count: " << omp_get_num_threads() << std::endl
              << "Current thread ID: " << omp_get_thread_num() << std::endl;
      std::cout << toPrint.str();
    }
  }
}