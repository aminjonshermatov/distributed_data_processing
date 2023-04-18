//
// Created by aminjon on 4/18/23.
//
#include <iostream>
#include <random>
#include <chrono>
#include <mpi.h>

inline constexpr int64_t N = 100'000'000;
inline constexpr int64_t R = 10'000;

inline constexpr int MASTER_PROC = 0;
inline constexpr int SEND_LEN_TAG = 0;
inline constexpr int SEND_PARTIAL_TAG = 1;

inline std::mt19937_64 rnd(std::chrono::steady_clock::now().time_since_epoch().count());
inline std::uniform_int_distribution<int64_t> rnd_range(-R, R);

struct Point {
  int64_t x, y;
};
struct Circle {
  int64_t r;
};

inline constexpr Circle circle{R};

inline Point get_random_point() { return Point{rnd_range(rnd), rnd_range(rnd)}; }
template <typename T> concept Squarable = requires(T t) { t * t; };
template <typename T> inline auto sq(auto x) -> decltype(x * x) requires Squarable<T> { return x * x; }
inline bool is_inside(Point &&pt, const Circle &c = circle) { return sq<decltype(pt.x)>(pt.x) + sq<decltype(pt.y)>(pt.y) <= sq<decltype(c.r)>(c.r); }

typedef long double ld;

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);

  int n_procs;
  MPI_Comm_size(MPI_COMM_WORLD, &n_procs);
  int cur_proc;
  MPI_Comm_rank(MPI_COMM_WORLD, &cur_proc);

  if (cur_proc == MASTER_PROC) {
    const auto per_proc = (N + n_procs - 1) / n_procs;
    int64_t left = N;
    for (int proc_id = 1; proc_id < n_procs; ++proc_id) {
      MPI_Send(&per_proc,
               1,
               MPI_INT64_T,
               proc_id,
               SEND_LEN_TAG,
               MPI_COMM_WORLD);
      left -= per_proc;
    }

    ld inside = 0;
    for (std::size_t i = 0u; i < left; ++i) {
      inside += is_inside(get_random_point());
    }

    for (int proc_id = 1; proc_id < n_procs; ++proc_id) {
      ld partial;
      MPI_Status status;
      MPI_Recv(&partial,
               1,
               MPI_LONG_DOUBLE,
               MPI_ANY_SOURCE,
               SEND_PARTIAL_TAG,
               MPI_COMM_WORLD,
               &status);
      inside += partial;
    }

    ld pi = 4 * inside / N;
    std::cout << "PI=" << pi << std::endl;
  } else {
    int cnt;
    MPI_Status status;
    MPI_Recv(&cnt,
             1,
             MPI_INT64_T,
             MASTER_PROC,
             SEND_LEN_TAG,
             MPI_COMM_WORLD,
             &status);
    ld partial = 0;
    for (std::size_t i = 0u; i < cnt; ++i) {
      partial += is_inside(get_random_point());
    }
    MPI_Send(&partial,
             1,
             MPI_LONG_DOUBLE,
             MASTER_PROC,
             SEND_PARTIAL_TAG,
             MPI_COMM_WORLD);
  }

  MPI_Finalize();
}