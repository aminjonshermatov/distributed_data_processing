run:
	mpiexec -n $(threads) ../../cmake-build-debug/algorithm/mpi/mpi_floyd $(N)

all:
	for N in 240 480 720 960 ; do \
		for threads in 1 2 3 4 ; do \
			mpiexec -n $$threads ../../cmake-build-debug/algorithm/mpi/mpi_floyd $$N ; \
		done ; \
	done ; \
