#! /bin/bash

# Model file
MODEL_FILE=$1

# Number of processes and threads
NUMBER_OF_PROCS=1
NUMBER_OF_THRDS=1

# Flag for shared memory systems
FLAGS="-env I_MPI_SHM_LMT shm"

# Path to Magritte executable
PATH_TO_EXECUTABLE="bin/examples/example_2.exe"

# Set number of threads
export OMP_NUM_THREADS=$NUMBER_OF_THRDS


echo "Running Magritte..."

mpirun -np $NUMBER_OF_PROCS $FLAGS $PATH_TO_EXECUTABLE $MODEL_FILE

echo "Done."
