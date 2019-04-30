#! /bin/bash

if [ "$1" == "clean" ]; then


  echo "Removing entire run directory..."
  rm -rf run/
  echo "Done."
  exit 0


else


  mkdir run
  cd run

  # Model file
  MODEL_FILE=$1

  # Number of processes and threads
  NUMBER_OF_PROCS=2
  NUMBER_OF_THRDS=1

  # Flag for shared memory systems
  FLAGS="-env I_MPI_SHM_LMT shm"

  # Path to Magritte executable
  PATH_TO_EXECUTABLE="../bin/examples/example_2.exe"

  # Set number of threads
  export OMP_NUM_THREADS=$NUMBER_OF_THRDS

  # Allow Score-P to trace the run
  #export SCOREP_ENABLE_TRACING=true


  echo "Running Magritte..."

  mpirun -np $NUMBER_OF_PROCS $FLAGS $PATH_TO_EXECUTABLE $MODEL_FILE

  echo "Done."
  exit 0


fi
