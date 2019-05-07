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
  MODEL=$1

  # Number of processes and threads
  N_PROCS=1
  N_THRDS=1

  # Flag for shared memory systems
  FLAGS="-env I_MPI_SHM_LMT shm"

  # Path to Magritte executable
  EXECUTABLE="../bin/examples/example_2.exe"

  # Set number of threads
  export OMP_NUM_THREADS=$N_THRDS

  # Allow Score-P to trace the run
  #export SCOREP_ENABLE_TRACING=true


  echo "Running Magritte..."

  mpirun -np $N_PROCS $FLAGS $EXECUTABLE $MODEL

  echo "Done."
  exit 0


fi
