#! /bin/bash


# Move into the folder given as the first argument
cd $1

echo "WE ARE RUNNING"

NUMBER_OF_PROCESSES=1
NUMBER_OF_THREADS=1 
FLAG_FOR_SHARED_MEM='-env I_MPI_SHM_LMT shm'
PATH_TO_EXECUTABLE=$1'build/examples/example_Lines.exe'

export OMP_NUM_THREADS=$NUMBER_OF_THREADS

# This will run, but the output to screen gets lost
result="$(mpirun -np $NUMBER_OF_PROCESSES $FLAG_FOR_SHARED_MEM $PATH_TO_EXECUTABLE)"

echo $result

#while read -r line
#do
#    echo "$line"
#done < <(result)  
# Printf '%s\n' "$var" is necessary because printf '%s' "$var" on a
# variable that doesn't end with a newline then the while loop will
# completely miss the last line of the variable.
#while IFS= read -r line
#do
#    echo "$line"
#done < <(printf '%s\n' $result)
#
#echo "$result"
#

echo "We are here..."
