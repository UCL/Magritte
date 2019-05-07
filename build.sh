#! /bin/bash


# Get directory this script is in
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"


if [ "$1" == "clean" ]; then


  echo "Removing entire build directory..."
  rm -rf build/
  echo "Done."
  exit 0


elif [ "$1" == "minimal" ]; then


  echo "Building Magrite with minimal options..."
  mkdir build
  cd build

  cmake $DIR              \
    -DPYTHON_IO=OFF       \
    -DPYTHON_BINDINGS=OFF \
    -DOMP_PARALLEL=OFF    \
    -DMPI_PARALLEL=OFF    \
    -DGRID_SIMD=OFF       \

  make -j4

  cd ..
  echo "Done."
  exit 0


elif [ "$1" == "performance_audit" ]; then

  SCOREP_FOLDER=$DIR/dependencies/scorep/installed/bin

  echo "Building Magritte with Score-P instrumentation..."
  echo "-------------------------------------------------"
  mkdir build
  cd build

  SCOREP_WRAPPER=off                                \
  cmake                                             \
    -DPERF_ANALYSIS=ON                              \
    -DCMAKE_C_COMPILER=$SCOREP_FOLDER/scorep-gcc    \
    -DCMAKE_CXX_COMPILER=$SCOREP_FOLDER/scorep-g++  \
    -DOMP_PARALLEL=OFF                              \
    -DMPI_PARALLEL=OFF                              \
    -DGRID_SIMD=OFF                                 \
    $DIR

  make

  cd ..
  echo "-----"
  echo "Done."
  exit 0


else


   echo "Building Magrite..."
   echo "-------------------"
   mkdir build
   cd build

   PYTHON_EXECUTABLE=$(which python)

   #CC_FLAG=$(which icc)
   #CXX_FLAG=$(which icc)

   #CC=$CC_FLAG CXX=$CXX_FLAG
   cmake                                             \
     -DPYTHON_EXECUTABLE:FILEPATH=$PYTHON_EXECUTABLE \
     -DPYTHON_IO=ON                                  \
     -DPYTHON_BINDINGS=ON                            \
     -DOMP_PARALLEL=OFF                              \
     -DMPI_PARALLEL=OFF                              \
     -DGRID_SIMD=OFF                                 \
     $DIR

   make -j4

   cd ..

  echo "----"
  echo "Done"
  exit 0


fi
