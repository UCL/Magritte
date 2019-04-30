#! /bin/bash


if [ "$1" == "clean" ]; then


  echo "Removing entire build directory..."
  rm -rf build/
  echo "Done."
  exit 0


elif [ "$1" == "minimal" ]; then


  echo "Building Magrite with minimal options..."
  mkdir build
  cd build

  cmake                   \
    -DPYTHON_IO=OFF       \
    -DPYTHON_BINDINGS=OFF \
    -DOMP_PARALLEL=ON     \
    -DMPI_PARALLEL=OFF    \
    -DGRID_SIMD=OFF       \
    ../

  make -j4

  cd ..
  echo "Done."
  exit 0


elif [ "$1" == "analysis" ]; then


  echo "Building Magrite with minimal options and score-p analysis"
  mkdir build
  cd build


  SCOREP_WRAPPER=off                              \
  cmake ../                                       \
  -DCMAKE_C_COMPILER=/opt/scorep/bin/scorep-gcc   \
  -DCMAKE_CXX_COMPILER=/opt/scorep/bin/scorep-g++ \
  -DPYTHON_IO=OFF                                 \
  -DPYTHON_BINDINGS=OFF                           \
  -DOMP_PARALLEL=OFF                              \
  -DMPI_PARALLEL=ON                               \
  -DGRID_SIMD=OFF

  make -j4

  cd ..
  echo "Done."
  exit 0


else


  # echo "Building Magrite..."
  # mkdir build
  # cd build
  #
  # PYTHON_EXECUTABLE=$(which python)
  #
  # #CC_FLAG=$(which icc)
  # #CXX_FLAG=$(which icc)
  #
  # #CC=$CC_FLAG CXX=$CXX_FLAG
  # cmake                                             \
  #   -DPYTHON_EXECUTABLE:FILEPATH=$PYTHON_EXECUTABLE \
  #   -DPYTHON_IO=ON                                  \
  #   -DPYTHON_BINDINGS=ON                            \
  #   -DOMP_PARALLEL=OFF                              \
  #   -DMPI_PARALLEL=OFF                              \
  #   -DGRID_SIMD=OFF                                 \
  #   ../
  #
  # make #-j4
  #
  # cd ..

  echo "Please specify a build mode: clean, analysis"
  exit 0


fi
