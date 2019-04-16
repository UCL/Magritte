#! /bin/bash


if [ "$1" == "clean" ]; then

  echo "Removing entire build directory..."
  rm -rf build/
  echo "Done."

else

  echo "Building Magrite..."
  mkdir build
  cd build

  PYTHON_EXECUTABLE=$(which python)

  #CC_FLAG=$(which icc)
  #CXX_FLAG=$(which icc)

  #CC=$CC_FLAG CXX=$CXX_FLAG
  cmake -DPYTHON_EXECUTABLE:FILEPATH=$PYTHON_EXECUTABLE ../

  make #-j4

  cd ..
  echo "Done."

fi
