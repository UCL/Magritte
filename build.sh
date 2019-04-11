mkdir build

cd build

PYTHON_EXECUTABLE=$(which python)

cmake -DPYTHON_EXECUTABLE:FILEPATH=$PYTHON_EXECUTABLE ../

make -j4
