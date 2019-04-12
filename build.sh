mkdir build

cd build

PYTHON_EXECUTABLE=$(which python)

#CC_FLAG=$(which icc)
#CXX_FLAG=$(which icc)

#CC=$CC_FLAG CXX=$CXX_FLAG
cmake -DPYTHON_EXECUTABLE:FILEPATH=$PYTHON_EXECUTABLE ../

make -j4
