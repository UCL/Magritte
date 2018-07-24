#! /bin/bash

export CWD=`pwd`


# Install mpfr

wget https://www.mpfr.org/mpfr-current/mpfr-4.0.1.zip

unzip mpfr-4.0.1.zip
rm    mpfr-4.0.1.zip

cd mpfr-4.0.1

./configure

make

make install


# Install Grid

mkdir $CWD/GridInstall

wget https://github.com/paboyle/Grid/archive/master.zip

unzip master.zip
rm    master.zip

cd $CWD/Grid-master

./bootstrap.sh

mkdir build
cd    build

../configure --enable-precision=double --enable-simd=SSE4 --enable-comms=none --prefix=$CWD/GridInstall

make -j4
make install
