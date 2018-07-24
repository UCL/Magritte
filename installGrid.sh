#! /bin/bash

export CWD=`pwd`

mkdir GridInstall

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
