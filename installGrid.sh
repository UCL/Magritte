#! /bin/bash

export CWD=`pwd`


# install gmp

cd $CWD

wget https://gmplib.org/download/gmp/gmp-6.1.2.tar.xz

tar xf gmp-6.1.2.tar.xz
rm     gmp-6.1.2.tar.xz

cd gmp-6.1.2

./configure
make -j4
make install


# Install mpfr

cd $CWD

wget https://www.mpfr.org/mpfr-current/mpfr-4.0.1.zip

unzip mpfr-4.0.1.zip
rm    mpfr-4.0.1.zip

cd mpfr-4.0.1

./configure --prefix=$HOME/mpfr
make -j4
make install

export LD_LIBRARY_PATH=$HOME/mpfr/lib/:$LD_LIBRARY_PATH








cd $CWD

#
## Install Grid
#
#mkdir $CWD/GridInstall
#
#wget https://github.com/paboyle/Grid/archive/master.zip
#
#unzip master.zip
#rm    master.zip
#
#cd $CWD/Grid-master
#
#./bootstrap.sh
#
#mkdir build
#cd    build
#
#../configure --enable-precision=double --enable-simd=SSE4 --enable-comms=none --prefix=$CWD/GridInstall
#
#make -j4
#make install
#
