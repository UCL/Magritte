#! /bin/bash


# Get directory this script is in
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"


cd scorep/scorep-5.0

mkdir _build
cd    _build

bash ../configure                  \
  --prefix=$DIR/scorep/installed   \
  --without-libunwind
make
make install

cd ../..
