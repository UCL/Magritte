/*************************************************************************************

Grid physics library, www.github.com/paboyle/Grid

Source file: ./tests/Test_dwf_cg_prec.cc

Copyright (C) 2016

Author: Guido Cossu <guido.cossu@ed.ac.uk>

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License along
with this program; if not, write to the Free Software Foundation, Inc.,
51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

See the full license in the file "LICENSE" in the top level distribution
directory
*************************************************************************************/
/*  END LEGAL */
#include <Grid/Grid.h>

using namespace std;
using namespace Grid;
using namespace Grid::QCD;

template <class d>
struct scal {
  d internal;
};

int main(int argc, char** argv) {
  Grid_init(&argc, &argv);

  const int Ls = 16;

  GridCartesian* UGrid = SpaceTimeGrid::makeFourDimGrid(
      GridDefaultLatt(), GridDefaultSimd(Nd, vComplex::Nsimd()),
      GridDefaultMpi());
  GridRedBlackCartesian* UrbGrid =
      SpaceTimeGrid::makeFourDimRedBlackGrid(UGrid);

  GridCartesian* sUGrid = SpaceTimeGrid::makeFourDimDWFGrid(GridDefaultLatt(), GridDefaultMpi());
  GridRedBlackCartesian* sUrbGrid = SpaceTimeGrid::makeFourDimRedBlackGrid(sUGrid);

  GridCartesian*           FGrid = SpaceTimeGrid::makeFiveDimDWFGrid(Ls, UGrid);
  GridRedBlackCartesian* FrbGrid = SpaceTimeGrid::makeFiveDimDWFRedBlackGrid(Ls, UGrid);

  std::cout << GridLogMessage << "Lattice dimensions: " << GridDefaultLatt()
            << "   Ls: " << Ls << std::endl;

  std::vector<int> seeds4({1, 2, 3, 4});
  std::vector<int> seeds5({5, 6, 7, 8});
  GridParallelRNG RNG5(FGrid);
  RNG5.SeedFixedIntegers(seeds5);
  GridParallelRNG RNG4(UGrid);
  RNG4.SeedFixedIntegers(seeds4);

  std::cout << GridLogMessage << "Generating random fermion field" << std::endl;
  LatticeFermion src(FGrid);
  random(RNG5, src);
  LatticeFermion result(FGrid);
  result = zero;
  LatticeGaugeField Umu(UGrid);

  std::cout << GridLogMessage << "Generating random gauge field" << std::endl;
  SU3::HotConfiguration(RNG4, Umu);



  std::vector<LatticeColourMatrix> U(4, UGrid);
  for (int mu = 0; mu < Nd; mu++) {
    U[mu] = PeekIndex<LorentzIndex>(Umu, mu);
  }

  RealD mass = 0.01;
  RealD M5 = 1.8;
  DomainWallFermionVec5dR Ddwf(Umu, *FGrid, *FrbGrid, *sUGrid, *sUrbGrid, mass, M5);

  LatticeFermion src_o(FrbGrid);
  LatticeFermion result_o(FrbGrid);
  pickCheckerboard(Odd, src_o, src);
  result_o = zero;

  GridStopWatch CGTimer;

  SchurDiagMooeeOperator<DomainWallFermionVec5dR, LatticeFermion> HermOpEO(Ddwf);
  ConjugateGradient<LatticeFermion> CG(1.0e-8, 10000, 0);  // switch off the assert

  Ddwf.ZeroCounters();
  CGTimer.Start();
  CG(HermOpEO, src_o, result_o);
  CGTimer.Stop();

  std::cout << GridLogMessage << "Total CG time : " << CGTimer.Elapsed()
            << std::endl;

  std::cout << GridLogMessage << "######## Dhop calls summary" << std::endl;
  Ddwf.Report();

  Grid_finalize();
}
