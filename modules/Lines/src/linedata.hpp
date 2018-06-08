// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __LINEDATA_HPP_INCLUDED__
#define __LINEDATA_HPP_INCLUDED__


#include <string>
#include <vector>
using namespace std;
#include <Eigen/Core>
using namespace Eigen;

#include "linedata_config.hpp"
#include "RadiativeTransfer/src/species.hpp"


struct LINEDATA
{

	int nlspec = NLSPEC;

  vector<int> nlev = NLEV;
  vector<int> nrad = NRAD;
  
  vector<int> ncolpar = NCOLPAR;
  
  vector<vector<int>> ntmp = NCOLTEMP;
  vector<vector<int>> ncol = NCOLTRAN;

	vector<int>    num;                               // number of line producing species
  vector<string> sym;                               // symbol of line producing species

  vector<vector<int>> irad;                         // level index of radiative transition
  vector<vector<int>> jrad;                         // level index of radiative transition

  vector<VectorXd> energy;                          // energy of level
  vector<VectorXd> weight;                          // weight of level (statistical)

  vector<MatrixXd> frequency;                       // frequency corresponing to i <-> j transition

  vector<MatrixXd> A;                               // Einstein A_ij coefficient
  vector<MatrixXd> B;                               // Einstein B_ij coefficient


  // Collision related variables

  vector<vector<int>>            num_col_partner;   // species number corresponding to a collision partner
  vector<vector<char>>           orth_or_para_H2;   // stores whether it is ortho or para H2
  vector<vector<vector<double>>> temperature_col;   // Collision temperatures for each partner

  vector<vector<vector<MatrixXd>>> C_data;          // Einstein C_ij for each partner and temp.

  vector<vector<vector<int>>> icol;                 // level index corresp. to col. transition
  vector<vector<vector<int>>> jcol;                 // level index corresp. to col. transition


  LINEDATA ();   ///< Constructor

  int calc_Einstein_C (SPECIES species, double temperature_gas,
                       long o, int l, MatrixXd& C);   ///< interpolate Einstein C_ij given temperature 


};


#endif // __LINEDATA_HPP_INCLUDED__
