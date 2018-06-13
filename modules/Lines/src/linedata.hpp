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
#include "RadiativeTransfer/src/types.hpp"
#include "RadiativeTransfer/src/species.hpp"


struct LINEDATA
{

	int nlspec = NLSPEC;

  Int1 nlev = NLEV;
  Int1 nrad = NRAD;
  
  Int1 ncolpar = NCOLPAR;
  
  Int2 ntmp = NCOLTEMP;
  Int2 ncol = NCOLTRAN;

	Int1    num;               // number of line producing species
  String1 sym;               // symbol of line producing species

  Int2 irad;                 // level index of radiative transition
  Int2 jrad;                 // level index of radiative transition

  VectorXd1 energy;          // energy of level
  VectorXd1 weight;          // weight of level (statistical)

  MatrixXd1 frequency;       // frequency corresponing to i <-> j transition

  MatrixXd1 A;               // Einstein A_ij coefficient
  MatrixXd1 B;               // Einstein B_ij coefficient


  // Collision related variables

  Int2    num_col_partner;   // species number corresponding to collision partner
  Char2   orth_or_para_H2;   // stores whether it is ortho or para H2
  Double3 temperature_col;   // Collision temperatures for each partner

  MatrixXd3 C_data;          // Einstein C_ij for each partner and temp.

  Int3 icol;                 // level index corresp. to col. transition
  Int3 jcol;                 // level index corresp. to col. transition


  LINEDATA ();   ///< Constructor


  MatrixXd calc_Einstein_C (const SPECIES& species, const double temperature_gas,
			                      const long p, const int l); 


  MatrixXd calc_transition_matrix (const SPECIES& species, const double temperature_gas,
			                             const Double3& J_eff, const long p, const int l); 


};


#endif // __LINEDATA_HPP_INCLUDED__
