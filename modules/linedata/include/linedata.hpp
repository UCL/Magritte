// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __LINEDATA_HPP_INCLUDED__
#define __LINEDATA_HPP_INCLUDED__


#include "Eigen/Core"
using namespace Eigen;

#include "types.hpp"
#include "constants.hpp"


struct LINEDATA
{

  const int nlev;        ///< number of levels
  const int nrad;        ///<number of (radiative) transitions

  Int1 irad;             ///< lower level of transition
  Int1 jrad;             ///< upper level of transition

  VectorXd energy;       ///< energy of level
  VectorXd weight;       ///< weight of level (statistical)

  VectorXd population;   ///< population of level
  VectorXd frequency;    ///< frequency  of transition

  VectorXd emissivity;   ///< emissivity of transition
  VectorXd opacity;      ///< opacity    of transition

  MatrixXd A;            ///< Einstein A_ij coefficient
  MatrixXd B;            ///< Einstein B_ij coefficient


  LINEDATA                 (
      const int nlev_local,
      const int nrad_local );


  void compute_populations_LTE (
      const double temperature );

  void compute_emissivity_and_opacity (
      void                            );

//  const int nlspec /*= NLSPEC*/;
//
//  const Int1 nlev/* = NLEV*/;
//  const Int1 nrad/* = NRAD*/;
//
//  const Int1 ncolpar /*= NCOLPAR*/;
//
//  const Int2 ntmp /*= NCOLTEMP*/;
//  const Int2 ncol /*= NCOLTRAN*/;
//
//  const Int1    num;               // number of line producing species
//  const String1 sym;               // symbol of line producing species
//
//  const Int2 irad;                 // level index of radiative transition
//  const Int2 jrad;                 // level index of radiative transition
//
//  const VectorXd1 energy;          // energy of level
//  const VectorXd1 weight;          // weight of level (statistical)
//
//  const Double2 frequency;
//
//  const MatrixXd1 A;               // Einstein A_ij coefficient
//  const MatrixXd1 B;               // Einstein B_ij coefficient
//
//
//  // Collision related variables
//
//  const Int2    num_col_partner;   // species number corresponding to collision partner
//  const Char2   orth_or_para_H2;   // stores whether it is ortho or para H2
//  const Double3 temperature_col;   // Collision temperatures for each partner
//
//  const MatrixXd3 C_data;          // Einstein C_ij for each partner and temp.
//
//  //const Int3 icol;                 // level index corresp. to col. transition
//  //const Int3 jcol;                 // level index corresp. to col. transition
//
//
//  LINEDATA (const string linedata_folder);   ///< Constructor
//
////  int read (const string linedata_folder);
//
//  static int       get_nlspec          (const string linedata_folder);
//  static Int1      get_nlev            (const string linedata_folder);
//  static Int1      get_nrad            (const string linedata_folder);
//  static Int1      get_ncolpar         (const string linedata_folder);
//  static Int2      get_ntmp            (const string linedata_folder);
//  static Int2      get_ncol            (const string linedata_folder);
//  static Int1      get_num             (const string linedata_folder);
//  static String1   get_sym             (const string linedata_folder);
//  static Int2      get_irad            (const string linedata_folder);
//  static Int2      get_jrad            (const string linedata_folder);
//  static VectorXd1 get_energy          (const string linedata_folder);
//  static VectorXd1 get_weight          (const string linedata_folder);
//  static Double2   get_frequency       (const string linedata_folder);
//  static MatrixXd1 get_A               (const string linedata_folder);
//  static MatrixXd1 get_B               (const string linedata_folder);
//  static Int2      get_num_col_partner (const string linedata_folder);
//  static Char2     get_orth_or_para_H2 (const string linedata_folder);
//  static Double3   get_temperature_col (const string linedata_folder);
//  static MatrixXd3 get_C_data          (const string linedata_folder);
//
//  int print (const MatrixXd &M,
//             const string    tag) const;
//
//
//  MatrixXd calc_Einstein_C (const SPECIES &species,
//                            const double   temperature_gas,
//			    const long     p,
//                            const int      l               ) const;
//
//
//  MatrixXd calc_transition_matrix (const SPECIES &species,
//                                   const double   temperature_gas,
//                                   const Double3 &J_eff,
//                                   const long     p,
//                                   const int      l               ) const;


};


#endif // __LINEDATA_HPP_INCLUDED__
