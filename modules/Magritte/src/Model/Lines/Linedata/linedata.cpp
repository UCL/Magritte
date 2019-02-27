// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <string>
using std::string;
#include <Eigen/Core>

#include "linedata.hpp"
#include "Tools/constants.hpp"
#include "Functions/interpolation.hpp"


const string Linedata::prefix = "Lines/Linedata/";


///  read: read in line data
///    @param[in] io: io object
///////////////////////////////

int Linedata ::
    read (
        const Io &io,
        const int l  )
{

  const string prefix_l = prefix + "lspec_" + std::to_string (l) + "/";


  io.read_number (prefix_l+".num", num);
  io.read_word   (prefix_l+".sym", sym);

  io.read_length (prefix_l+"energy",    nlev);
  io.read_length (prefix_l+"frequency", nrad);

  irad.resize (nrad);
  jrad.resize (nrad);

  io.read_list (prefix_l+"irad", irad);
  io.read_list (prefix_l+"jrad", jrad);

  energy.resize (nlev);
  weight.resize (nlev);

  io.read_list (prefix_l+"energy", energy);
  io.read_list (prefix_l+"weight", weight);

  frequency.resize (nrad);

  io.read_list (prefix_l+"frequency", frequency);

  A.resize  (nrad);
  Bs.resize (nrad);
  Ba.resize (nrad);

  io.read_list (prefix_l+"A",  A);
  io.read_list (prefix_l+"Bs", Bs);
  io.read_list (prefix_l+"Ba", Ba);


  // Get ncolpar
  io.read_length (prefix_l+"CollisionPartner", ncolpar);


  colpar.resize (ncolpar);

  for (int c = 0; c < ncolpar; c++)
  {
    colpar[c].read (io, l, c);
  }


  return (0);

}




///  write: write out data structure
///    @param[in] io: io object
////////////////////////////////////

int Linedata ::
    write (
        const Io &io,
        const int l  ) const
{

  const string prefix_l = prefix + "lspec_" + std::to_string (l) + "/";


  io.write_number (prefix_l+".num", num);
  io.write_word   (prefix_l+".sym", sym);

  io.write_list (prefix_l+"irad", irad);
  io.write_list (prefix_l+"jrad", jrad);

  io.write_list (prefix_l+"energy", energy);
  io.write_list (prefix_l+"weight", weight);

  io.write_list (prefix_l+"frequency", frequency);

  io.write_list (prefix_l+"A",  A);
  io.write_list (prefix_l+"Bs", Bs);
  io.write_list (prefix_l+"Ba", Ba);


  for (int c = 0; c < ncolpar; c++)
  {
    colpar[c].write (io, l, c);
  }


  return (0);

}



/////  calc_Einstein_C: calculate the Einstein C coefficient
/////    @param[in] species: data structure containing chamical species
/////    @param[in] temperature_gas: local gas temperature
/////    @param[in] p: number of the cell under consideration
/////    @param[in] l: number of the line producing species under consideration
/////    @return Einstein C collisional transition matrix
///////////////////////////////////////////////////////////////////////////////
//
//MatrixXd Linedata ::
//    calc_Einstein_C                    (
//        const Species &species,
//        const double   temperature_gas,
//        const long     p               ) const
//{
//
//  MatrixXd C = MatrixXd::Zero (nlev, nlev);   // Einstein C_ij coefficient
//
//
//  // Calculate H2 ortho/para fraction at equilibrium for given temperature
//
//  double frac_H2_para  = 0.0;   // fraction of para-H2
//  double frac_H2_ortho = 0.0;   // fraction of ortho-H2
//
//
//  if (species.abundance[p][species.nr_H2] > 0.0)
//  {
//    frac_H2_para  = 1.0 / (1.0 + 9.0*exp(-170.5/temperature_gas));
//    frac_H2_ortho = 1.0 - frac_H2_para;
//  }
//
//
//  // For all collision partners
//
//  for (int c = 0; c < ncolpar; c++)
//  {
//
//    // Weigh contributions by abundance
//
//    const int spec = num_col_partner[c];
//
//    double abundance = species.abundance[p][spec];
//
//    if      (orth_or_para_H2[c] == 'o')
//    {
//      abundance *= frac_H2_ortho;
//    }
//
//    else if (orth_or_para_H2[c] == 'p')
//    {
//      abundance *= frac_H2_para;
//    }
//
//
//    int t = search (temperature_col[c], temperature_gas);
//
//    if      (t == 0)
//    {
//      C += C_data[c][0] * abundance;
//    }
//
//    else if (t == ntmp[l][c])
//    {
//      C += C_data[c][ntmp[l][c]-1] * abundance;
//    }
//
//    else
//    {
//      const double step = (temperature_gas - tmp[c][t-1]) / (tmp[l][c][t] - tmp[c][t-//1]);
//
//      C += ( C_data[c][t-1] + (C_data[c][t] - C_data[c][t-1])*step ) * abundance;
//    }
//
//  } // end of par loop over collision partners
//
//
//  return C;
//
//}
//
//
//
//
///  calc_transition_matrix: calculate the transition matrix
///    @param[in] species: data structure containing chamical species
///    @param[in] temperature_gas: local gas temperature
///    @param[in] J_eff: effective mean intensity
///    @param[in] p: number of the cell under consideration
///    @param[in] l: number of the line producing species under consideration
///    @return Einstein C collisional transition matrix
/////////////////////////////////////////////////////////////////////////////

//MatrixXd Linedata ::
//    calc_transition_matrix (
//        const Species &species,
////       const LINES   &lines,
//        const double   temperature_gas,
//        const Double3 &J_eff,
//        const long     p               ) const
//{
//
//  // Calculate collissional Einstein coefficients
//
//  MatrixXd C = calc_Einstein_C (species, temperature_gas, p);
//
//
//  // Add Einstein A and C to transition matrix
//
//  MatrixXd R = A + C;
//
//
//  // Add B_ij<J_ij> term
//
//  for (int k = 0; k < nrad; k++)
//  {
//    const int i = irad[k];   // i index corresponding to transition k
//    const int j = jrad[k];   // j index corresponding to transition k
//
////    const long ind = lines.index(p,l,k);
//
//    const double Jeff = J_eff[p][l][k];// - lines.emissivity[ind] * L_eff[p][l][k];
//
//    R(i,j) += Jeff * Bs[k];// - linedata.A[l](i,j) * lines.opacity[ind] * //L_eff[p][l][k];
//    R(j,i) += Jeff * Ba[k];
//  }
//
//  //if(p==0)
//  //{
//  //  cout << "A" << endl;
//  //  cout << A[l] << endl;
//
//  //  cout << "B" << endl;
//  //  cout << B[l] << endl;
//
//  //  cout << "C" << endl;
//  //  cout << C << endl;
//
//  //  cout << "R-C" << endl;
//  //  cout << R-C << endl;
//
//  //  cout << "J_eff" << endl;
//
//  //  for (int k = 0; k < nrad[l]; k++)
//  //  {
//  //    cout << J_eff[p][l][k] << endl; // - linedata.A[l](i,j)*Lambda();
//  //  }
//  //}
//
//  //cout << "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA : " << endl;
//  //cout << A[l] << endl;
//  //cout << "BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB : " << endl;
//  //cout << B[l] << endl;
//
//  //cout << endl;
//  //cout << R << endl;
//  //cout << endl;
//
//
//  return R;
//
//}
