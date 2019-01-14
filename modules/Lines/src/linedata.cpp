// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <math.h>
#include <string>
#include <vector>
#include <fstream>
#include <iostream>
using namespace std;
#include <Eigen/Core>
using namespace Eigen;

#include "linedata.hpp"
#include "linedata_config.hpp"
#include "RadiativeTransfer/src/folders.hpp"
#include "RadiativeTransfer/src/constants.hpp"
#include "RadiativeTransfer/src/species.hpp"
#include "RadiativeTransfer/src/interpolation.hpp"


///  Constructor for LINEDATA
/////////////////////////////

LINEDATA ::
LINEDATA (const string linedata_folder)
 : nlspec          (get_nlspec          (linedata_folder))
 , nlev            (get_nlev            (linedata_folder))
 , nrad            (get_nrad            (linedata_folder))
 , ncolpar         (get_ncolpar         (linedata_folder))
 , ntmp            (get_ntmp            (linedata_folder))
 , ncol            (get_ncol            (linedata_folder))
 , num             (get_num             (linedata_folder))
 , sym             (get_sym             (linedata_folder))
 , irad            (get_irad            (linedata_folder))
 , jrad            (get_jrad            (linedata_folder))
 , energy          (get_energy          (linedata_folder))
 , weight          (get_weight          (linedata_folder))
 , frequency       (get_frequency       (linedata_folder))
 , A               (get_A               (linedata_folder))
 , B               (get_B               (linedata_folder))
 , num_col_partner (get_num_col_partner (linedata_folder))
 , orth_or_para_H2 (get_orth_or_para_H2 (linedata_folder))
 , temperature_col (get_temperature_col (linedata_folder))
 , C_data          (get_C_data          (linedata_folder))
{


  //num.resize (nlspec);
  //num = NUMBER;
  //sym.resize (nlspec);
  //sym = NAME;

  //irad.resize (nlspec);
  //irad = IRAD;
  //jrad.resize (nlspec);
  //jrad = JRAD;

  //energy.resize (nlspec);
  //Double2 energy_buffer = ENERGY;
  //weight.resize (nlspec);
  //Double2 weight_buffer = WEIGHT;

  //frequency.resize (nlspec);
  //Double2 frequency_buffer = FREQ;

  //A.resize (nlspec);
  //Double3 A_buffer = A_COEFF;
  //B.resize (nlspec);
  //Double3 B_buffer = B_COEFF;

  //num_col_partner.resize (nlspec);
  //num_col_partner = PARTNER_NR;
  //orth_or_para_H2.resize (nlspec);
  //orth_or_para_H2 = ORTHO_PARA;
  //temperature_col.resize (nlspec);
  //temperature_col = COLTEMP;

  //C_data.resize (nlspec);
  //Double5 C_data_buffer = C_DATA;

  //icol.resize (nlspec);
  //icol = ICOL;
  //jcol.resize (nlspec);
  //jcol = JCOL;


//  for (int l = 0; l < nlspec; l++)
//  {
//    //irad[l].resize (nrad[l]);
//    //jrad[l].resize (nrad[l]);
//
//    //energy[l].resize (nlev[l]);
//    //weight[l].resize (nlev[l]);
//
//    //frequency[l].resize (nrad[l]);
//
//    //A[l].resize (nlev[l],nlev[l]);
//    //B[l].resize (nlev[l],nlev[l]);
//
//    for (int i = 0; i < nlev[l]; i++)
//    {
//      //energy[l](i) =  energy_buffer[l][i];
//      //weight[l](i) =  weight_buffer[l][i];
//
//      for (int j = 0; j < nlev[l]; j++)
//      {
//      //  A[l](i,j) = A_buffer[l][i][j];
//      //  B[l](i,j) = B_buffer[l][i][j];
//      }
//    }
//
//
//   // for (int k = 0; k < nrad[l]; k++)
//   // {
//   //   frequency[l][k] = frequency_buffer[l][k];
//   // }
//
//    //num_col_partner[l].resize (ncolpar[l]);
//    //orth_or_para_H2[l].resize (ncolpar[l]);
//    //temperature_col[l].resize (ncolpar[l]);
//
//    //C_data[l].resize (ncolpar[l]);
//
//    //icol[l].resize (ncolpar[l]);
//    //jcol[l].resize (ncolpar[l]);
//
//
//    for (int c = 0; c < ncolpar[l]; c++)
//    {
//      //temperature_col[l][c].resize (ntmp[l][c]);
//
//      //C_data[l][c].resize (ntmp[l][c]);
//
//      //icol[l][c].resize (ncol[l][c]);
//      //jcol[l][c].resize (ncol[l][c]);
//
//      for (int t = 0; t < ntmp[l][c]; t++)
//      {
//        //C_data[l][c][t].resize(nlev[l],nlev[l]);
//
//        for (int i = 0; i < nlev[l]; i++)
//        {
//          for (int j = 0; j < nlev[l]; j++)
//          {
//            //C_data[l][c][t](i,j) = C_data_buffer[l][c][t][i][j];
//          }
//        }
//      }
//    }
//  }


}   // END OF CONSTRUCTOR




int LINEDATA ::
    get_nlspec (const string linedata_folder)
{

  int nlspec_local;

  ifstream infile (linedata_folder + "nlspec.txt");

  infile >> nlspec_local;

  infile.close();


  return nlspec_local;

}


Int1 LINEDATA ::
     get_nlev (const string linedata_folder)
{

  Int1 nlev_local;

  ifstream infile (linedata_folder + "nlev.txt");

  int n;

  while (infile >> n)
  {
    nlev_local.push_back (n);
  }

  infile.close();


  return nlev_local;

}


Int1 LINEDATA ::
     get_nrad (const string linedata_folder)
{

  Int1 nrad_local;

  ifstream infile (linedata_folder + "nrad.txt");

  int n;

  while (infile >> n)
  {
    nrad_local.push_back (n);
  }

  infile.close();


  return nrad_local;

}


Int1 LINEDATA ::
     get_ncolpar (const string linedata_folder)
{

  Int1 ncolpar_local;

  ifstream infile (linedata_folder + "ncolpar.txt");

  int n;

  while (infile >> n)
  {
    ncolpar_local.push_back (n);
  }

  infile.close();


  return ncolpar_local;

}

Int2 LINEDATA ::
     get_ntmp (const string linedata_folder)
{

  Int2 ntmp_local;

  ifstream infile (linedata_folder + "ntmp.txt");

  int n;

  const int   nlspec_local = get_nlspec  (linedata_folder);
  const Int1 ncolpar_local = get_ncolpar (linedata_folder);

  for (int l = 0; l < nlspec_local; l++)
  {
    Int1 ntmp_local_local;

    for (int c = 0; c < ncolpar_local[l]; c++)
    {
      infile >> n;

      ntmp_local_local.push_back (n);
    }

    ntmp_local.push_back (ntmp_local_local);
  }

  infile.close();


  return ntmp_local;

}

Int2 LINEDATA ::
     get_ncol (const string linedata_folder)
{

  Int2 ncol_local;

  ifstream infile (linedata_folder + "ncol.txt");

  int n;

  const int   nlspec_local = get_nlspec  (linedata_folder);
  const Int1 ncolpar_local = get_ncolpar (linedata_folder);

  for (int l = 0; l < nlspec_local; l++)
  {
    Int1 ncol_local_local;

    for (int c = 0; c < ncolpar_local[l]; c++)
    {
      infile >> n;

      ncol_local_local.push_back (n);
    }

    ncol_local.push_back (ncol_local_local);
  }

  infile.close();


  return ncol_local;

}

Int1 LINEDATA ::
     get_num (const string linedata_folder)
{

  Int1 num_local;

  ifstream infile (linedata_folder + "num.txt");

  int n;

  while (infile >> n)
  {
    num_local.push_back (n);
  }

  infile.close();


  return num_local;

}

String1 LINEDATA ::
        get_sym (const string linedata_folder)
{

  String1 sym_local;

  ifstream infile (linedata_folder + "num.txt");

  string n;

  while (infile >> n)
  {
    sym_local.push_back (n);
  }

  infile.close();


  return sym_local;

}


Int2 LINEDATA ::
     get_irad (const string linedata_folder)
{

  Int2 irad_local;

  ifstream infile (linedata_folder + "irad.txt");

  int n;

  const int nlspec_local = get_nlspec  (linedata_folder);
  const Int1  nrad_local = get_nrad    (linedata_folder);

  for (int l = 0; l < nlspec_local; l++)
  {
    Int1 irad_local_local;

    for (int k = 0; k < nrad_local[l]; k++)
    {
      infile >> n;

      irad_local_local.push_back (n);
    }

    irad_local.push_back (irad_local_local);
  }

  infile.close();


  return irad_local;

}


Int2 LINEDATA ::
     get_jrad (const string linedata_folder)
{

  Int2 jrad_local;

  ifstream infile (linedata_folder + "jrad.txt");

  int n;

  const int nlspec_local = get_nlspec  (linedata_folder);
  const Int1  nrad_local = get_nrad    (linedata_folder);

  for (int l = 0; l < nlspec_local; l++)
  {
    Int1 jrad_local_local;

    for (int k = 0; k < nrad_local[l]; k++)
    {
      infile >> n;

      jrad_local_local.push_back (n);
    }

    jrad_local.push_back (jrad_local_local);
  }

  infile.close();


  return jrad_local;

}


VectorXd1 LINEDATA ::
          get_energy (const string linedata_folder)
{

  const int nlspec_local = get_nlspec (linedata_folder);
  const Int1  nlev_local = get_nlev   (linedata_folder);

  VectorXd1 energy_local (nlspec_local);

  ifstream infile (linedata_folder + "energy.txt");

  double x;


  for (int l = 0; l < nlspec_local; l++)
  {
    energy_local[l].resize(nlev_local[l]);

    for (int i = 0; i < nlev_local[l]; i++)
    {
      infile >> x;

      energy_local[l](i) = x;
    }
  }

  infile.close();


  return energy_local;

}


VectorXd1 LINEDATA ::
          get_weight (const string linedata_folder)
{

  const int nlspec_local = get_nlspec (linedata_folder);
  const Int1  nlev_local = get_nlev   (linedata_folder);

  VectorXd1 weight_local (nlspec_local);

  ifstream infile (linedata_folder + "weight.txt");

  double x;


  for (int l = 0; l < nlspec_local; l++)
  {
    weight_local[l].resize(nlev_local[l]);

    for (int i = 0; i < nlev_local[l]; i++)
    {
      infile >> x;

      weight_local[l](i) = x;
    }
  }

  infile.close();


  return weight_local;

}


Double2 LINEDATA ::
        get_frequency (const string linedata_folder)
{

  const int nlspec_local = get_nlspec (linedata_folder);
  const Int1  nrad_local = get_nrad   (linedata_folder);

  Double2 frequency_local (nlspec_local);

  ifstream infile (linedata_folder + "frequency.txt");

  double x;


  for (int l = 0; l < nlspec_local; l++)
  {
    frequency_local[l].resize(nrad_local[l]);

    for (int k = 0; k < nrad_local[l]; k++)
    {
      infile >> x;

      frequency_local[l][k] = x;
    }
  }

  infile.close();


  return frequency_local;

}

MatrixXd1 LINEDATA ::
          get_A (const string linedata_folder)
{

  const int nlspec_local = get_nlspec (linedata_folder);
  const Int1  nlev_local = get_nlev   (linedata_folder);

  MatrixXd1 A_local (nlspec_local);


  double x;


  for (int l = 0; l < nlspec_local; l++)
  {
    ifstream infile (linedata_folder + "A_" + to_string(l) + ".txt");

    A_local[l].resize (nlev_local[l],nlev_local[l]);

    for (int i = 0; i < nlev_local[l]; i++)
    {
      for (int j = 0; j < nlev_local[l]; j++)
      {
        infile >> x;

        A_local[l](i,j) = x;
      }
    }

    infile.close();
  }


  return A_local;

}



MatrixXd1 LINEDATA ::
          get_B (const string linedata_folder)
{

  const int nlspec_local = get_nlspec (linedata_folder);
  const Int1  nlev_local = get_nlev   (linedata_folder);

  MatrixXd1 B_local (nlspec_local);


  double x;


  for (int l = 0; l < nlspec_local; l++)
  {
    ifstream infile (linedata_folder + "B_" + to_string(l) + ".txt");

    B_local[l].resize (nlev_local[l],nlev_local[l]);

    for (int i = 0; i < nlev_local[l]; i++)
    {
      for (int j = 0; j < nlev_local[l]; j++)
      {
        infile >> x;

        B_local[l](i,j) = x;
      }
    }

    infile.close();
  }


  return B_local;

}


Int2 LINEDATA ::
     get_num_col_partner (const string linedata_folder)
{

  const int nlspec_local   = get_nlspec  (linedata_folder);
  const Int1 ncolpar_local = get_ncolpar (linedata_folder);

  Int2 num_col_partner_local (nlspec_local);

  ifstream infile (linedata_folder + "num_col_partner.txt");

  int n;


  for (int l = 0; l < nlspec_local; l++)
  {
    num_col_partner_local[l].resize (ncolpar_local[l]);

    for (int c = 0; c < ncolpar_local[l]; c++)
    {
      infile >> n;

      num_col_partner_local[l][c] = n;
    }
  }

  infile.close();


  return num_col_partner_local;

}


Char2 LINEDATA ::
      get_orth_or_para_H2 (const string linedata_folder)
{

  const int nlspec_local   = get_nlspec  (linedata_folder);
  const Int1 ncolpar_local = get_ncolpar (linedata_folder);

  Char2 orth_or_para_H2_local (nlspec_local);

  ifstream infile (linedata_folder + "orth_or_para_H2.txt");

  char n;


  for (int l = 0; l < nlspec_local; l++)
  {
    orth_or_para_H2_local[l].resize (ncolpar_local[l]);

    for (int c = 0; c < ncolpar_local[l]; c++)
    {
      infile >> n;

      orth_or_para_H2_local[l][c] = n;
    }
  }

  infile.close();


  return orth_or_para_H2_local;

}


Double3 LINEDATA ::
        get_temperature_col (const string linedata_folder)
{

  const int   nlspec_local = get_nlspec  (linedata_folder);
  const Int1 ncolpar_local = get_ncolpar (linedata_folder);
  const Int2    ntmp_local = get_ntmp    (linedata_folder);

  Double3 temperature_col_local (nlspec_local);


  double x;


  for (int l = 0; l < nlspec_local; l++)
  {
    temperature_col_local[l].resize (ncolpar_local[l]);

    ifstream infile (linedata_folder + "temperature_col_" + to_string(l) + ".txt");

    for (int c = 0; c < ncolpar_local[l]; c++)
    {
      temperature_col_local[l][c].resize (ntmp_local[l][c]);

      for (int t = 0; t < ntmp_local[l][c]; t++)
      {
        infile >> x;

        temperature_col_local[l][c][t] = x;
      }
    }

    infile.close();
  }


  return temperature_col_local;

}

MatrixXd3 LINEDATA ::
          get_C_data (const string linedata_folder)
{

  const int   nlspec_local = get_nlspec  (linedata_folder);
  const Int1 ncolpar_local = get_ncolpar (linedata_folder);
  const Int2    ntmp_local = get_ntmp    (linedata_folder);
  const Int1    nlev_local = get_nlev    (linedata_folder);

  MatrixXd3 C_data_local (nlspec_local);


  double x;


  for (int l = 0; l < nlspec_local; l++)
  {
    C_data_local[l].resize (ncolpar_local[l]);

    for (int c = 0; c < ncolpar_local[l]; c++)
    {
      C_data_local[l][c].resize (ntmp_local[l][c]);

      for (int t = 0; t < ntmp_local[l][c]; t++)
      {
        C_data_local[l][c][t].resize (nlev_local[l],nlev_local[l]);

        const string fname = linedata_folder + "C_data_"
                             + to_string (l) + "_" + to_string (c) + "_"
                             + to_string (t) + ".txt";

        ifstream infile (fname);

        for (int i = 0; i < nlev_local[l]; i++)
        {
          for (int j = 0; j < nlev_local[l]; j++)
          {
            infile >> x;

            C_data_local[l][c][t](i,j) = x;
          }
        }

        infile.close();
      }
    }
  }


  return C_data_local;

}


int LINEDATA ::
    print (const MatrixXd& M,
           const string tag  ) const
{

//	int world_rank;
//	MPI_Comm_rank (MPI_COMM_WORLD, &world_rank);
//
//
//	if (world_rank == 0)
//	{
  string file_name = output_folder + tag + ".txt";

  ofstream outputFile (file_name);

  outputFile << M << endl;

  outputFile.close ();

//	}


  return (0);

}



///  calc_Einstein_C: calculate the Einstein C coefficient
///    @param[in] species: data structure containing chamical species
///    @param[in] temperature_gas: local gas temperature
///    @param[in] p: number of the cell under consideration
///    @param[in] l: number of the line producing species under consideration
///    @return Einstein C collisional transition matrix
/////////////////////////////////////////////////////////////////////////////

MatrixXd LINEDATA ::
    calc_Einstein_C                    (
        const SPECIES &species,
        const double   temperature_gas,
        const long     p,
        const int      l               ) const
{

  MatrixXd C = MatrixXd::Zero (nlev[l],nlev[l]);   // Einstein C_ij coefficient


  // Calculate H2 ortho/para fraction at equilibrium for given temperature

  double frac_H2_para  = 0.0;   // fraction of para-H2
  double frac_H2_ortho = 0.0;   // fraction of ortho-H2


  if (species.abundance[p][species.nr_H2] > 0.0)
  {
    frac_H2_para  = 1.0 / (1.0 + 9.0*exp(-170.5/temperature_gas));
    frac_H2_ortho = 1.0 - frac_H2_para;
  }


  // For all collision partners

  for (int c = 0; c < ncolpar[l]; c++)
  {

    // Weigh contributions by abundance

    const int spec = num_col_partner[l][c];

    double abundance = species.abundance[p][spec];

    if      (orth_or_para_H2[l][c] == 'o')
    {
      abundance *= frac_H2_ortho;
    }

    else if (orth_or_para_H2[l][c] == 'p')
    {
      abundance *= frac_H2_para;
    }


    int t = search (temperature_col[l][c], temperature_gas);

    if      (t == 0)
    {
      C += C_data[l][c][0] * abundance;
    }

    else if (t == ntmp[l][c])
    {
      C += C_data[l][c][ntmp[l][c]-1] * abundance;
    }

    else
    {
      const double step = (temperature_gas - temperature_col[l][c][t-1])
                          / (temperature_col[l][c][t] - temperature_col[l][c][t-1]);

      C += ( C_data[l][c][t-1] + (C_data[l][c][t] - C_data[l][c][t-1])*step ) * abundance;
    }

  } // end of par loop over collision partners


  return C;

}




///  calc_transition_matrix: calculate the transition matrix
///    @param[in] species: data structure containing chamical species
///    @param[in] temperature_gas: local gas temperature
///    @param[in] J_eff: effective mean intensity
///    @param[in] p: number of the cell under consideration
///    @param[in] l: number of the line producing species under consideration
///    @return Einstein C collisional transition matrix
/////////////////////////////////////////////////////////////////////////////

MatrixXd LINEDATA ::
         calc_transition_matrix (const SPECIES &species,
//                                 const LINES   &lines,
                                 const double   temperature_gas,
                                 const Double3 &J_eff,
                                 const long     p,
                                 const int      l               ) const
{

  // Calculate collissional Einstein coefficients

  MatrixXd C = calc_Einstein_C (species, temperature_gas, p, l);


  // Add Einstein A and C to transition matrix

  MatrixXd R = A[l] + C;


  // Add B_ij<J_ij> term

  for (int k = 0; k < nrad[l]; k++)
  {
    const int i = irad[l][k];   // i index corresponding to transition k
    const int j = jrad[l][k];   // j index corresponding to transition k

//    const long ind = lines.index(p,l,k);

    const double Jeff = J_eff[p][l][k];// - lines.emissivity[ind] * L_eff[p][l][k];

    R(i,j) += /*FOUR_PI */ Jeff * B[l](i,j);// - linedata.A[l](i,j) * lines.opacity[ind] * L_eff[p][l][k];
    R(j,i) += /*FOUR_PI */ Jeff * B[l](j,i);
  }

  //if(p==0)
  //{
  //  cout << "A" << endl;
  //  cout << A[l] << endl;

  //  cout << "B" << endl;
  //  cout << B[l] << endl;

  //  cout << "C" << endl;
  //  cout << C << endl;

  //  cout << "R-C" << endl;
  //  cout << R-C << endl;

  //  cout << "J_eff" << endl;

  //  for (int k = 0; k < nrad[l]; k++)
  //  {
  //    cout << J_eff[p][l][k] << endl; // - linedata.A[l](i,j)*Lambda();
  //  }
  //}

  //cout << "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA : " << endl;
  //cout << A[l] << endl;
  //cout << "BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB : " << endl;
  //cout << B[l] << endl;

  //cout << endl;
  //cout << R << endl;
  //cout << endl;


  return R;

}
