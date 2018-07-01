// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <math.h>
#include <string>
#include <vector>
#include <iostream>
using namespace std;
#include <Eigen/Core>
using namespace Eigen;

#include "linedata.hpp"
#include "linedata_config.hpp"
#include "RadiativeTransfer/src/species.hpp"
#include "RadiativeTransfer/src/interpolation.hpp"


///  Constructor for LINEDATA
/////////////////////////////

LINEDATA :: LINEDATA ()  
{

  num.resize (nlspec);
  num = NUMBER;
	sym.resize (nlspec);
  sym = NAME;

  irad.resize (nlspec);
  irad = IRAD;
  jrad.resize (nlspec);
  jrad = JRAD;
  
  energy.resize (nlspec);
	Double2 energy_buffer = ENERGY;
  weight.resize (nlspec);
	Double2 weight_buffer = WEIGHT;

  frequency.resize (nlspec);
	Double3 frequency_buffer = FREQUENCY;

  A.resize (nlspec);
	Double3 A_buffer = A_COEFF;
 	B.resize (nlspec);
	Double3 B_buffer = B_COEFF;

	num_col_partner.resize (nlspec);
	num_col_partner = PARTNER_NR;  
  orth_or_para_H2.resize (nlspec);
	orth_or_para_H2 = ORTHO_PARA;
  temperature_col.resize (nlspec);
	temperature_col = COLTEMP;

  C_data.resize (nlspec);
  Double5 C_data_buffer = C_DATA;

  icol.resize (nlspec);
	icol = ICOL;
  jcol.resize (nlspec);
	jcol = JCOL;


	for (int l = 0; l < nlspec; l++)
	{
    irad[l].resize (nrad[l]);
    jrad[l].resize (nrad[l]);

		energy[l].resize (nlev[l]);
		weight[l].resize (nlev[l]);

		frequency[l].resize (nlev[l],nlev[l]);

		A[l].resize (nlev[l],nlev[l]);
		B[l].resize (nlev[l],nlev[l]);

    for (int i = 0; i < nlev[l]; i++)
		{
		  energy[l](i) =  energy_buffer[l][i];
		  weight[l](i) =  weight_buffer[l][i];

      for (int j = 0; j < nlev[l]; j++)
			{	
			  frequency[l](i,j) = frequency_buffer[l][i][j];

			  A[l](i,j) = A_buffer[l][i][j];
			  B[l](i,j) = B_buffer[l][i][j];
			}
		}


 	  num_col_partner[l].resize (ncolpar[l]);
    orth_or_para_H2[l].resize (ncolpar[l]);
    temperature_col[l].resize (ncolpar[l]);

    C_data[l].resize (ncolpar[l]);

    icol[l].resize (ncolpar[l]);
    jcol[l].resize (ncolpar[l]);


		for (int c = 0; c < ncolpar[l]; c++)
		{
      temperature_col[l][c].resize (ntmp[l][c]);

      C_data[l][c].resize (ntmp[l][c]);

      icol[l][c].resize (ncol[l][c]);
      jcol[l][c].resize (ncol[l][c]);

			for (int t = 0; t < ntmp[l][c]; t++)
			{
				C_data[l][c][t].resize(nlev[l],nlev[l]);

				for (int i = 0; i < nlev[l]; i++)
				{
          for (int j = 0; j < nlev[l]; j++)
					{
            C_data[l][c][t](i,j) = C_data_buffer[l][c][t][i][j];
					}
				}
			}
 		}
	}


}   // END OF CONSTRUCTOR




///  calc_Einstein_C: calculate the Einstein C coefficient
///    @param[in] species: data structure containing chamical species
///    @param[in] temperature_gas: local gas temperature
///    @param[in] p: number of the cell under consideration
///    @param[in] l: number of the line producing species under consideration
///    @return Einstein C collisional transition matrix
/////////////////////////////////////////////////////////////////////////////

MatrixXd LINEDATA ::
         calc_Einstein_C (const SPECIES& species, const double temperature_gas,
						              const long p, const int l) const
{

  MatrixXd C = MatrixXd :: Zero (nlev[l],nlev[l]);   // Einstein C_ij coefficient


  // Calculate H2 ortho/para fraction at equilibrium for given temperature

  double frac_H2_para  = 0.0;   // fraction of para-H2
  double frac_H2_ortho = 0.0;   // fraction of ortho-H2

	cout << "abn H2 = " << species.abundance[p][species.nr_H2] << endl;

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

    double abundance = species.density[p] * species.abundance[p][spec];

		cout << "den = " << species.density[p] << endl;
		cout << "sab = " << species.abundance[p][spec] << endl;
		cout << "n col par " << spec << endl;
		cout << "abn = " << abundance << endl;

    if      (orth_or_para_H2[l][c] == 'o')
    {
      abundance *= frac_H2_ortho;
    }

    else if (orth_or_para_H2[l][c] == 'p')
    {
      abundance *= frac_H2_para;
    }


    
    int t = search (temperature_col[l][c], 0, ntmp[l][c], temperature_gas);   


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
         calc_transition_matrix (const SPECIES& species, const double temperature_gas,
						                     const Double3& J_eff, const long p, const int l) const
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

    R(i,j) += B[l](i,j) * J_eff[p][l][k]; // - linedata.A[l](i,j)*Lambda(); 
    R(j,i) += B[l](j,i) * J_eff[p][l][k];
  }
	

	return R;

}
