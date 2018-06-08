// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <vector>
using namespace std;
#include <Eigen/Core>
using namespace Eigen;

#include "set_up_ray.hpp"
#include "cells.hpp"
#include "lines.hpp"
#include "scattering.hpp"
#include "radiation.hpp"


const double tau_max = 1.0E9;


///  set_up_ray: extract sources and opacities from the grid on the ray
///    @param[in] cells: reference to the geometric cell data containing the grid
///    @param[in] frequencies: reference to data structure containing freqiencies
///    @param[in] lines: reference to data structure containing line transfer data
///    @param[in] scattering: reference to data structure containing scattering data
///    @param[in] radiation: reference to (previously calculated) radiation field
///    @param[in] o: number of the cell from which the ray originates
///    @param[in] r: number of the ray which is being set up
///    @param[in] sign: +1  if the ray is in the "right" direction, "-1" if opposite
///    @param[out] n: reference to the resulting number of points along the ray
///    @param[out] Su: reference to the source for u extracted along the ray
///    @param[out] Sv: reference to the source for v extracted along the ray
///    @param[out] dtau: reference to the optical depth increments along the ray 
////////////////////////////////////////////////////////////////////////////////////

template <int Dimension, long Nrays>
int set_up_ray (CELLS <Dimension, Nrays>& cells, FREQUENCIES& frequencies, TEMPERATURE& temperature,
		            LINES& lines, SCATTERING& scattering, RADIATION& radiation, long o, long r, double sign,
	              long& n, vector<vector<double>>& Su, vector<vector<double>>& Sv, vector<vector<double>>& dtau)
{

  //double tau  = 0.0;   // optical depth along ray

  double  Z = 0.0;   // distance from origin (o)
  double dZ = 0.0;   // last distance increment from origin (o)

  long current = o;
  long next    = cells.next (o, r, current, Z, dZ);


	if (next != cells.ncells)   // if we are not going out of grid
	{

		vector<double> eta_c (frequencies.nfreq);
		vector<double> chi_c (frequencies.nfreq);
		
		lines.add_emissivity_and_opacity (frequencies, temperature, frequencies.all[o], o, eta_c, chi_c);

		scattering.add_opacity (frequencies.all[o], chi_c);  


    vector<double> term1_c (frequencies.nfreq);
	  vector<double> term2_c (frequencies.nfreq);

		for (long f = 0; f < frequencies.nfreq; f++)
    {    	
		 	term1_c[f] = (radiation.U[r][current][f] + eta_c[f]) / chi_c[f];
			term2_c[f] =  radiation.V[r][current][f]             / chi_c[f];
		}


		do
		{
      const double velocity = cells.relative_velocity (o, r, next);
      const double    scale = 1.0 - velocity/CC;

			vector<double> frequencies_scaled (frequencies.nfreq);

			for (long f = 0; f < frequencies.nfreq; f++)
			{
			  frequencies_scaled[f] = scale * frequencies.all[o][f];
			}	


		  vector<double> eta_n (frequencies.nfreq);
		  vector<double> chi_n (frequencies.nfreq);

			lines.add_emissivity_and_opacity (frequencies, temperature, frequencies_scaled, o, eta_n, chi_n);

			scattering.add_opacity (frequencies_scaled, chi_n);  


			vector<double> U_scaled (frequencies.nfreq);
			vector<double> V_scaled (frequencies.nfreq);

      radiation.resample_U (frequencies, next, r, frequencies_scaled, U_scaled);
      radiation.resample_V (frequencies, next, r, frequencies_scaled, V_scaled);


			vector<double> term1_n (frequencies.nfreq);
			vector<double> term2_n (frequencies.nfreq);

			for (long f = 0; f < frequencies.nfreq; f++)
			{
				term1_n[f] = (U_scaled[f] + eta_n[f]) / chi_n[f];
        term2_n[f] =  V_scaled[f]             / chi_n[f];
				
				dtau[n][f] = 0.5 * dZ * PC *(chi_c[f] + chi_n[f]);
          Su[n][f] = 0.5 * (term1_n[f] + term1_c[f]) + sign * (term2_n[f] - term2_c[f]) / dtau[n][f];
       		Sv[n][f] = 0.5 * (term2_n[f] + term2_c[f]) + sign * (term1_n[f] - term1_c[f]) / dtau[n][f];
			}

 
			if (cells.boundary[next])
			{
				// Add boundary condition

			  for (long f = 0; f < frequencies.nfreq; f++)
				{
			  	Su[n][f] += 2.0 / dtau[n][f] * (0.0 - sign * 0.5 * (term2_c[f] + term2_n[f]));
				  Sv[n][f] += 2.0 / dtau[n][f] * (0.0 - sign * 0.5 * (term1_c[f] + term1_n[f]));
				}
			}


      current = next;
      next    = cells.next (o, r, current, Z, dZ);
  

			for (long f = 0; f < frequencies.nfreq; f++)
			{
          chi_c[f] =   chi_n[f];
        term1_c[f] = term1_n[f];
        term2_c[f] = term2_n[f];
			}

			//tau += dtau[n];
      n++;
		}
	
    while ( (!cells.boundary[current]) /*&& (tau < tau_max)*/ );
		
	} // end of if


	return (0);

}
