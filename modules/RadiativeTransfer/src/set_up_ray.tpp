// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <vector>
using namespace std;
#include <Eigen/Core>
using namespace Eigen;

#include "set_up_ray.hpp"
#include "types.hpp"
#include "GridTypes.hpp"
#include "cells.hpp"
#include "lines.hpp"
#include "scattering.hpp"
#include "radiation.hpp"


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
int set_up_ray (const CELLS <Dimension, Nrays>& cells, FREQUENCIES& frequencies,
		            const TEMPERATURE& temperature, LINES& lines, const SCATTERING& scattering,
								RADIATION& radiation, const long o, const long r, const long R, const RAYTYPE raytype,
	              long& n, vReal2& Su, vReal2& Sv, vReal2& dtau)
{

	long   dir;
	double sign;


	if (raytype == ray)
	{
    dir  = r;  
	  sign =  1.0;
	}
	else if (raytype == antipod)
	{
		dir  = cells.rays.antipod[r];
	  sign = -1.0;
	}


	const long nfreq_red = frequencies.nfreq_red;


  double  Z = 0.0;   // distance from origin (o)
  double dZ = 0.0;   // last distance increment from origin (o)

  long current = o;                                   // current cell under consideration
  long next    = cells.next (o, dir, current, Z, dZ);   // next cell under consideration


	if (next != cells.ncells)   // if we are not going out of grid
	{

		vReal1 eta_c (nfreq_red);
		vReal1 chi_c (nfreq_red);
		
		lines.add_emissivity_and_opacity (frequencies, temperature, frequencies.all[o], o, eta_c, chi_c);

		scattering.add_opacity (frequencies.all[o], chi_c);


    vReal1 term1_c (nfreq_red);
	  vReal1 term2_c (nfreq_red);

		for (long f = 0; f < nfreq_red; f++)
    {    	
		 	term1_c[f] = (radiation.U[R][radiation.index(current,f)] + eta_c[f]) / chi_c[f];
			term2_c[f] =  radiation.V[R][radiation.index(current,f)]             / chi_c[f];
		}


		do
		{
      const double velocity = cells.relative_velocity (o, dir, next);
      const double    scale = 1.0 - velocity/CC;

			vReal1 frequencies_scaled (nfreq_red);

			for (long f = 0; f < nfreq_red; f++)
			{
			  frequencies_scaled[f] = scale * frequencies.all[o][f];
			}	


		  vReal1 eta_n (nfreq_red);
		  vReal1 chi_n (nfreq_red);

			lines.add_emissivity_and_opacity (frequencies, temperature,
					                              frequencies_scaled, o, eta_n, chi_n);

			scattering.add_opacity (frequencies_scaled, chi_n);  


			vReal1 U_scaled (nfreq_red);
			vReal1 V_scaled (nfreq_red);

      radiation.resample_U (frequencies, next, r, frequencies_scaled, U_scaled);
      radiation.resample_V (frequencies, next, r, frequencies_scaled, V_scaled);


			vReal1 term1_n (nfreq_red);
			vReal1 term2_n (nfreq_red);

			for (long f = 0; f < nfreq_red; f++)
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

			  for (long f = 0; f < nfreq_red; f++)
				{
			  	Su[n][f] += 2.0 / dtau[n][f] * (vZero - sign * 0.5 * (term2_c[f] + term2_n[f]));
				  Sv[n][f] += 2.0 / dtau[n][f] * (vZero - sign * 0.5 * (term1_c[f] + term1_n[f]));
				}
			}


      current = next;
      next    = cells.next (o, dir, current, Z, dZ);
  

			for (long f = 0; f < nfreq_red; f++)
			{
          chi_c[f] =   chi_n[f];
        term1_c[f] = term1_n[f];
        term2_c[f] = term2_n[f];
			}

      n++;
		}
	
    while (!cells.boundary[current]);
		
	} // end of if


	return (0);

}
