// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <vector>
using namespace std;
#include <Eigen/Core>
using namespace Eigen;

#include "declarations.hpp"
#include "set_up_ray.hpp"
#include "cells.hpp"
#include "medium.hpp"
#include "radiation.hpp"


#define RCF(r,c,f) ( Nfreq*(cells.ncells*(r) + (c)) + (f) )
#define FC(c,f) ( cells.ncells*(c) + (f) )

const double tau_max = 1.0E9;


///  set_up_ray: extract sources and opacities from the grid on the ray
///    @param[in] cells: reference to the geometric cell data containing the grid
///    @param[in] radiation: reference to (previously calculated) radiation field
///    @param[in] medium: reference to the opacity and emissivity data of the medium
///    @param[in] o: number of the cell from which the ray originates
///    @param[in] r: number of the ray which is being set up
///    @param[in] f: number of the frequency bin which is used
///    @param[in] sign: +1  if the ray is in the "right" direction, "-1" if opposite
///    @param[out] n: reference to the resulting number of points along the ray
///    @param[out] Su: reference to the source for u extracted along the ray
///    @param[out] Sv: reference to the source for v extracted along the ray
///    @param[out] dtau: reference to the optical depth increments along the ray 
////////////////////////////////////////////////////////////////////////////////////

template <int Dimension, long Nrays, long Nfreq>
int set_up_ray (CELLS <Dimension, Nrays>& cells, RADIATION& radiation,
		            MEDIUM& medium, long o, long r, long f, double sign,
	              long& n, vector<double>& Su, vector<double>& Sv, vector<double>& dtau)
{

  double tau  = 0.0;   // optical depth along ray

  double  Z = 0.0;   // distance from origin (o)
  double dZ = 0.0;   // last distance increment from origin (o)

  long current = o;
  long next    = cells.next (o, r, current, Z, dZ);


	if (next != cells.ncells)   // if we are not going out of grid
	{
    long s_c = current; //LSPECGRIDRAD(ls,current,kr);

		const double freq = radiation.frequencies[f];

		double chi_c =   medium.chi_line (current, freq)
		               + medium.chi_cont (current, freq)
									 + medium.chi_scat (current, freq);
		
		double eta_c =   medium.eta_line (current, freq)
		               + medium.eta_cont (current, freq);

		double term1_c = (radiation.U(current, r, freq) + eta_c) / chi_c;
		double term2_c =  radiation.V(current, r, freq)          / chi_c;


		do
		{
      const double velocity = cells.relative_velocity (o, r, next);
    
			const double nu = (1.0 - velocity/CC)*freq;

			const double chi_n =   medium.chi_line (next, nu)
				                   + medium.chi_cont (next, nu)
													 + medium.chi_scat (next, nu);
			
			const double eta_n =   medium.eta_line (next, nu)
				                   + medium.eta_cont (next, nu);

			const double term1_n = (radiation.U(next, r, nu) + eta_n) / chi_n;
			const double term2_n =  radiation.V(next, r, nu)          / chi_n;

      dtau[n] = 0.5 * dZ * PC *(chi_c + chi_n);
        Su[n] = 0.5 * (term1_n + term1_c) + sign * (term2_n - term2_c) / dtau[n];
        Sv[n] = 0.5 * (term2_n + term2_c) + sign * (term1_n - term1_c) / dtau[n];
 
			if (cells.boundary[next])
			{
				// Add boundary condition

				Su[n] += 2.0 / dtau[n] * (0.0 - sign * 0.5 * (term2_c + term2_n));
				Sv[n] += 2.0 / dtau[n] * (0.0 - sign * 0.5 * (term1_c + term1_n));
			}

      current = next;
      next    = cells.next (o, r, current, Z, dZ);
  
        chi_c =   chi_n;
      term1_c = term1_n;
      term2_c = term2_n;

			tau += dtau[n];
      n++;
		}
	
    while ( (!cells.boundary[current]) && (tau < tau_max) );
	}  


	return (0);

}
