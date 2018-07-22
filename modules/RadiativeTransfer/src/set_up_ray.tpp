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
///    @param[in] scattering: reference to structure containing scattering data
///    @param[in] radiation: reference to (previously calculated) radiation field
///    @param[in] o: number of the cell from which the ray originates
///    @param[in] r: number of the ray which is being set up
///		 @param[in] R: local number of the ray which is being set up
///    @param[in] raytype: indicates whether we walk forward or backward along ray
///    @param[out] n: reference to the resulting number of points along the ray
///    @param[out] Su: reference to the source for u extracted along the ray
///    @param[out] Sv: reference to the source for v extracted along the ray
///    @param[out] dtau: reference to the optical depth increments along the ray
//////////////////////////////////////////////////////////////////////////////////

//template <int Dimension, long Nrays>
//int set_up_ray (const CELLS <Dimension, Nrays>& cells, FREQUENCIES& frequencies,
//		            const TEMPERATURE& temperature, LINES& lines,
//								const SCATTERING& scattering, RADIATION& radiation,
//								const long f, long notch,
//								const long o, const long r, const long R, const RAYTYPE raytype,
//								vReal eta_c, vReal chi_c, vReal term1_c, vReal term2_c,
//								vReal eta_n, vReal chi_n, vReal term1_n, vReal term2_n,
//								vReal freq_scaled, vReal U_scaled, vReal V_scaled,
//	              long& n, vReal* Su, vReal* Sv, vReal* dtau)
//{
//
//	long   dir;
//	double sign;
//
//	//MPI_TIMER timer_0 ("0");
//	//timer_0.start ();
//
//	const long nfreq_red = frequencies.nfreq_red;
//
//	// Move these out !!! They might block the multithreadings
//
//
//	//timer_0.stop ();
//	//timer_0.print_to_file ();
//
//
//	if (raytype == ray)
//	{
//    dir  = r;
//	  sign =  1.0;
//	}
//	else if (raytype == antipod)
//	{
//		dir  = cells.rays.antipod[r];
//	  sign = -1.0;
//	}
//
//
//
//
//  double  Z = 0.0;   // distance from origin (o)
//  double dZ = 0.0;   // last distance increment from origin (o)
//
//	//MPI_TIMER timer_DO ("nxt");
//	//timer_DO.start ();
//
//	long current = o;                                   // current cell under consideration
//  long next    = cells.next (o, dir, current, Z, dZ);   // next cell under consideration
//
//	//timer_DO.stop ();
//	//timer_DO.print ();
//
//
//	if (next != cells.ncells)   // if we are not going out of grid
//	{
//
//		lines.add_emissivity_and_opacity (frequencies, temperature, frequencies.all[o][f], o, //eta_c, chi_c);
//
//		scattering.add_opacity (frequencies.all[o][f], chi_c);
//
//
//		term1_c = (radiation.U[R][radiation.index(current,f)] + eta_c) / chi_c;
//		term2_c =  radiation.V[R][radiation.index(current,f)]          / chi_c;
//
//	  //MPI_TIMER timer_DO ("do");
//	  //timer_DO.start ();
//
//		do
//		{
//      const double velocity = cells.relative_velocity (o, dir, next);
//      const double    scale = 1.0 - velocity/CC;
//
//	    //MPI_TIMER timer_ADD ("add");
//	    //timer_ADD.start ();
//
//			freq_scaled = scale * frequencies.all[o][f];
//
//			lines.add_emissivity_and_opacity (frequencies, temperature,
//					                              freq_scaled, o, eta_n, chi_n);
//
//			//timer_ADD.stop ();
//			//timer_ADD.print ();
//
//			scattering.add_opacity (freq_scaled, chi_n);
//
//
//      //radiation.rescale_U_and_V (frequencies, next, R, notch,
//			//	                         freq_scaled, U_scaled, V_scaled);
//
//
//			term1_n = (U_scaled + eta_n) / chi_n;
//      term2_n =  V_scaled          / chi_n;
//
//			dtau[n] = 0.5 * dZ * PC *(chi_c + chi_n);
//        Su[n] = 0.5 * (term1_n + term1_c) + sign * (term2_n - term2_c) / dtau[n];
//       	Sv[n] = 0.5 * (term2_n + term2_c) + sign * (term1_n - term1_c) / dtau[n];
//
//////				if (cells.boundary[o])
//////				{
//////				  cout << Su[n][f] << endl;
//////				  cout << Sv[n][f] << endl;
//////				}
//
//
//			if (cells.boundary[next])
//			{
//				// Add boundary condition
//
//				const long b = cells.cell_to_bdy_nr[next];
//
//				// Add something to account for Doppler shift in boundary intensity
//
//			  Su[n] += 2.0 / dtau[n] * (radiation.boundary_intensity[R][b][f]
//						                      - sign*0.5 * (term2_c + term2_n));
//				Sv[n] += 2.0 / dtau[n] * (radiation.boundary_intensity[R][b][f]
//						                      - sign*0.5 * (term1_c + term1_n));
//			}
//
//	  //  MPI_TIMER timer_NXT ("nxt");
//	  //  timer_NXT.start ();
//			//cout << "c" << next << endl;
//      current = next;
//      next    = cells.next (o, dir, current, Z, dZ);
//			//cout << "n" << next << endl;
//		//	timer_NXT.stop ();
//		//	timer_NXT.print ();
//
//
//        chi_c =   chi_n;
//      term1_c = term1_n;
//      term2_c = term2_n;
//
//      n++;
//		}
//
//    while (!cells.boundary[current]);
//	  //timer_DO.stop ();
//	  //timer_DO.print ();
//
//	} // end of if
//
//
//	return (0);
//
//}


template <int Dimension, long Nrays>
int get_cells_on_raypair (const CELLS <Dimension, Nrays>& cells,
								          const long o, const long r,
                          long *cellNrs, double *dZs, long& n)
{

  double  Z = 0.0;   // distance from origin (o)
	double dZ = 0.0;

	long next = cells.next (o, r, o, Z, dZ);   // next cell under consideration


	if (next != cells.ncells)   // if we are not going out of grid
	{
    cellNrs[n] = next;
        dZs[n] = dZ;   // last distance increment from origin (o)

    n++;

    while (!cells.boundary[next])
		{
      next = cells.next (o, r, next, Z, dZ);

      cellNrs[n] = next;
          dZs[n] = dZ;   // last distance increment from origin (o)

      n++;
		}
	}


	return (0);

}




template <int Dimension, long Nrays>
int set_up_ray (const CELLS<Dimension, Nrays>& cells, const RAYTYPE raytype,
	              FREQUENCIES& frequencies, const TEMPERATURE& temperature,
								LINES& lines, const SCATTERING& scattering, RADIATION& radiation,
								const long f, long *notch, const long o, const long R,
								long *cellNrs, double *shifts, double *dZ, long n_ray,
								vReal eta_c, vReal chi_c, vReal term1_c, vReal term2_c,
								vReal eta_n, vReal chi_n, vReal term1_n, vReal term2_n,
								vReal freq_scaled, vReal U_scaled, vReal V_scaled,
	              vReal* Su, vReal* Sv, vReal* dtau)
{

	double sign;

	const long nfreq_red = frequencies.nfreq_red;


	if (raytype == ray)
	{
	  sign =  1.0;
	}
	else if (raytype == antipod)
	{
	  sign = -1.0;
	}


	lines.add_emissivity_and_opacity (frequencies, temperature, frequencies.all[o][f], o,eta_c, chi_c);

	scattering.add_opacity (frequencies.all[o][f], chi_c);


	term1_c = (radiation.U[R][radiation.index(o,f)] + eta_c) / chi_c;
	term2_c =  radiation.V[R][radiation.index(o,f)]          / chi_c;


	for (long q = 0; q < n_ray; q++)
	{

		freq_scaled = shifts[q] * frequencies.all[o][f];

		lines.add_emissivity_and_opacity (frequencies, temperature,
				                              freq_scaled, cellNrs[q], eta_n, chi_n);

		scattering.add_opacity (freq_scaled, chi_n);

    radiation.rescale_U_and_V (frequencies, cellNrs[q], R, notch[q],
			                         freq_scaled, U_scaled, V_scaled);


		term1_n = (U_scaled + eta_n) / chi_n;
    term2_n =  V_scaled          / chi_n;

		dtau[q] = 0.5 * dZ[q] * PC *(chi_c + chi_n);
      Su[q] = 0.5 * (term1_n + term1_c) + sign * (term2_n - term2_c) / dtau[q];
     	Sv[q] = 0.5 * (term2_n + term2_c) + sign * (term1_n - term1_c) / dtau[q];

      chi_c =   chi_n;
    term1_c = term1_n;
    term2_c = term2_n;

  }


	// Add boundary condition

	const long b = cells.cell_to_bdy_nr[cellNrs[n_ray-1]];
	//cout << "n" << n << endl;
	//cout << "bdy nr" << b << endl;
	// Add something to account for Doppler shift in boundary intensity

	Su[n_ray-1] += 2.0 / dtau[n_ray-1] * (radiation.boundary_intensity[R][b][f]
			                                  - sign*0.5 * (term2_c + term2_n));
	Sv[n_ray-1] += 2.0 / dtau[n_ray-1] * (radiation.boundary_intensity[R][b][f]
			                                  - sign*0.5 * (term1_c + term1_n));


	return (0);

}
