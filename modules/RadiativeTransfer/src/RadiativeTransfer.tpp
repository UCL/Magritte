// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <mpi.h>
#include <omp.h>
#include <vector>
#include<iostream>
using namespace std;
#include <Eigen/Core>
using namespace Eigen;

#include "RadiativeTransfer.hpp"
#include "timer.hpp"
#include "types.hpp"
#include "GridTypes.hpp"
#include "mpiTools.hpp"
#include "ompTools.hpp"
#include "cells.hpp"
#include "lines.hpp"
#include "scattering.hpp"
#include "radiation.hpp"
#include "image.hpp"
#include "set_up_ray.hpp"
#include "solve_ray.hpp"


///  RadiativeTransfer: solves the transfer equation for the radiation field
///    @param[in] cells: reference to the geometric cell data containing the grid
///    @param[in] temperature: data structure containing the temperature data
///    @param[in] frequencies: data structure containing the frequencies data
///    @param[in] lines: data structure containing the line transfer data
///    @param[in] scattering: data structure containing the scattering data
///    @param[in/out] radiation: reference to the  radiation field
///    @param[out] J: reference to the  radiation field
/////////////////////////////////////////////////////////////////////////////////

template <int Dimension, long Nrays>
int RadiativeTransfer (const CELLS <Dimension, Nrays> &cells,
                       const TEMPERATURE              &temperature,
	               const FREQUENCIES              &frequencies,
                       const LINES                    &lines,
                       const SCATTERING               &scattering,
	                     RADIATION                &radiation   )
{


  IMAGE image(cells.ncells, Nrays, frequencies.nfreq_red);


  // For all ray pairs

  for (long r = MPI_start (Nrays/2); r < MPI_stop (Nrays/2); r++)
  {
    const long R  = r - MPI_start (Nrays/2);

    const long ar = cells.rays.antipod[r];


    // Loop over all cells

#   pragma omp parallel                                                                \
    shared  (cells, temperature, frequencies, lines, scattering, radiation, r, image, cout)   \
    default (none)
    {

    for (long o = OMP_start (cells.ncells); o < OMP_stop (cells.ncells); o++)
    {
      vReal   Su [cells.ncells];   // effective source for u along ray r
      vReal   Sv [cells.ncells];   // effective source for v along ray r
      vReal dtau [cells.ncells];   // optical depth increment along ray r
  
      vReal Lambda [cells.ncells];

      long  cellNrs_r [cells.ncells];
      long    notch_r [cells.ncells];
      long   lnotch_r [cells.ncells];
      double shifts_r [cells.ncells];   // indicates where we are in frequency space
      double    dZs_r [cells.ncells];

      long  cellNrs_ar [cells.ncells];
      long    notch_ar [cells.ncells];
      long   lnotch_ar [cells.ncells];
      double shifts_ar [cells.ncells];   // indicates where we are in frequency space
      double    dZs_ar [cells.ncells];


      // Extract the cell on ray r and antipodal ar

      long n_r  = cells.on_ray (o, r,  cellNrs_r,  dZs_r);
      long n_ar = cells.on_ray (o, ar, cellNrs_ar, dZs_ar);

      const long ndep = n_r + n_ar;

      for (long q = 0; q < n_ar; q++)
      {
         notch_ar[q] = 0;
        lnotch_ar[q] = 0;
        shifts_ar[q] = 1.0 - cells.relative_velocity (o, ar, cellNrs_ar[q]) / CC;
      }

      for (long q = 0; q < n_r; q++)
      {
         notch_r[q] = 0;
        lnotch_r[q] = 0;
        shifts_r[q] = 1.0 - cells.relative_velocity (o, r, cellNrs_r[q]) / CC;
      }


      lnotch_r[cells.ncells]  = 0;
      lnotch_ar[cells.ncells] = 0;


      if (ndep > 1)
      {



        for (long f = 0; f < frequencies.nfreq_red; f++)
        {
          // Setup and solve the ray equations

          set_up_ray <Dimension, Nrays>
                     (cells, frequencies, temperature,lines, scattering, radiation, f, o, R,
                      lnotch_ar, notch_ar, cellNrs_ar, shifts_ar, dZs_ar, n_ar,
                      lnotch_r,  notch_r,  cellNrs_r,  shifts_r,  dZs_r,  n_r,
                      Su, Sv, dtau, ndep);


          const long ndiag = 0;

          solve_ray (ndep, Su, Sv, dtau, ndiag, Lambda, cells.ncells);

          
          // Store solution of the radiation field

          const long index = radiation.index(o,f);


          if ( (n_ar > 0) && (n_r > 0) )
          {
            radiation.u[R][index] = 0.5 * (Su[n_ar-1] + Su[n_ar]);
            radiation.v[R][index] = 0.5 * (Sv[n_ar-1] + Sv[n_ar]);
          }

          else if (n_ar > 0)   // and hence n_r  == 0
          {
            radiation.u[R][index] = Su[n_ar-1];
            radiation.v[R][index] = Sv[n_ar-1];
          }

          else if (n_r  > 0)   // and hence n_ar == 0
          {
            radiation.u[R][index] = Su[0];
            radiation.v[R][index] = Sv[0];
          }

          image.I_p[R][o][f] = Su[ndep-1] + Sv[ndep-1];
          image.I_m[R][o][f] = Su[ndep-1] - Sv[ndep-1];


        } // end of loop over frequencies

      }

      else if (ndep == 1)
      {
        // set up ray

	// trivially solve ray

      }

      else
      {
        const long b = cells.cell_to_bdy_nr[o];

        for (long f = 0; f < frequencies.nfreq_red; f++)
        {
            const long index = radiation.index(o,f);

            radiation.u[R][index] = 0.5 * (  radiation.boundary_intensity[r][b][f]
                                           + radiation.boundary_intensity[ar][b][f]);
            radiation.v[R][index] = 0.5 * (  radiation.boundary_intensity[r][b][f]
                                           - radiation.boundary_intensity[ar][b][f]);
        }
      }

      //long index = radiation.index(o,frequencies.nr_line[o][0][0][0]);

      //cout << radiation.u[R][index] << endl;

  //timer_SS.stop ();
  //timer_SS.`rint ();

  //timer_RT_CALC.stop ();
  //timer_RT_CALC.print_to_file ();


    }
    } // end of pragma omp parallel


  } // end of loop over ray pairs


  image.print("");

  // Reduce results of all MPI processes to get J, U and V
  
//  MPI_TIMER timer_RT_COMM ("RT_COMM");
//  timer_RT_COMM.start ();
  
  radiation.calc_J ();
  
  radiation.calc_U_and_V (scattering);
  
//  timer_RT_COMM.stop ();
//  timer_RT_COMM.print_to_file ();


  return (0);

}
