// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <fstream>
#include <iostream>
#include <string>
#include <omp.h>

#include "../src/timer.hpp"
#include "../src/RadiativeTransfer.hpp"
#include "../src/cells.hpp"
#include "../src/medium.hpp"
#include "../src/radiation.hpp"


int read_data (/*CELLS *cells,std::string file_name*/)
{
  std::ifstream infile ("thefile.txt");

  int a, b;

  while (infile >> a >> b)
  {
    std::cout << a << b << std::endl;
  }

}



int main (void)
{
//
//	// Setup data
//
//	const int Dimension = 1;
//	const long Nrays    = 2;
//	const long Nfreq    = 10;
//
//	const long ncells   = 10;
//
//	long nfreq_l = 1;
//	long nfreq_c = 1;
//	long nfreq_s = 1;
//
//	CELLS <Dimension, Nrays> cells (ncells);
//	
//	cells.initialize ();
//
// 
//	for (long p = 0; p < ncells; p++)
//	{
//		cells.x[p] = 1.23 * p;
//	}
//
//  cells.boundary[0]        = true;
//  cells.boundary[ncells-1] = true;
//
//	cells.neighbor[RINDEX(0,0)]        = 1;
//	cells.neighbor[RINDEX(ncells-1,0)] = ncells-2;
//
//
//	for (long p = 1; p < ncells-1; p++)
//	{
//		cells.neighbor[RINDEX(p,0)] = p-1;
//		cells.neighbor[RINDEX(p,1)] = p+1;
//	}
//
//
//  RADIATION radiation (ncells, Nrays, Nfreq);
//
//
//	long freq[Nfreq];
//	long rays[Nrays] = {0, 1};
//

//	RadiativeTransfer <Dimension, Nrays, Nfreq>
//										(cells, medium, Nrays, rays, d_radiation, a_radiation);

  /* radiation.U_d[RC(r,o,f)] += chi_s/chi_e * medium->Phi_scat(,r,freq) * u_loc; */
  /* radiation.V_d[RC(r,o,f)] += chi_s/chi_e * medium->Phi_scat(,r,freq) * v_loc; */


  return(0);

}
