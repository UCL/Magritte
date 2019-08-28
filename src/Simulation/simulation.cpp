// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


//#include <Eigen/QR>
#include <limits>

#include "simulation.hpp"
#include "Tools/debug.hpp"
#include "Tools/logger.hpp"
#include "Tools/Parallel/wrap_Grid.hpp"
#include "Tools/Parallel/wrap_omp.hpp"
#include "Tools/Parallel/wrap_mpi.hpp"
#include "Tools/Parallel/hybrid.hpp"
#include "Functions/heapsort.hpp"
#include "Functions/planck.hpp"


#include "sim_spectrum.tpp"
#include "sim_radiation.tpp"
#include "sim_lines.tpp"


///  Computer for the number of points on each ray
//////////////////////////////////////////////////
int Simulation ::
    compute_number_of_points_on_rays () const
{

  const long hnrays = parameters.nrays  () / 2;
  const long ncells = parameters.ncells ();

  const int nthrds = get_nthreads ();


  // Initialisation
  Long2 npoints (hnrays, Long1 (ncells));
  Long3 bins    (hnrays, Long2 (nthrds));

//   MPI_PARALLEL_FOR (r, hnrays);
//   {
//     for (int t = 0; t < nthrds; t++)
//     {
//       bins.reserve (ncells/nthrds);
//     }
//
// #   pragma omp parallel default (shared)
//     {
//       OMP_FOR (o, ncells)
//       {
//         const long           ar = geometry.rays.antipod[o][r];
//         const double dshift_max = get_dshift_max (o);
//
//         RayData rayData_r  = geometry.trace_ray <CoMoving> (o, r,  dshift_max);
//         RayData rayData_ar = geometry.trace_ray <CoMoving> (o, ar, dshift_max);
//
//         npoints[r][o] = rayData_ar.size() + rayData_r.size() + 1;
//
//         bins[r][omp_get_thread_num()].push_back ();
//       }
//     }
//   }


 return (0);

}
