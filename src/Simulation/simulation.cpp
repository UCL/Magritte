// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include "simulation.hpp"
#include "Tools/Parallel/wrap_omp.hpp"
#include "Tools/Parallel/wrap_omp.hpp"
#include "Tools/logger.hpp"

#include "sim_spectrum.tpp"
#include "sim_radiation.tpp"
#include "sim_lines.tpp"



///  Getter for the number of points on each ray pair in each point
///    @param[in] frame : frame of reference (for velocities)
///    ! Frame is required since different reference frames yield
///      different interpolations of the velocities and hence
///      differnet numbers of points along the ray pair
///////////////////////////////////////////////////////////////////

template <Frame frame>
Long2 Simulation ::
    get_npoints_on_rays () const
{

  const long hnrays = parameters.nrays  () / 2;
  const long ncells = parameters.ncells ();

  const int nthrds = get_nthreads ();


  Long2 npoints (hnrays, Long1 (ncells));


  MPI_PARALLEL_FOR (r, hnrays)
  {
    cout << "r = " << r << endl;

    OMP_FOR (o, ncells)
    {
      const long           ar = geometry.rays.antipod[o][r];
      const double dshift_max = get_dshift_max (o);

      RayData rayData_r  = geometry.trace_ray <frame> (o, r,  dshift_max);
      RayData rayData_ar = geometry.trace_ray <frame> (o, ar, dshift_max);

      npoints[r][o] = rayData_ar.size() + rayData_r.size() + 1;

      cout << "(r="<<r<<")    npoints = " << npoints[r][o] << endl;
    }
  }


  return npoints;

}




///  Getter for the maximum number of points on a ray pair
///    @param[in] frame : frame of reference (for velocities)
///    ! Frame is required since different reference frames yield
///      different interpolations of the velocities and hence
///      differnet numbers of points along the ray pair
///////////////////////////////////////////////////////////////////

template <Frame frame>
long Simulation ::
    get_max_npoints_on_rays () const
{

  const long hnrays = parameters.nrays  () / 2;
  const long ncells = parameters.ncells ();


  long  maximum = 0;
  Long2 npoints = get_npoints_on_rays <frame> ();


  MPI_PARALLEL_FOR (r, hnrays)
  {
    cout << "r = " << r << endl;

    OMP_FOR (o, ncells)
    {
      if (maximum < npoints[r][o])
      {
        maximum = npoints[r][o];
      }
    }
  }


  return maximum;

}
