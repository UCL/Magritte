// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <math.h>

#include "rays.hpp"
#include "Tools/logger.hpp"
#include "Tools/Parallel/wrap_omp.hpp"


const string Rays::prefix = "Geometry/Rays/";


///  read: read the input into the data structure
///  @param[in] io: io object
///  @param[in] parameters: model parameters object
///////////////////////////////////////////////////

void Rays :: read (const Io &io, Parameters &parameters)
{
    cout << "Reading rays..." << endl;

    io.read_length (prefix+"rays", nrays);

    parameters.set_nrays (nrays);

    ncells = parameters.ncells ();


    rays.   resize (nrays);
    weights.resize (nrays);
    antipod.resize (nrays);


    // Read rays
    Double2 rays_array (nrays, Double1(3));

    io.read_array(prefix+"rays", rays_array);

    for (size_t r = 0; r < nrays; r++)
    {
        rays[r] = {rays_array[r][0], rays_array[r][1], rays_array[r][2]};
    }

    io.read_list (prefix+"weights", weights);


    // Setup rays
    setup ();
}




///  write: write out the data structure
///  @param[in] io: io object
/////////////////////////////////////////////////

void Rays :: write (const Io &io) const
{
    cout << "Writing rays..." << endl;

    Double2 rays_array (rays.size(), Double1(3));

    for (size_t r = 0; r < rays.size(); r++)
    {
        rays_array[r] = {rays[r][0], rays[r][1], rays[r][2]};
    }

    io.write_array (prefix+"rays",    rays_array);
    io.write_list  (prefix+"weights", weights   );
}




///  setup: setup data structure
////////////////////////////////

int Rays ::
    setup ()
{

  setup_antipodal_rays ();

  //setup_image_axis ();


  return (0);

}




///  setup_antipodal_rays: identify which rays are each others antipodes
////////////////////////////////////////////////////////////////////////

int Rays ::
    setup_antipodal_rays ()
{

  // (!) HEALPix vectors are not perfectly antipodal, so a tolerance is given
  const double tolerance = 1.0E-9;

//  OMP_PARALLEL_FOR (p, ncells)
//  {
    for (size_t r1 = 0; r1 < nrays; r1++)
    {
      for (size_t r2 = 0; r2 < nrays; r2++)
      {
        if ( (rays[r1] + rays[r2]).squaredNorm() < tolerance)
        {
          antipod[r1] = r2;
        }
      }
    }
//  }

  return (0);

}




// int Rays ::
//     setup_mirror_rays ()
// {
//
//   // Find mirror rays about xz-plane
//
//   for (long r1 = 0; r1 < Nrays; r1++)
//   {
//     for (long r2 = 0; r2 < Nrays; r2++)
//     {
//       if (    (fabs(x[r1]-x[r2]) < tolerance)
//            && (fabs(y[r1]+y[r2]) < tolerance)
//            && (fabs(z[r1]-z[r2]) < tolerance) )
//       {
//         mirror_xz[r1] = r2;
//       }
//     }
//   }
//
//   return (0);
//
// }
