// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <limits>
using namespace std;

#include "constants.hpp"


///  next: find number of next cell on ray and its distance along ray
///    @param[in] origin: number of cell from which the ray originates
///    @param[in] r: number of the ray along which we are looking
///    @param[in] current: number of the cell put last on the ray
///    @param[in/out] Z: reference to the current distance along the ray
///    @param[out] dZ: reference to the distance increment to the next ray
///    @return number of the next cell on the ray after the current cell
/////////////////////////////////////////////////////////////////////////

inline long Cells ::
    next (const long    origin,
          const long    r,
          const long    current,
                double &Z,
                double &dZ      ) const
{

  // Pick neighbor on "right side" closest to ray

  double D_min = numeric_limits<double> :: max();   // Initialize to "infinity"

  long next = -1;   // return -1 when there is no next cell


  for (long n = 0; n < n_neighbors[current]; n++)
  {
    long nb = neighbors[current][n];

    double rvec[3];

    rvec[0] = x[nb] - x[origin];
    rvec[1] = y[nb] - y[origin];
    rvec[2] = z[nb] - z[origin];

    double Z_new =   rvec[0]*rays.x[r]
                   + rvec[1]*rays.y[r]
                   + rvec[2]*rays.z[r];

    if (Z < Z_new)
    {
      double rvec2 = rvec[0]*rvec[0] + rvec[1]*rvec[1] + rvec[2]*rvec[2];

      double D = rvec2 - Z_new*Z_new;

      if (D < D_min)
      {
        D_min = D;
        next  = nb;
        dZ    = Z_new - Z;   // such that dZ > 0.0
      }
    }

  } // end of n loop over neighbors


  // Update distance along ray

  Z = Z + dZ;


  return next;

}




///  relative_velocity: get relative velocity of current w.r.t. origin along ray
///    @param[in] origin: number of cell from which the ray originates
///    @param[in] r: number of the ray along which we are looking
///    @param[in] current: number of the cell for which we want the velocity
///    @return relative velocity of cell current w.r.t. cell origin
////////////////////////////////////////////////////////////////////////////////

inline double Cells ::
    doppler_shift         (
        const long origin,
        const long r,
        const long current) const
{

  return 1.0 - (  (vx[current] - vx[origin]) * rays.x[r]
                + (vy[current] - vy[origin]) * rays.y[r]
                + (vz[current] - vz[origin]) * rays.z[r]);

}




///  x_projected: x coordinate of the point p on the image in direction r
///    @param[in] p: number of cell to be projected on the image
///    @param[in] r: number of the ray orthogonal to the image
///    @return: x coordinate on the image
/////////////////////////////////////////////////////////////////////////

inline double Cells ::
    x_projected       (
        const long p,
        const long r  ) const
{

  return x[p]*rays.Ix[r] + y[p]*rays.Iy[r];

}




///  y_projected: y coordinate of the point p on the image in direction r
///    @param[in] p: number of cell to be projected on the image
///    @param[in] r: number of the ray orthogonal to the image
///    @return: y coordinate on the image
/////////////////////////////////////////////////////////////////////////

inline double Cells ::
    y_projected       (
        const long p,
        const long r  ) const
{

  return x[p]*rays.Jx[r] + y[p]*rays.Jy[r] + z[p]*rays.Jz[r];

}
