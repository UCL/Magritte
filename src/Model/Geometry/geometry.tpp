// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <limits>

#include "Tools/logger.hpp"


inline RayData Geometry ::
    trace_ray (
        const long   origin,
        const long   ray,
        const double dshift_max) const
{

  RayData rayData;


  // Find projected cells on ray

  double  Z = 0.0;   // distance from origin (o)
  double dZ = 0.0;   // last increment in Z

  long nxt = next (origin, ray, origin, Z, dZ);


  if (nxt != -1)   // if we are not going out of grid
  {
    double shift_crt = 1.0;
    long   crt       = origin;
    double shift_nxt = doppler_shift (origin, ray, nxt);

    set_data (crt, nxt, shift_crt, shift_nxt, dZ, dshift_max, rayData);


    while (!boundary.boundary[nxt])   // while we have not hit the boundary
    {
      shift_crt = shift_nxt;
      nxt       = next          (origin, ray, nxt, Z, dZ);
      shift_nxt = doppler_shift (origin, ray, nxt);

      set_data (crt, nxt, shift_crt, shift_nxt, dZ, dshift_max, rayData);

//      cout << nxt << endl;
    }
  }


  return rayData;

}




inline void Geometry ::
    set_data (
        const long     crt,
        const long     nxt,
        const double   shift_crt,
        const double   shift_nxt,
        const double   dZ_loc,
        const double   dshift_max,
              RayData &rayData    ) const
{

  ProjectedCellData data;

  const double dshift     = shift_nxt - shift_crt;
  const double dshift_abs = fabs (dshift);


  // If velocity gradient is not well-sampled enough

  if (dshift_abs > dshift_max)
  {

    // Interpolate velocity gradient field
    const long        n_interpl = dshift_abs / dshift_max + 1;
    const long   half_n_interpl =        0.5 * n_interpl;
    const double     dZ_interpl =     dZ_loc / n_interpl;
    const double dshift_interpl =     dshift / n_interpl;


    // Assign current cell to first half of interpolation points
    for (long m = 1; m < half_n_interpl; m++)
    {
      data.cellNr = crt;
      data.shift  = shift_crt + m * dshift_interpl;
      data.dZ     = dZ_interpl;

      data.lnotch = 0;               // CHECK IF THIS IS NECESSARY !!!
      data.notch  = 0;               // CHECK IF THIS IS NECESSARY !!!

      rayData.push_back (data);
    }


    // Assign next cell to second half of interpolation points
    for (long m = half_n_interpl; m <= n_interpl; m++)
    {
      data.cellNr = nxt;
      data.shift  = shift_crt + m * dshift_interpl;
      data.dZ     = dZ_interpl;

      data.lnotch = 0;               // CHECK IF THIS IS NECESSARY !!!
      data.notch  = 0;               // CHECK IF THIS IS NECESSARY !!!

      rayData.push_back (data);
    }
  }

  else
  {
    data.cellNr = nxt;
    data.shift  = shift_nxt;
    data.dZ     = dZ_loc;

    data.lnotch = 0;               // CHECK IF THIS IS NECESSARY !!!
    data.notch  = 0;               // CHECK IF THIS IS NECESSARY !!!

    rayData.push_back (data);
  }

}




///  next: find number of next cell on ray and its distance along ray
///    @param[in] origin: number of cell from which the ray originates
///    @param[in] r: number of the ray along which we are looking
///    @param[in] current: number of the cell put last on the ray
///    @param[in/out] Z: reference to the current distance along the ray
///    @param[out] dZ: reference to the distance increment to the next ray
///    @return number of the next cell on the ray after the current cell
/////////////////////////////////////////////////////////////////////////

inline long Geometry ::
    next (const long    origin,
          const long    ray,
          const long    current,
                double &Z,
                double &dZ      ) const
{

  // Pick neighbor on "right side" closest to ray

  double dmin = std::numeric_limits<double>::max();   // Initialize to "infinity"

  long next = -1;   // return -1 when there is no next cell


  for (long n = 0; n < cells.n_neighbors[current]; n++)
  {
    long neighbor = cells.neighbors[current][n];

    double position[3];

    position[0] = cells.x[neighbor] - cells.x[origin];
    position[1] = cells.y[neighbor] - cells.y[origin];
    position[2] = cells.z[neighbor] - cells.z[origin];

    double Z_new =   position[0]*rays.x[origin][ray]
                   + position[1]*rays.y[origin][ray]
                   + position[2]*rays.z[origin][ray];

    if (Z_new > Z)
    {
      double distance_from_origin2 =   position[0]*position[0]
                                     + position[1]*position[1]
                                     + position[2]*position[2];

      double distance_from_ray2 = distance_from_origin2 - Z_new*Z_new;

      if (distance_from_ray2 < dmin)
      {
        dmin = distance_from_ray2;
        next = neighbor;
        dZ   = Z_new - Z;   // such that dZ > 0.0
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

inline double Geometry ::
    doppler_shift (
        const long origin,
        const long ray,
        const long current) const
{

  return 1.0 - (  (cells.vx[current] - cells.vx[origin]) * rays.x[origin][ray]
                + (cells.vy[current] - cells.vy[origin]) * rays.y[origin][ray]
                + (cells.vz[current] - cells.vz[origin]) * rays.z[origin][ray]);

}
