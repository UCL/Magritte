// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <limits>

#include "Tools/logger.hpp"
#include "Tools/constants.hpp"
#include <iomanip>


template <Frame frame>
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

  long nxt = get_next (origin, ray, origin, Z, dZ);

  // cout << "nxt " << nxt << endl;

  if (nxt != -1)   // if we are not going out of grid
  {
    long   crt       = origin;
    double shift_crt = get_doppler_shift <frame> (origin, ray, crt);
    double shift_nxt = get_doppler_shift <frame> (origin, ray, nxt);


    //cout << std::scientific << std::setprecision (16);
    //cout << "shift_crt " << shift_crt << endl;
    //cout << "shift_nxt " << shift_nxt << endl;
    //cout << cells.vx[crt]*CC << "   " << cells.vy[crt]*CC << "   " << cells.vz[crt]*CC << endl;
    //cout << cells.vx[nxt]*CC << "   " << cells.vy[nxt]*CC << "   " << cells.vz[nxt]*CC << endl;

    //cout << "----------------------------" << endl;

    set_data (crt, nxt, shift_crt, shift_nxt, dZ, dshift_max, rayData);


    while (!boundary.boundary[nxt])   // while we have not hit the boundary
    {
      shift_crt = shift_nxt;
      nxt       = get_next (origin, ray, nxt, Z, dZ);

      if (nxt < 0)
      {
        cout << "--- ERROR ------------------------------------------" << endl;
        cout << " (nxt<0) No proper neighbor found inside the mesh!  " << endl;
        cout << "                                                    " << endl;
        cout << "----------------------------------------------------" << endl;
      }

      shift_nxt = get_doppler_shift <frame> (origin, ray, nxt);

      // cout << "nxt " << nxt << endl;
      //cout << std::scientific << std::setprecision (16);
      //cout << "shift_crt " << shift_crt << endl;
      //cout << "shift_nxt " << shift_nxt << endl;
      //cout << cells.vx[crt]*CC << "   " << cells.vy[crt]*CC << "   " << cells.vz[crt]*CC << endl;
      //cout << cells.vx[nxt]*CC << "   " << cells.vy[nxt]*CC << "   " << cells.vz[nxt]*CC << endl;

      set_data (crt, nxt, shift_crt, shift_nxt, dZ, dshift_max, rayData);

      //cout << nxt << endl;
    }
  }


  return rayData;

}




inline int Geometry ::
    set_data (
        const long     crt,
        const long     nxt,
        const double   shift_crt,
        const double   shift_nxt,
        const double   dZ_loc,
        const double   dshift_max,
              RayData &rayData    ) const
{

  //cout << "Can data be set?" << endl;

  ProjectedCellData data;

  const double dshift     = shift_nxt - shift_crt;
  const double dshift_abs = fabs (dshift);

  //cout << " shift_nxt = " <<  shift_nxt << endl;
  //cout << " shift_crt = " <<  shift_crt << endl;
  //cout << "dshift_max = " << dshift_max << endl;
  //cout << "dshift_abs = " << dshift_abs << endl;
  //cout << "Let's try the if statement" << endl;

  // If velocity gradient is not well-sampled enough

  if (dshift_abs > dshift_max)
  {
    //cout << "dshift_abs > dshift_max" << endl;

    // Interpolate velocity gradient field
    const long        n_interpl = dshift_abs / dshift_max + 1;
    const long   half_n_interpl =        0.5 * n_interpl;
    const double     dZ_interpl =     dZ_loc / n_interpl;
    const double dshift_interpl =     dshift / n_interpl;



    if ( (n_interpl > 10000) ||
         (n_interpl <     0)    )
    {
      cout << "--- ERROR ------------------------------------------" << endl;
      cout << "Too many (> 10000) interpolations needed!"            << endl;
      cout << "or dshift_max is negative (probably due to overflow)" << endl;
      cout << "----------------------------------------------------" << endl;

      return (-1);
    }

    //cout << "        n_interpl = " <<       n_interpl << endl;
    //cout << "   half_n_interpl = " <<  half_n_interpl << endl;
    //cout << "       dZ_interpl = " <<      dZ_interpl << endl;
    //cout << "   dshift_interpl = " <<  dshift_interpl << endl;


    // Assign current cell to first half of interpolation points
    for (long m = 1; m < half_n_interpl; m++)
    {
      data.cellNr = crt;
      data.shift  = shift_crt + m * dshift_interpl;
      data.dZ     = dZ_interpl;

      data.lnotch = 0;               // CHECK IF THIS IS NECESSARY !!!
      data.notch  = 0;               // CHECK IF THIS IS NECESSARY !!!

      rayData.push_back (data);

      //cout << "m = " << m << endl;
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

      //cout << "m = " << m << endl;
    }
  }

  else
  {
    //cout << "else..." << endl;

    data.cellNr = nxt;
    data.shift  = shift_nxt;
    data.dZ     = dZ_loc;

    data.lnotch = 0;               // CHECK IF THIS IS NECESSARY !!!
    data.notch  = 0;               // CHECK IF THIS IS NECESSARY !!!

    rayData.push_back (data);

    //cout << "just added it (no interpolation)" << endl;
  }


  return (0);

}




///  Getter for the number of the next cell on ray and its distance along ray
///    @param[in]     origin  : number of cell from which the ray originates
///    @param[in]     r       : number of the ray along which we are looking
///    @param[in]     current : number of the cell put last on the ray
///    @param[in/out] Z       : reference to the current distance along the ray
///    @param[out]    dZ      : reference to the distance increment to the next ray
///    @return number of the next cell on the ray after the current cell
///////////////////////////////////////////////////////////////////////////////////

inline long Geometry ::
    get_next (
        const long    origin,
        const long    ray,
        const long    current,
              double &Z,
              double &dZ      ) const
{

  // Pick neighbor on "right side" closest to ray

  double dmin = std::numeric_limits<double>::max();   // Initialize to "infinity"
  long   next = -1;                                   // return -1 when there is no next cell

  ///////////////////////
  double Z_new_max = 0.0;
  long   n_new_max = 0;
  ///////////////////////

  for (const long neighbor : cells.neighbors[current])
  {
    const double x = cells.x[neighbor] - cells.x[origin];
    const double y = cells.y[neighbor] - cells.y[origin];
    const double z = cells.z[neighbor] - cells.z[origin];

    const double Z_new =  x * rays.x[origin][ray]
                        + y * rays.y[origin][ray]
                        + z * rays.z[origin][ray];

    ///////////////////////
    if (Z_new > Z_new_max)
    {
      Z_new_max = Z_new;
      n_new_max = neighbor;
    }
    ///////////////////////

    if (Z_new > Z)
    {
      const double distance_from_ray2 = (x*x + y*y + z*z) - Z_new*Z_new;

      if (distance_from_ray2 < dmin)
      {
        dmin = distance_from_ray2;
        next = neighbor;
        dZ   = Z_new - Z;   // such that dZ > 0.0
      }
    }
  }


  //////////////////////////////////////////////////
  // Try to catch the error of no neighbors found.
  //////////////////////////////////////////////////
  if (next == -1)
  {

    // Just do one try with the furthest points (largest Z)

    for (const long neighbor : cells.neighbors[n_new_max])
    {
      if (neighbor != current)
      {
        const double x = cells.x[neighbor] - cells.x[origin];
        const double y = cells.y[neighbor] - cells.y[origin];
        const double z = cells.z[neighbor] - cells.z[origin];

        const double Z_new =  x * rays.x[origin][ray]
                            + y * rays.y[origin][ray]
                            + z * rays.z[origin][ray];

        if (Z_new > Z)
        {
          const double distance_from_ray2 = (x*x + y*y + z*z) - Z_new*Z_new;

          if (distance_from_ray2 < dmin)
          {
            dmin = distance_from_ray2;
            next = neighbor;
            dZ   = Z_new - Z;   // such that dZ > 0.0
          }
        }
      }
      // cout << "cell with nxt=-1 was : " << current << endl;
    }
  }
  //////////////////////////////////////////////////

  // Update distance along ray

  Z = Z + dZ;


  return next;

}




///  Getter for the doppler shift along the ray between the current cell and the origin
///    @param[in] origin  : number of cell from which the ray originates
///    @param[in] r       : number of the ray along which we are looking
///    @param[in] current : number of the cell for which we want the velocity
///    @return doppler shift along the ray between the current cell and the origin
///////////////////////////////////////////////////////////////////////////////////////

template <Frame frame>
inline double Geometry ::
    get_doppler_shift (
        const long  origin,
        const long  ray,
        const long  current) const
{

  // Co-moving frame implementation

  if (frame == CoMoving)
  {
    return 1.0 - (  (cells.vx[current] - cells.vx[origin]) * rays.x[origin][ray]
                  + (cells.vy[current] - cells.vy[origin]) * rays.y[origin][ray]
                  + (cells.vz[current] - cells.vz[origin]) * rays.z[origin][ray]);
  }

  // Rest frame implementation

  if (frame == Rest)
  {
    // In the rest frame the direction of the projected should be fixed
    // We choose to fix it to "up the ray"

    long ray_correct = ray;

    if (ray >= nrays/2)
    {
      ray_correct = rays.antipod[origin][ray];
    }

    return 1.0 - (  cells.vx[current] * rays.x[origin][ray_correct]
                  + cells.vy[current] * rays.y[origin][ray_correct]
                  + cells.vz[current] * rays.z[origin][ray_correct]);
  }

}
