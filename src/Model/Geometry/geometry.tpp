// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <limits>
#include <Eigen/Geometry>

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

      const long nxtnxt = nxt;

      crt = nxt;
      nxt = get_next (origin, ray, nxt, Z, dZ);

      if (nxt < 0)
      {
        cout << "--- ERROR ------------------------------------------" << endl;
        cout << "origin = " << origin << " nxt = " << nxtnxt << " ray = " << ray << endl;
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

    data.crt    = crt;
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
///    @param[in]     ray     : number of the ray along which we are looking
///    @param[in]     current : number of the cell put last on the ray
///    @param[in/out]  Z      : reference to the current distance along the ray
///    @param[out]    dZ      : reference to the distance increment to the next ray
///    @return number of the next cell on the ray after the current cell
///////////////////////////////////////////////////////////////////////////////////

inline long Geometry :: get_next (
    const long  origin,
    const long  ray,
    const long  current,
    double     &Z,
    double     &dZ               ) const
{
    if (spherical_symmetry)
    {
        return get_next_spherical_symmetry (origin, ray, current, Z, dZ);
    }
    else
    {
        return get_next_general            (origin, ray, current, Z, dZ);
    }
}




///  Getter for the number of the next cell on ray and its distance along ray when
///  assuming spherical symmetry and such that the positions are in ascending order!
///    @param[in]     origin  : number of cell from which the ray originates
///    @param[in]     ray     : number of the ray along which we are looking
///    @param[in]     current : number of the cell put last on the ray
///    @param[in/out]  Z      : reference to the current distance along the ray
///    @param[out]    dZ      : reference to the distance increment to the next ray
///    @return number of the next cell on the ray after the current cell
///////////////////////////////////////////////////////////////////////////////////

inline long Geometry :: get_next_spherical_symmetry (
    const long  origin,
    const long  ray,
    const long  current,
    double     &Z,
    double     &dZ                                  ) const
{
    // Pick neighbor on "right side" closest to ray
    long next;

    const double Rsin = cells.position[origin].cross(rays.rays[ray]).z();
    const double Rcos = cells.position[origin].dot  (rays.rays[ray]);

//    if (Rcos == 0)
//    {
//        if (Rsin > 0)
//        {
//            if (current == cells.position.size()-1) return (-1);
//            next = current + 1;
//            dZ   = cells.position[next].x() - cells.position[current].x();
//        }
//        else
//        {
//            if (current == 0) return (-1);
//            next = current - 1;
//            dZ   = cells.position[current].x() - cells.position[next].x();
//        }
//    }
//    else
//    {
        const double Rsin2       = Rsin * Rsin;
        const double Rcos_plus_Z = Rcos + Z;

        if (Z < -Rcos)
        {
            if (current <= 0 ) return (-1);

            if (cells.position[current-1].squaredNorm() >= Rsin2)
            {
                next = current - 1;
                dZ   = -sqrt(cells.position[next].squaredNorm() - Rsin2) - Rcos_plus_Z;
            }
            else
            {
                next = current;
                dZ   = - 2.0 * Rcos_plus_Z;
            }
        }
        else
        {
            if (current >= cells.position.size()-1) return (-1);
            next = current + 1;
            dZ   = +sqrt(cells.position[next].squaredNorm() - Rsin2) - Rcos_plus_Z;
        }
//    }

    // Update distance along ray
    Z = Z + dZ;

    cout << "r = " << ray << "   o = " << origin << "   c = " << current << "   n = " << next << endl;

    return next;
}




///  Getter for the number of the next cell on ray and its distance along ray in
///  the general case without any further assumptions
///    @param[in]     origin  : number of cell from which the ray originates
///    @param[in]     ray       : number of the ray along which we are looking
///    @param[in]     current : number of the cell put last on the ray
///    @param[in/out]  Z      : reference to the current distance along the ray
///    @param[out]    dZ      : reference to the distance increment to the next ray
///    @return number of the next cell on the ray after the current cell
///////////////////////////////////////////////////////////////////////////////////

inline long Geometry ::
    get_next_general (
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
  //double Z_new_max = 0.0;
  //long   n_new_max = 0;
  ///////////////////////

  for (const long neighbor : cells.neighbors[current])
  {
    const Vector3d R = cells.position[neighbor] - cells.position[origin];

    const double Z_new = R.dot(rays.rays[ray]);

    ///////////////////////
    //if (Z_new > Z_new_max)
    //{
    //  Z_new_max = Z_new;
    //  n_new_max = neighbor;
    //}
    ///////////////////////

    if (Z_new > Z)
    {
      const double distance_from_ray2 = R.dot(R) - Z_new*Z_new;

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
  //if (next == -1)
  //{
  //  cout << "Intervening!" << endl;
  //  cout << "cell with nxt=-1 was : " << current << endl;

  //  // Just do one try with the furthest points (largest Z)

  //  for (const long neighbor : cells.neighbors[n_new_max])
  //  {
  //    if (neighbor != current)
  //    {
  //      const double x = cells.x[neighbor] - cells.x[origin];
  //      const double y = cells.y[neighbor] - cells.y[origin];
  //      const double z = cells.z[neighbor] - cells.z[origin];

  //      const double Z_new =  x * rays.x[origin][ray]
  //                          + y * rays.y[origin][ray]
  //                          + z * rays.z[origin][ray];

  //      if (Z_new > Z)
  //      {
  //        const double distance_from_ray2 = (x*x + y*y + z*z) - Z_new*Z_new;

  //        if (distance_from_ray2 < dmin)
  //        {
  //          dmin = distance_from_ray2;
  //          next = neighbor;
  //          dZ   = Z_new - Z;   // such that dZ > 0.0
  //        }
  //      }
  //    }
  //  }

  //}
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
    return 1.0 - (cells.velocity[current]-cells.velocity[origin]).dot(rays.rays[ray]);
  }

  // Rest frame implementation

  if (frame == Rest)
  {
    // In the rest frame the direction of the projected should be fixed
    // We choose to fix it to "up the ray"

    long ray_correct = ray;

    if (ray >= nrays/2)
    {
      ray_correct = rays.antipod[ray];
    }

    return 1.0 - cells.velocity[current].dot(rays.rays[ray_correct]);
  }

}




inline size_t get_required_npoints (
        const double   shift_crt,
        const double   shift_nxt,
        const double   dshift_max  )
{
    const double dshift     = shift_nxt - shift_crt;
    const double dshift_abs = fabs (dshift);

    // If velocity gradient is not well-sampled enough
    if (dshift_abs > dshift_max)
    {
        // Interpolate velocity gradient field
        const size_t n_interpol = dshift_abs/dshift_max + 1;

        if (n_interpol > 10000) {throw "Too many (> 10 000) interpolations needed!";}

        return n_interpol;
    }

    else
    {
        return 1;
    }
}




template <Frame frame>
inline size_t Geometry :: get_npoints_on_ray (
        const size_t origin,
        const size_t ray,
        const double dshift_max              ) const
{
    size_t npoints = 0;

    // Find projected cells on ray
    double  Z = 0.0;   // distance from origin (o)
    double dZ = 0.0;   // last increment in Z

    long nxt = get_next (origin, ray, origin, Z, dZ);

    if (nxt != -1)   // if we are not going out of grid
    {
        double shift_crt = get_doppler_shift <frame> (origin, ray, origin);
        double shift_nxt = get_doppler_shift <frame> (origin, ray, nxt   );

        npoints += get_required_npoints (shift_crt, shift_nxt, dshift_max);

        while (!boundary.boundary[nxt])   // while we have not hit the boundary
        {
            shift_crt = shift_nxt;

            const long nxtnxt = nxt;

            nxt = get_next (origin, ray, nxt, Z, dZ);

            if (nxt < 0)
            {
                cout << "--- ERROR ------------------------------------------" << endl;
                cout << "origin = " << origin << " nxt = " << nxtnxt << " ray = " << ray << endl;
                cout << " (nxt<0) No proper neighbor found inside the mesh!  " << endl;
                cout << "                                                    " << endl;
                cout << "----------------------------------------------------" << endl;
            }

            shift_nxt = get_doppler_shift <frame> (origin, ray, nxt);

            npoints += get_required_npoints (shift_crt, shift_nxt, dshift_max);
        }
    }

    return npoints;
}




