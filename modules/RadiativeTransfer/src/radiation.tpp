// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include "mpiTools.hpp"
#include "raypair.hpp"
#include "GridTypes.hpp"
#include "interpolation.hpp"
#include "image.hpp"



inline long RADIATION ::
    index            (
        const long p,
        const long f ) const
{
  return f + p*nfreq_red;
}




inline int RADIATION ::
    rescale_U_and_V                    (
        const FREQUENCIES &frequencies,
        const long         p,
        const long         R,
              long        &notch,
        const vReal       &freq_scaled,
              vReal       &U_scaled,
              vReal       &V_scaled    ) const

#if (GRID_SIMD)

{

  vReal nu1, nu2, U1, U2, V1, V2;

  for (int lane = 0; lane < n_simd_lanes; lane++)
  {

    const double freq = freq_scaled.getlane (lane);

    search_with_notch (frequencies.nu[p], notch, freq);

    const long f1    =  notch    / n_simd_lanes;
    const  int lane1 =  notch    % n_simd_lanes;

    const long f2    = (notch-1) / n_simd_lanes;
    const  int lane2 = (notch-1) % n_simd_lanes;

    //const double nu1 = frequencies.nu[p][f1].getlane(lane1);
    //const double nu2 = frequencies.nu[p][f2].getlane(lane2);
    
    //const double U1 = U[R][index(p,f1)].getlane(lane1);
    //const double U2 = U[R][index(p,f2)].getlane(lane2);
    
    //const double V1 = V[R][index(p,f1)].getlane(lane1);
    //const double V2 = V[R][index(p,f2)].getlane(lane2);
    
    //U_scaled.putlane(interpolate_linear (nu1, U1, nu2, U2, freq), lane);
    //V_scaled.putlane(interpolate_linear (nu1, V1, nu2, V2, freq), lane);
    
    nu1.putlane (frequencies.nu[p][f1].getlane (lane1), lane);
    nu2.putlane (frequencies.nu[p][f2].getlane (lane2), lane);
    
     U1.putlane (U[R][index(p,f1)].getlane (lane1), lane);
     U2.putlane (U[R][index(p,f2)].getlane (lane2), lane);
    
     V1.putlane (V[R][index(p,f1)].getlane (lane1), lane);
     V2.putlane (V[R][index(p,f2)].getlane (lane2), lane);
  }

  U_scaled = interpolate_linear (nu1, U1, nu2, U2, freq_scaled);
  V_scaled = interpolate_linear (nu1, V1, nu2, V2, freq_scaled);


  return (0);

}

#else

{

  search_with_notch (frequencies.nu[p], notch, freq_scaled);

  const long f1    = notch;
  const long f2    = notch-1;

  const double nu1 = frequencies.nu[p][f1];
  const double nu2 = frequencies.nu[p][f2];

  const double U1 = U[R][index(p,f1)];
  const double U2 = U[R][index(p,f2)];

  const double V1 = V[R][index(p,f1)];
  const double V2 = V[R][index(p,f2)];

  U_scaled = interpolate_linear (nu1, U1, nu2, U2, freq_scaled);
  V_scaled = interpolate_linear (nu1, V1, nu2, V2, freq_scaled);


  return (0);

}

#endif




inline int RADIATION ::
    rescale_U_and_V_and_bdy_I         (
        const FREQUENCIES &frequencies,
        const long         p,
        const long         R,
              long        &notch,
        const vReal       &freq_scaled,
              vReal       &U_scaled,
              vReal       &V_scaled,
              vReal       &Ibdy_scaled ) const
   
#if (GRID_SIMD)

{

  vReal nu1, nu2, U1, U2, V1, V2, Ibdy1, Ibdy2;

  const long b = cell2boundary_nr[p];


  for (int lane = 0; lane < n_simd_lanes; lane++)
  {
    double freq = freq_scaled.getlane (lane);

    search_with_notch (frequencies.nu[p], notch, freq);

    const long f1    =  notch    / n_simd_lanes;
    const  int lane1 =  notch    % n_simd_lanes;

    const long f2    = (notch-1) / n_simd_lanes;
    const  int lane2 = (notch-1) % n_simd_lanes;

    //const double nu1 = frequencies.nu[p][f1].getlane(lane1);
    //const double nu2 = frequencies.nu[p][f2].getlane(lane2);
    
    //const double U1 = U[R][index(p,f1)].getlane(lane1);
    //const double U2 = U[R][index(p,f2)].getlane(lane2);
    
    //const double V1 = V[R][index(p,f1)].getlane(lane1);
    //const double V2 = V[R][index(p,f2)].getlane(lane2);
    
    //const double Ibdy1 = boundary_intensity[R][b][f1].getlane(lane1);
    //const double Ibdy2 = boundary_intensity[R][b][f2].getlane(lane2);
    
    //   U_scaled.putlane (interpolate_linear (nu1, U1,    nu2, U2,    freq), lane);
    //   V_scaled.putlane (interpolate_linear (nu1, V1,    nu2, V2,    freq), lane);
    //Ibdy_scaled.putlane (interpolate_linear (nu1, Ibdy1, nu2, Ibdy2, freq), lane);
    
      nu1.putlane (      frequencies.nu[p][f1].getlane (lane1), lane);
      nu2.putlane (      frequencies.nu[p][f2].getlane (lane2), lane);
    
       U1.putlane (           U[R][index(p,f1)].getlane (lane1), lane);
       U2.putlane (           U[R][index(p,f2)].getlane (lane2), lane);
    
       V1.putlane (           V[R][index(p,f1)].getlane (lane1), lane);
       V2.putlane (           V[R][index(p,f2)].getlane (lane2), lane);
    
    Ibdy1.putlane (boundary_intensity[R][b][f1].getlane (lane1), lane);
    Ibdy2.putlane (boundary_intensity[R][b][f2].getlane (lane2), lane);
  }
    
     U_scaled = interpolate_linear (nu1, U1,    nu2,    U2, freq_scaled);
     V_scaled = interpolate_linear (nu1, V1,    nu2,    V2, freq_scaled);
  Ibdy_scaled = interpolate_linear (nu1, Ibdy1, nu2, Ibdy2, freq_scaled);


   return (0);
}

#else

{
  const long b = cell2boundary_nr[p];

  search_with_notch (frequencies.nu[p], notch, freq_scaled);
  
  const long f1    = notch;
  const long f2    = notch-1;

  const double nu1 = frequencies.nu[p][f1];
  const double nu2 = frequencies.nu[p][f2];
  
  const double U1 = U[R][index(p,f1)];
  const double U2 = U[R][index(p,f2)];
  
  const double V1 = V[R][index(p,f1)];
  const double V2 = V[R][index(p,f2)];
  
  const double Ibdy1 = boundary_intensity[R][b][f1];
  const double Ibdy2 = boundary_intensity[R][b][f2];
  
     U_scaled = interpolate_linear (nu1, U1,    nu2, U2,    freq_scaled);
     V_scaled = interpolate_linear (nu1, V1,    nu2, V2,    freq_scaled);
  Ibdy_scaled = interpolate_linear (nu1, Ibdy1, nu2, Ibdy2, freq_scaled);


  return (0);

}

#endif


template <int Dimension, long Nrays>
int RADIATION ::
    compute_mean_intensity                          (
        const CELLS <Dimension, Nrays> &cells,
        const TEMPERATURE              &temperature,
        const FREQUENCIES              &frequencies,
        const LINES                    &lines,
        const SCATTERING               &scattering  )
{

  // For all ray pairs

  for (long r = MPI_start (Nrays/2); r < MPI_stop (Nrays/2); r++)
  {
    const long R  = r - MPI_start (Nrays/2);   // (local) ray index
    const long ar = cells.rays.antipod[r];     // (global) antipodal ray index


    // Loop over all cells

#   pragma omp parallel                                                     \
    shared  (cells, temperature, frequencies, lines, scattering, r, cout)   \
    default (none)
    {

    RAYPAIR raypair (cells.ncells, frequencies.nfreq_red, r, ar, R, U, V, boundary_intensity, cell2boundary_nr);

    for (long o = OMP_start (cells.ncells); o < OMP_stop (cells.ncells); o++)
    {

      raypair.initialize <Dimension, Nrays> (cells, temperature, o);


      if (raypair.ndep > 1)
      {

        for (long f = 0; f < frequencies.nfreq_red; f++)
        {
          // Setup and solve the ray equations

          raypair.setup    (
               frequencies,
               temperature,
               lines,
               scattering,
               f           );

          raypair.solve ();


          // Store solution of the radiation field

          const long ind = index(o,f);

          u[R][ind] = raypair.get_u_at_origin();
          v[R][ind] = raypair.get_v_at_origin();
          

        } // end of loop over frequencies

      }

      else if (raypair.ndep == 1)
      {

        for (long f = 0; f < frequencies.nfreq_red; f++)
        {
          // Setup and solve the ray equations

          raypair.setup    (
               frequencies,
               temperature,
               lines,
               scattering,
               f           );

          raypair.solve_ndep_is_1 ();


          // Store solution of the radiation field

          const long ind = index(o,f);

          u[R][ind] = raypair.get_u_at_origin();
          v[R][ind] = raypair.get_v_at_origin();

        } // end of loop over frequencies

      }

      else
      {
      
        const long b = cells.cell2boundary_nr[o];

        for (long f = 0; f < frequencies.nfreq_red; f++)
        {
          const long ind = index(o,f);

          u[R][ind] = 0.5 * (boundary_intensity[r][b][f] + boundary_intensity[ar][b][f]);
          v[R][ind] = 0.5 * (boundary_intensity[r][b][f] - boundary_intensity[ar][b][f]);
        }

      }


    }
    } // end of pragma omp parallel


  } // end of loop over ray pairs



  // Reduce results of all MPI processes to get J, U and V
  
  calc_J ();
  
  calc_U_and_V (scattering);
  

  return (0);

}


template <int Dimension, long Nrays>
int RADIATION ::
    compute_images                                  (
        const CELLS <Dimension, Nrays> &cells,
        const TEMPERATURE              &temperature,
        const FREQUENCIES              &frequencies,
        const LINES                    &lines,
        const SCATTERING               &scattering  )
{

  // Create an image object

  IMAGE image (cells.ncells, Nrays, frequencies.nfreq_red);

  // For all ray pairs

  for (long r = MPI_start (Nrays/2); r < MPI_stop (Nrays/2); r++)
  {
    const long R  = r - MPI_start (Nrays/2);   // (local) ray index
    const long ar = cells.rays.antipod[r];     // (global) antipodal ray index


    // Loop over all cells

#   pragma omp parallel                                                            \
    shared  (cells, temperature, frequencies, lines, scattering, r, image, cout)   \
    default (none)
    {

    RAYPAIR raypair (cells.ncells, frequencies.nfreq_red, r, ar, R, U, V, boundary_intensity, cell2boundary_nr);


    for (long o = OMP_start (cells.ncells); o < OMP_stop (cells.ncells); o++)
    {

      raypair.initialize <Dimension, Nrays> (cells, temperature, o);


      if (raypair.ndep > 2)
      {

        for (long f = 0; f < frequencies.nfreq_red; f++)
        {
          // Setup and solve the ray equations


          raypair.setup    (
               frequencies,
               temperature,
               lines,
               scattering,
               f           );

          raypair.solve ();


          // Store intensity on the ray ends

          image.I_p[R][o][f] = raypair.get_I_p();
          image.I_m[R][o][f] = raypair.get_I_m();

        } // end of loop over frequencies

      }

      else if (raypair.ndep == 2)
      {

        for (long f = 0; f < frequencies.nfreq_red; f++)
        {
          // Setup and solve the ray equations


          raypair.setup    (
               frequencies,
               temperature,
               lines,
               scattering,
               f           );


          raypair.solve_ndep_is_1 ();


          // Store intensity on the ray ends

          image.I_p[R][o][f] = raypair.get_I_p();
          image.I_m[R][o][f] = raypair.get_I_m();

        } // end of loop over frequencies

      }

      else
      {
      
        const long b = cells.cell2boundary_nr[o];

        for (long f = 0; f < frequencies.nfreq_red; f++)
        {
          image.I_p[R][o][f] = boundary_intensity[ar][b][f];
          image.I_m[R][o][f] = boundary_intensity[r][b][f];
        }

      }


    }
    } // end of pragma omp parallel


  } // end of loop over ray pairs


  // Print images

  image.print("");


  return (0);

}




template <int Dimension, long Nrays>
int RADIATION ::
    compute_mean_intensity_and_images               (
        const CELLS <Dimension, Nrays> &cells,
        const TEMPERATURE              &temperature,
        const FREQUENCIES              &frequencies,
        const LINES                    &lines,
        const SCATTERING               &scattering  )
{

  IMAGE image (cells.ncells, Nrays, frequencies.nfreq_red);


  // For all ray pairs

  for (long r = MPI_start (Nrays/2); r < MPI_stop (Nrays/2); r++)
  {
    const long R  = r - MPI_start (Nrays/2);   // (local) ray index
    const long ar = cells.rays.antipod[r];     // (global) antipodal ray index


    // Loop over all cells

#   pragma omp parallel                                                            \
    shared  (cells, temperature, frequencies, lines, scattering, r, image, cout)   \
    default (none)
    {

    RAYPAIR raypair (cells.ncells, frequencies.nfreq_red, r, ar, R, U, V, boundary_intensity, cell2boundary_nr);

    for (long o = OMP_start (cells.ncells); o < OMP_stop (cells.ncells); o++)
    {

      raypair.initialize <Dimension, Nrays> (cells, temperature, o);

      if (raypair.ndep > 1)
      {

        for (long f = 0; f < frequencies.nfreq_red; f++)
        {

          // Setup and solve the ray equations

          raypair.setup    (
               frequencies,
               temperature,
               lines,
               scattering,
               f           );

          raypair.solve ();
          

          // Store solution of the radiation field

          const long ind = index(o,f);

          u[R][ind] = raypair.get_u_at_origin();
          v[R][ind] = raypair.get_v_at_origin();

          image.I_p[R][o][f] = raypair.get_I_p();
          image.I_m[R][o][f] = raypair.get_I_m();

        } // end of loop over frequencies

      }

      else if (raypair.ndep == 1)
      {

        for (long f = 0; f < frequencies.nfreq_red; f++)
        {
          // Setup and solve the ray equations

          raypair.setup    (
               frequencies,
               temperature,
               lines,
               scattering,
               f           );

          raypair.solve_ndep_is_1 ();


          // Store solution of the radiation field

          const long ind = index(o,f);

          u[R][ind] = raypair.get_u_at_origin();
          v[R][ind] = raypair.get_v_at_origin();

          image.I_p[R][o][f] = raypair.get_I_p();
          image.I_m[R][o][f] = raypair.get_I_m();

        } // end of loop over frequencies

      }

      else
      {
      
        const long b = cells.cell2boundary_nr[o];

        for (long f = 0; f < frequencies.nfreq_red; f++)
        {
          const long ind = index(o,f);

          u[R][ind] = 0.5 * (boundary_intensity[r][b][f] + boundary_intensity[ar][b][f]);
          v[R][ind] = 0.5 * (boundary_intensity[r][b][f] - boundary_intensity[ar][b][f]);

          image.I_p[R][o][f] = boundary_intensity[ar][b][f];
          image.I_m[R][o][f] = boundary_intensity[r][b][f];
        }

      }


    }
    } // end of pragma omp parallel


  } // end of loop over ray pairs



  // Reduce results of all MPI processes to get J, U and V
  
  calc_J ();
  
  calc_U_and_V (scattering);
  
  
  // Print images

  image.print("");


  return (0);

}


