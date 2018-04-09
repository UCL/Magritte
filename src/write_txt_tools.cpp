// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <iostream>
#include <string>
#include <sstream>

#include "declarations.hpp"
#include "write_txt_tools.hpp"
#include "radfield_tools.hpp"
#include "initializers.hpp"
#include "calc_LTE_populations.hpp"


// write_txt_output: write output in txt format
// --------------------------------------------

int write_txt_output (std::string tag, CELLS *cells, LINES lines)
{

  // write_abundances (tag, cells);

  // write_transition_levels (tag, lines);

  write_level_populations (tag, cells, lines);

  // write_line_intensities (tag, cells, lines);

  write_temperature_gas (tag, cells);

  write_temperature_gas_prev(tag, cells);

  write_thermal_ratio (tag, cells);

  write_thermal_ratio_prev(tag, cells);

  // write_temperature_dust (tag, cells);

  // write_double_vector ("level_populations", tag, NCELLS*TOT_NLEV, cells->pop);


  return (0);

}




// write_grid: write input back
// ----------------------------

int write_grid (std::string tag, CELLS *cells)
{

  if (!tag.empty())
  {
    tag = "_" + tag;
  }

  long total = 0;

  std::string file_name = output_directory + "grid" + tag + ".txt";

  FILE *file = fopen (file_name.c_str(), "w");

  if (file == NULL)
  {
    printf ("Error opening file!\n");
    std::cout << file_name + "\n";
    exit (1);
  }

  for (long p = 0; p < NCELLS; p++)
  {
    if (cells->id[p] == p)   // cell[p].id == p means cell was not removed
    {
      fprintf (file, "%ld\t%lE\t%lE\t%lE\t%lE\t%lE\t%lE\t%lE\n",
               cells->id[p],
               cells->x[p],  cells->y[p],  cells->z[p],
               cells->vx[p], cells->vy[p], cells->vz[p],
               cells->density[p]);

      total++;
    }
  }

  fclose (file);


  printf ("Printed a grid with %ld cells out of a total of %ld\n", total, cells->ncells);

  return (0);

}




// write_neighbors: write neighbors of each cell
// ---------------------------------------------

int write_neighbors (std::string tag, CELLS *cells)
{

  if (!tag.empty())
  {
    tag = "_" + tag;
  }


  std::string file_name = output_directory + "neighbors" + tag + ".txt";

  FILE *file = fopen (file_name.c_str(), "w");

  if (file == NULL)
  {
    printf ("Error opening file!\n");
    std::cout << file_name + "\n";
    exit (1);
  }

  for (long p = 0; p < NCELLS; p++)
  {
    fprintf (file, "%ld\t", cells->n_neighbors[p]);

    for (long n = 0; n < NRAYS; n++)
    {
      fprintf (file, "%ld\t", cells->neighbor[RINDEX(p,n)]);
    }

    fprintf (file, "\n");
  }

  fclose (file);


  return (0);

}




// write_rays: write HEALPix vectors
// -------------------------------------------

int write_rays (std::string tag, RAYS rays)
{

  if (!tag.empty())
  {
    tag = "_" + tag;
  }

  std::string file_name = output_directory + "healpix" + tag + ".txt";

  FILE *file = fopen(file_name.c_str(), "w");

  if (file == NULL)
  {
    printf ("Error opening file!\n");
    std::cout << file_name + "\n";
    exit (1);
  }


  for (long r = 0; r < NRAYS; r++)
  {
    fprintf (file, "%.15f\t%.15f\t%.15f\n", rays.x[r],
                                            rays.y[r],
                                            rays.z[r]);
  }

  fclose (file);


  return (0);

}




// write_abundances: write abundances at each point
// ------------------------------------------------

int write_abundances (std::string tag, CELLS *cells)
{

  if (!tag.empty())
  {
    tag = "_" + tag;
  }

  std::string file_name = output_directory + "abundances" + tag + ".txt";

  FILE *file = fopen (file_name.c_str(), "w");

  if (file == NULL)
  {
    printf ("Error opening file!\n");
    std::cout << file_name + "\n";
    exit (1);
  }


  for (long p = 0; p < NCELLS; p++)
  {
    for (int s = 0; s < NSPEC; s++)
    {
      fprintf (file, "%lE\t", cells->abundance[SINDEX(p,s)]);
    }

    fprintf (file, "\n");
  }

  fclose (file);


  return (0);

}




// write_level_populations: write level populations at each point for each transition
// ----------------------------------------------------------------------------------

int write_level_populations (std::string tag, CELLS *cells, LINES lines)
{

  if (!tag.empty())
  {
    tag = "_" + tag;
  }


  for (int ls = 0; ls < NLSPEC; ls++)
  {
    std::string lspec_name = lines.sym[ls];
    std::string file_name  = output_directory + "level_populations_" + lspec_name + tag + ".txt";

    FILE *file = fopen (file_name.c_str(), "w");


    if (file == NULL)
    {
      std::cout << "Error opening file " << file_name << "!\n";
      std::cout << file_name + "\n";
      exit (1);
    }


    for (long p = 0; p < NCELLS; p++)
    {
      for (int i = 0; i < nlev[ls]; i++)
      {
        fprintf (file, "%lE\t", cells->pop[LINDEX(p,LSPECLEV(ls, i))]); // /cells->density[p]/cells->abundance[SINDEX(p,1)]);
      }

      fprintf (file, "\n");
    }


    fclose (file);

  } // end of lspec loop over line producing species


  return (0);

}


//
//
// // write_line_intensities: write line intensities for each species, point and transition
// // -------------------------------------------------------------------------------------
//
// int write_line_intensities (std::string tag, long ncells, CELL *cell, LINES lines)
// {
//
//   if(!tag.empty())
//   {
//     tag = "_" + tag;
//   }
//
//
//   for (int lspec = 0; lspec < NLSPEC; lspec++)
//   {
//     std::string lspec_name = lines.sym[lspec];
//
//     std::string file_name = output_directory + "line_intensities_" + lspec_name + tag + ".txt";
//
//     FILE *file = fopen (file_name.c_str(), "w");
//
//
//     if (file == NULL)
//     {
//       printf ("Error opening file!\n");
//       std::cout << file_name + "\n";
//       exit (1);
//     }
//
//
//     for (long n = 0; n < NCELLS; n++)
//     {
//       for (int kr = 0; kr < nrad[lspec]; kr++)
//       {
//         fprintf (file, "%lE\t", cell[n].mean_intensity[LSPECRAD(lspec,kr)]);
//       }
//
//       fprintf (file, "\n");
//     }
//
//     fclose (file);
//
//   }
//
//
//   return (0);
//
// }
//
//


// write_thermal_ratio: write thermal ratio at each cell
// -----------------------------------------------------

int write_thermal_ratio (std::string tag, CELLS *cells)
{

  if (!tag.empty())
  {
    tag = "_" + tag;
  }

  std::string file_name = output_directory + "thermal_ratio" + tag + ".txt";

  FILE *file = fopen(file_name.c_str(), "w");

  if (file == NULL)
  {
    printf ("Error opening file!\n");
    std::cout << file_name + "\n";
    exit (1);
  }


  for (long p = 0; p < NCELLS; p++)
  {
    fprintf (file, "%lE\n", cells->thermal_ratio[p]);
  }

  fclose (file);


  return (0);

}



// write_thermal_ratio: write thermal ratio at each cell
// -----------------------------------------------------

int write_thermal_ratio_prev (std::string tag, CELLS *cells)
{

  if (!tag.empty())
  {
    tag = "_" + tag;
  }

  std::string file_name = output_directory + "thermal_ratio_prev" + tag + ".txt";

  FILE *file = fopen(file_name.c_str(), "w");

  if (file == NULL)
  {
    printf ("Error opening file!\n");
    std::cout << file_name + "\n";
    exit (1);
  }


  for (long p = 0; p < NCELLS; p++)
  {
    fprintf (file, "%lE\n", cells->thermal_ratio_prev[p]);
  }

  fclose (file);


  return (0);

}




// write_temperature_gas: write gas temperatures at each cell
// ----------------------------------------------------------

int write_temperature_gas (std::string tag, CELLS *cells)
{

  if (!tag.empty())
  {
    tag = "_" + tag;
  }

  std::string file_name = output_directory + "temperature_gas" + tag + ".txt";

  FILE *file = fopen(file_name.c_str(), "w");

  if (file == NULL)
  {
    printf ("Error opening file!\n");
    std::cout << file_name + "\n";
    exit (1);
  }


  for (long p = 0; p < NCELLS; p++)
  {
    fprintf (file, "%lE\n", cells->temperature_gas[p]);
  }

  fclose (file);


  return (0);

}



// write_temperature_dust: write dust temperatures at each cell
// ------------------------------------------------------------

int write_temperature_dust (std::string tag, CELLS *cells)
{

  if (!tag.empty())
  {
    tag = "_" + tag;
  }

  std::string file_name = output_directory + "temperature_dust" + tag + ".txt";

  FILE *file = fopen (file_name.c_str(), "w");

  if (file == NULL)
  {
    printf ("Error opening file!\n");
    std::cout << file_name + "\n";
    exit (1);
  }


  for (long p = 0; p < NCELLS; p++)
  {
    fprintf (file, "%lE\n", cells->temperature_dust[p]);
  }

  fclose (file);


  return (0);

}





// write_temperature_gas_prev: write previous gas temperatures at each cell
// ------------------------------------------------------------------------

int write_temperature_gas_prev(std::string tag, CELLS *cells)
{

  if (!tag.empty())
  {
    tag = "_" + tag;
  }

  std::string file_name = output_directory + "temperature_gas_prev" + tag + ".txt";

  FILE *file = fopen (file_name.c_str(), "w");

  if (file == NULL)
  {
    printf ("Error opening file!\n");
    std::cout << file_name + "\n";
    exit (1);
  }


  for (long p = 0; p < NCELLS; p++)
  {
    fprintf (file, "%lE\n", cells->temperature_gas_prev[p]);
  }

  fclose (file);


  return (0);

}

//
//
//
// // write_UV_field: write UV field at each cell
// // -------------------------------------------
//
// int write_UV_field (std::string tag, long ncells, double *UV_field)
// {
//
//
//   if (!tag.empty())
//   {
//     tag = "_" + tag;
//   }
//
//   std::string file_name = output_directory + "UV_field" + tag + ".txt";
//
//   FILE *file = fopen (file_name.c_str(), "w");
//
//   if (file == NULL)
//   {
//     printf ("Error opening file!\n");
//     std::cout << file_name + "\n";
//     exit (1);
//   }
//
//
//   for (long n = 0; n < NCELLS; n++)
//   {
//     fprintf (file, "%lE\n", UV_field[n]);
//   }
//
//   fclose (file);
//
//
//   return (0);
//
// }
//
//
//
//
// // write_UV_field: write visual extinction (AV) at each point
// // ----------------------------------------------------------
//
// int write_AV (std::string tag, long ncells, double *AV)
// {
//
//   if (!tag.empty())
//   {
//     tag = "_" + tag;
//   }
//
//   std::string file_name = output_directory + "AV" + tag + ".txt";
//
//   FILE *file = fopen (file_name.c_str(), "w");
//
//   if (file == NULL)
//   {
//     printf ("Error opening file!\n");
//     std::cout << file_name + "\n";
//     exit (1);
//   }
//
//
//   for (long n = 0; n < NCELLS; n++)
//   {
//     for (long r = 0; r < NRAYS; r++)
//     {
//       fprintf (file, "%lE\t", AV[RINDEX(n,r)]);
//     }
//
//     fprintf (file, "\n");
//   }
//
//   fclose (file);
//
//
//   return (0);
//
// }
//
//
//
//
// // write_rad_surface: write rad surface at each point
// // --------------------------------------------------
//
// int write_rad_surface (std::string tag, long ncells, double *rad_surface)
// {
//
//   if (!tag.empty())
//   {
//     tag = "_" + tag;
//   }
//
//   std::string file_name = output_directory + "rad_surface" + tag + ".txt";
//
//   FILE *file = fopen (file_name.c_str(), "w");
//
//   if (file == NULL)
//   {
//     printf ("Error opening file!\n");
//     std::cout << file_name + "\n";
//     exit (1);
//   }
//
//
//   for (long n = 0; n < NCELLS; n++)
//   {
//     for (long r = 0; r < NRAYS; r++)
//     {
//       fprintf (file, "%lE\t", rad_surface[RINDEX(n,r)]);
//     }
//
//     fprintf (file, "\n");
//   }
//
//   fclose (file);
//
//
//   return (0);
//
// }
//
//
//
//
// // write_reaction_rates: write rad surface at each cell
// // ----------------------------------------------------
//
// int write_reaction_rates (std::string tag, long ncells, CELL *cell)
// {
//
//   if (!tag.empty())
//   {
//     tag = "_" + tag;
//   }
//
//   std::string file_name = output_directory + "reaction_rates" + tag + ".txt";
//
//   FILE *file = fopen (file_name.c_str(), "w");
//
//   if (file == NULL)
//   {
//     printf ("Error opening file!\n");
//     std::cout << file_name + "\n";
//     exit (1);
//   }
//
//
//   for (long n = 0; n < NCELLS; n++)
//   {
//     for (int reac = 0; reac < NREAC; reac++)
//     {
//       fprintf (file, "%lE\t", cell[n].rate[reac]);
//     }
//
//     fprintf (file, "\n");
//   }
//
//   fclose (file);
//
//
//   return (0);
//
// }
//
//
//
//
// // write_certain_reactions: write rates of certain reactions
// // ---------------------------------------------------------
//
// int write_certain_rates (std::string tag, long ncells, CELL *cell, std::string name,
//                          int nr_certain_reac, int *certain_reactions)
// {
//
//   if (!tag.empty())
//   {
//     tag = "_" + tag;
//   }
//
//
//   std::string file_name0 = output_directory + name + "_reactions" + tag + ".txt";
//
//   FILE *file0 = fopen (file_name0.c_str(), "w");
//
//   if (file0 == NULL)
//   {
//     printf ("Error opening file!\n");
//     std::cout << file_name0 + "\n";
//     exit (1);
//   }
//
//
//   for (int reac = 0; reac < nr_certain_reac; reac++)
//   {
//     fprintf (file0, "%d\n", certain_reactions[reac]);
//   }
//
//   fclose (file0);
//
//
//   std::string file_name = output_directory + name + "_rates" + tag + ".txt";
//
//   FILE *file = fopen (file_name.c_str(), "w");
//
//   if (file == NULL)
//   {
//     printf ("Error opening file!\n");
//     std::cout << file_name + "\n";
//     exit (1);
//   }
//
//
//   for (int reac = 0; reac < nr_certain_reac; reac++)
//   {
//     fprintf (file, "%d \t", certain_reactions[reac]);
//   }
//
//
//   fprintf (file, "\n");
//
//   for (long n = 0; n < NCELLS; n++)
//   {
//     for (int reac = 0; reac < nr_certain_reac; reac++)
//     {
//       fprintf (file, "%lE \t", cell[n].rate[certain_reactions[reac]]);
//     }
//
//     fprintf (file, "\n");
//   }
//
//   fclose (file);
//
//
//   return (0);
//
// }
//



// write_double_vector: write a vector of doubles
// ----------------------------------------------

int write_double_vector (std::string name, std::string tag, long length, const double *variable)
{

  if (!tag.empty())
  {
    tag = "_" + tag;
  }

  std::string file_name = output_directory + name + tag + ".txt";

  FILE *file = fopen(file_name.c_str(), "w");

  if (file == NULL)
  {
    printf ("Error opening file!\n");
    std::cout << file_name + "\n";
    exit (1);
  }


  for (long n = 0; n < length; n++)
  {
    fprintf (file, "%lE\n", variable[n]);
  }

  fclose (file);


  return (0);

}




// write_double_matrix: write a matrix of doubles
// ----------------------------------------------

int write_double_matrix (std::string name, std::string tag, long nrows, long ncols, const double *variable)
{

  if (!tag.empty())
  {
    tag = "_" + tag;
  }

  std::string file_name = output_directory + name + tag + ".txt";

  FILE *file = fopen (file_name.c_str(), "w");

  if (file == NULL)
  {
    printf ("Error opening file!\n");
    std::cout << file_name + "\n";
    exit (1);
  }


  for (long row = 0; row < nrows; row++)
  {
    for (long col = 0; col < ncols; col++)
    {
      fprintf (file, "%lE\t", variable[col + ncols*row]);
    }

    fprintf (file, "\n");
  }

  fclose (file);


  return (0);

}
//
//
//
//
// /* write_radfield_tools: write the output of the functoins defined in radfield_tools             */
// /*-----------------------------------------------------------------------------------------------*/
// //
// // int write_radfield_tools( std::string tag, double *AV ,double lambda,
// //                           double *column_H2, double *column_CO )
// // {
// //
// //
// //   if ( !tag.empty() ){
// //
// //     tag = "_" + tag;
// //   }
// //
// //
// //
// //   /* Write dust scattering */
// //
// //   std::string file_name = "output/files/dust_scattering" + tag + ".txt";
// //
// //   FILE *ds_file = fopen(file_name.c_str(), "w");
// //
// //   if (ds_file == NULL){
// //
// //     printf("Error opening file!\n");
// //     exit(1);
// //   }
// //
// //
// //   for (long n=0; n<NCELLS; n++){
// //
// //     for (long r=0; r<NRAYS; r++){
// //
// //       double w = log10(1.0 + column_H2[RINDEX(n,r)]);
// //       double LLLlambda = (5675.0 - 200.6*w);
// //
// //       fprintf( ds_file, "%lE\t", dust_scattering(AV[RINDEX(n,r)], LLLlambda) );
// //     }
// //
// //     fprintf( ds_file, "\n" );
// //   }
// //
// //
// //   fclose(ds_file);
// //
// //
// //
// //   /* Write H2 shield */
// //
// //   std::string file_name2 = "output/files/shielding_H2" + tag + ".txt";
// //
// //   FILE *s_file = fopen(file_name2.c_str(), "w");
// //
// //   if (s_file == NULL){
// //
// //     printf("Error opening file!\n");
// //     exit(1);
// //   }
// //
// //
// //
// //   double doppler_width = V_TURB / (lambda*1.0E-8);    /* linewidth (in Hz) of typical transition */
// //                                                 /* (assuming turbulent broadening with b=3 km/s) */
// //
// //
// //   double radiation_width = 8.0E7;         /* radiative linewidth (in Hz) of a typical transition */
// //
// //
// //   for (long n=0; n<NCELLS; n++){
// //
// //     for (long r=0; r<NRAYS; r++){
// //
// //       fprintf( s_file, "%lE\t", self_shielding_H2( column_H2[RINDEX(n,r)], doppler_width, radiation_width ) );
// //     }
// //
// //     fprintf( s_file, "\n" );
// //   }
// //
// //
// //   fclose(s_file);
// //
// //
// //   /* Write CO shield */
// //
// //   std::string file_name3 = "output/files/shielding_CO" + tag + ".txt";
// //
// //   FILE *c_file = fopen(file_name3.c_str(), "w");
// //
// //   if (c_file == NULL){
// //
// //     printf("Error opening file!\n");
// //     exit(1);
// //   }
// //
// //
// //
// //   for (long n=0; n<NCELLS; n++){
// //
// //     for (long r=0; r<NRAYS; r++){
// //
// //       fprintf( c_file, "%lE\t", self_shielding_CO( column_CO[RINDEX(n,r)], column_H2[RINDEX(n,r)] ) );
// //     }
// //
// //     fprintf( c_file, "\n" );
// //   }
// //
// //
// //   fclose(c_file);
// //
// //
// //
// //   /* Write X_lambda */
// //
// //   std::string file_name4 = "output/files/X_lambda" + tag + ".txt";
// //
// //   FILE *xl_file = fopen(file_name4.c_str(), "w");
// //
// //   if (xl_file == NULL){
// //
// //     printf("Error opening file!\n");
// //     exit(1);
// //   }
// //
// //
// //
// //   for (long n=1; n<=200; n++){
// //
// //     double LLLlambda = pow(10.0, (9.0-2.0)/200.0*n+2.0);
// //     fprintf( xl_file, "%lE\t%lE\n", LLLlambda, X_lambda(LLLlambda) );
// //
// //   }
// //
// //
// //   fclose(xl_file);
// //
// //
// //   // cout << "X lambda " << X_lambda(1000.0) << "\n";
// //
// //   return(0);
// //
// // }
//
//
//
//
// // write_Einstein_coeff: write Einstein A, B or C coefficients
// // -----------------------------------------------------------
//
// int write_Einstein_coeff (std::string tag, LINES lines,
//                           double *A_coeff, double *B_coeff, double *C_coeff)
// {
//
//   if (!tag.empty())
//   {
//     tag = "_" + tag;
//   }
//
//
//   for (int lspec = 0; lspec < NLSPEC; lspec++)
//   {
//     std::string lspec_name = lines.sym[lspec];
//
//
//     std::string file_name_A = output_directory + "Einstein_A_" + lspec_name + tag + ".txt";
//     std::string file_name_B = output_directory + "Einstein_B_" + lspec_name + tag + ".txt";
//     std::string file_name_C = output_directory + "Einstein_C_" + lspec_name + tag + ".txt";
//
//
//     FILE *file_A = fopen (file_name_A.c_str(), "w");
//     FILE *file_B = fopen (file_name_B.c_str(), "w");
//     FILE *file_C = fopen (file_name_C.c_str(), "w");
//
//     if (file_A == NULL)
//     {
//       printf ("Error opening file!\n");
//       std::cout << file_name_A + "\n";
//       exit (1);
//     }
//
//     if (file_B == NULL)
//     {
//       printf ("Error opening file!\n");
//       std::cout << file_name_B + "\n";
//       exit (1);
//     }
//
//     if (file_C == NULL)
//     {
//       printf ("Error opening file!\n");
//       std::cout << file_name_C + "\n";
//       exit (1);
//     }
//
//
//     for (long row = 0; row < nlev[lspec]; row++)
//     {
//       for (long col = 0; col < nlev[lspec]; col++)
//       {
//         fprintf (file_A, "%lE\t", lines.A_coeff[LSPECLEVLEV(lspec,row,col)]);
//         fprintf (file_B, "%lE\t", lines.B_coeff[LSPECLEVLEV(lspec,row,col)]);
//         fprintf (file_C, "%lE\t", C_coeff[LSPECLEVLEV(lspec,row,col)]);
//       }
//
//       fprintf (file_A, "\n");
//       fprintf (file_B, "\n");
//       fprintf (file_C, "\n");
//     }
//
//
//     fclose (file_A);
//     fclose (file_B);
//     fclose (file_C);
//
//   } // end of lspec loop over line producing species
//
//
//   return (0);
//
// }
//
//
//
//
// // write_R: write the transition matrix R
// // --------------------------------------
//
// int write_R (std::string tag, long ncells, LINES lines, long o, double *R)
// {
//
//   if (!tag.empty())
//   {
//     tag = "_" + tag;
//   }
//
//
//   for (int lspec = 0; lspec < NLSPEC; lspec++)
//   {
//     std::string lspec_name = lines.sym[lspec];
//
//     std::string file_name = output_directory + "R_" + lspec_name + tag + ".txt";
//
//     FILE *file = fopen (file_name.c_str(), "w");
//
//
//     if (file == NULL)
//     {
//       printf ("Error opening file!\n");
//       std::cout << file_name + "\n";
//       exit (1);
//     }
//
//
//     for (long row = 0; row < nlev[lspec]; row++)
//     {
//       for (long col = 0; col < nlev[lspec]; col++)
//       {
//         fprintf (file, "%lE\t", R[LSPECGRIDLEVLEV(lspec,o,row,col)]);
//
//       }
//
//       fprintf (file, "\n");
//
//     }
//
//     fclose (file);
//
//   } // end of lspec loop over line producing species
//
//
//   return (0);
//
// }
//
//
//
//
// // write_transition_levels: write levels corresponding to each transition
// // ----------------------------------------------------------------------
//
// int write_transition_levels (std::string tag, LINES lines)
// {
//
//   if (!tag.empty())
//   {
//     tag = "_" + tag;
//   }
//
//
//   for (int lspec = 0; lspec < NLSPEC; lspec++)
//   {
//     std::string lspec_name = lines.sym[lspec];
//
//     std::string file_name = output_directory + "transition_levels_" + lspec_name + tag + ".txt";
//
//     FILE *file = fopen (file_name.c_str(), "w");
//
//
//     if (file == NULL)
//     {
//       printf ("Error opening file!\n");
//       std::cout << file_name + "\n";
//       exit (1);
//     }
//
//
//     for (int kr = 0; kr < nrad[lspec]; kr++)
//     {
//       int i = lines.irad[LSPECRAD(lspec,kr)];   // i level index corresponding to transition kr
//       int j = lines.jrad[LSPECRAD(lspec,kr)];   // j level index corresponding to transition kr
//
//       fprintf (file, "%d\t%d\n", i, j);
//
//     }
//
//     fclose (file);
//
//   } // end of lspec loop over line producing species
//
//
//   return (0);
//
// }
//
//
//
//
// // /* write_LTE_deviation: write the relative deviation of the level populations from the LTE value */
// // /*-----------------------------------------------------------------------------------------------*/
// //
// // int write_LTE_deviation( std::string tag, CELL *cell, double *energy, double* weight,
// //                          double *temperature_gas, double *pop )
// // {
// //
// //
// //   if ( !tag.empty() ){
// //
// //     tag = "_" + tag;
// //   }
// //
// //
// //   double LTE_pop[NCELLS*TOT_NLEV];                                        /* level population n_i */
// //
// //   initialize_double_array(NCELLS*TOT_NLEV, LTE_pop);
// //
// //   calc_LTE_populations(cell, energy, weight, temperature_gas, LTE_pop);
// //
// //
// //   for (int lspec=0; lspec<NLSPEC; lspec++){
// //
// //     std::string lspec_name = species[ lspec_nr[lspec] ].sym;
// //
// //     std::string file_name = output_directory + "LTE_deviations_" + lspec_name + tag + ".txt";
// //
// //     FILE *file = fopen(file_name.c_str(), "w");
// //
// //
// //     if (file == NULL){
// //
// //       std :: cout << "Error opening file " << file_name << "!\n";
// //       std::cout << file_name + "\n";
// //       exit(1);
// //     }
// //
// //
// //     for (long n=0; n<NCELLS; n++){
// //
// //       for (int i=0; i<nlev[lspec]; i++){
// //
// //         double dev = 2.0 * (pop[LSPECGRIDLEV(lspec,n, i)] - LTE_pop[LSPECGRIDLEV(lspec,n, i)])
// //                          / (pop[LSPECGRIDLEV(lspec,n, i)] + LTE_pop[LSPECGRIDLEV(lspec,n, i)]);
// //
// //         fprintf(file, "%lE\t", dev);
// //       }
// //
// //       fprintf(file, "\n");
// //     }
// //
// //
// //     fclose(file);
// //
// //   } /* end of lspec loop over line producing species */
// //
// //
// //   return(0);
// //
// // }
// //
// // /*-----------------------------------------------------------------------------------------------*/
// //
// //
// //
// //
// //
// // /* write_true_level_populations: write the true level populations                                */
// // /*-----------------------------------------------------------------------------------------------*/
// //
// // int write_true_level_populations( std::string tag, CELL *cell, double *pop )
// // {
// //
// //
// //   if ( !tag.empty() ){
// //
// //     tag = "_" + tag;
// //   }
// //
// //
// //   for (int lspec=0; lspec<NLSPEC; lspec++){
// //
// //     std::string lspec_name = species[ lspec_nr[lspec] ].sym;
// //
// //     std::string file_name = output_directory + "true_level_populations_" + lspec_name + tag + ".txt";
// //
// //     FILE *file = fopen(file_name.c_str(), "w");
// //
// //
// //     if (file == NULL){
// //
// //       std :: cout << "Error opening file " << file_name << "!\n";
// //       std::cout << file_name + "\n";
// //       exit(1);
// //     }
// //
// //
// //     for (long n=0; n<NCELLS; n++){
// //
// //       for (int i=0; i<nlev[lspec]; i++){
// //
// //         double rel = pop[LSPECGRIDLEV(lspec,n, i)]
// //                      / cell[n].density / cell[n].abundance[lspec_nr[lspec]];
// //
// //         fprintf(file, "%lE\t", rel);
// //       }
// //
// //       fprintf(file, "\n");
// //     }
// //
// //
// //     fclose(file);
// //
// //   } /* end of lspec loop over line producing species */
// //
// //
// //   return(0);
// //
// // }
