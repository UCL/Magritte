// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <omp.h>

#include <string>
#include <iostream>

#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include "../../../../src/declarations.hpp"
#include "../../../../src/definitions.hpp"
#include "../../../../src/calc_C_coeff.hpp"
#include "../../../../src/initializers.hpp"
#include "../../../../src/write_txt_tools.hpp"

#define EPS 1.0E-4


TEST_CASE ("Einstein Collisional coefficient at different temperatures")
{

  long ncells = 1;

  CELLS Cells (ncells);    // create CELLS object Cells
  CELLS *cells = &Cells;   // pointer to Cells

  cells->initialize();


  // Set temperature

  cells->temperature_gas[0] = 0.3714417616594907E+2;
  cells->density[0]         = 1.0;


  const SPECIES species (spec_datafile);

  initialize_abundances (cells, species);

  cells->abundance[SINDEX(0,species.nr_H2)] = 0.1876605698107578;
  cells->abundance[SINDEX(0,species.nr_CO)] = 1.0;


  const LINES lines;

  long o  = 0;   // cell number of cell under consideration
  int  ls = 0;   // number of line producing species

  double C_coeff[TOT_NLEV2];



  calc_C_coeff (cells, species, lines, C_coeff, o, ls);

  //
  // std::string file_name1 = output_directory + "Einstein_A_CO.txt";
  // std::string file_name2 = output_directory + "Einstein_B_CO.txt";
  // std::string file_name3 = output_directory + "Einstein_C_CO.txt";
  //
  // FILE *file1 = fopen (file_name1.c_str(), "w");
  // FILE *file2 = fopen (file_name2.c_str(), "w");
  // FILE *file3 = fopen (file_name3.c_str(), "w");
  //
  //
  // if (file1 == NULL)
  // {
  //   printf ("Error opening file!\n");
  //   std::cout << file_name1 + "\n";
  //   exit (1);
  // }
  //
  // if (file2 == NULL)
  // {
  //   printf ("Error opening file!\n");
  //   std::cout << file_name2 + "\n";
  //   exit (1);
  // }
  //
  // if (file3 == NULL)
  // {
  //   printf ("Error opening file!\n");
  //   std::cout << file_name3 + "\n";
  //   exit (1);
  // }
  //
  //
  // for (long i = 0; i < nlev[ls]; i++)
  // {
  //   for (long j = 0; j < nlev[ls]; j++)
  //   {
  //     fprintf (file1, "%lE\t", lines.A_coeff[LSPECLEVLEV(ls,i,j)]);
  //     fprintf (file2, "%lE\t", lines.B_coeff[LSPECLEVLEV(ls,i,j)]);
  //     fprintf (file3, "%lE\t", C_coeff[LSPECLEVLEV(ls,i,j)]);
  //   }
  //   fprintf (file1, "\n");
  //   fprintf (file2, "\n");
  //   fprintf (file3, "\n");
  // }
  //
  // fclose (file1);
  // fclose (file2);
  // fclose (file3);


  // Get reference data

  double A_ref[TOT_NLEV2];
  double B_ref[TOT_NLEV2];
  double C_ref[TOT_NLEV2];


  std::string file_name1 = "output/compare/Einstein_A_CO_3D-PDR.txt";
  std::string file_name2 = "output/compare/Einstein_B_CO_3D-PDR.txt";
  std::string file_name3 = "output/compare/Einstein_C_CO_3D-PDR.txt";

  FILE *file1 = fopen (file_name1.c_str(), "r");
  FILE *file2 = fopen (file_name2.c_str(), "r");
  FILE *file3 = fopen (file_name3.c_str(), "r");


  if (file1 == NULL)
  {
    printf ("Error opening file!\n");
    std::cout << file_name1 + "\n";
    exit (1);
  }

  if (file2 == NULL)
  {
    printf ("Error opening file!\n");
    std::cout << file_name2 + "\n";
    exit (1);
  }

  if (file3 == NULL)
  {
    printf ("Error opening file!\n");
    std::cout << file_name3 + "\n";
    exit (1);
  }


  for (long i = 0; i < nlev[ls]; i++)
  {
    for (long j = 0; j < nlev[ls]; j++)
    {
      fscanf (file1, "%lE", &(A_ref[LSPECLEVLEV(ls,i,j)]));
      fscanf (file2, "%lE", &(B_ref[LSPECLEVLEV(ls,i,j)]));
      fscanf (file3, "%lE", &(C_ref[LSPECLEVLEV(ls,i,j)]));
    }



    fscanf (file1, "%*[^\n]\n");
    fscanf (file2, "%*[^\n]\n");
    fscanf (file3, "%*[^\n]\n");
  }

  fclose (file1);
  fclose (file2);
  fclose (file3);


  // Check against reference

  for (long i = 0; i < nlev[ls]; i++)
  {
    for (long j = 0; j < nlev[ls]; j++)
    {
      CHECK( lines.A_coeff[LSPECLEVLEV(ls,i,j)]
              == Approx(A_ref[LSPECLEVLEV(ls,i,j)]).epsilon(EPS) );
      CHECK( lines.B_coeff[LSPECLEVLEV(ls,i,j)]
              == Approx(B_ref[LSPECLEVLEV(ls,i,j)]).epsilon(EPS) );
      CHECK( C_coeff[LSPECLEVLEV(ls,i,j)]
              == Approx(C_ref[LSPECLEVLEV(ls,i,j)]).epsilon(EPS) );
    }
  }


}
