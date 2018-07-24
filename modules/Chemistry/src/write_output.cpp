// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <stdio.h>
#include <stdlib.h>

#include <iostream>
#include <string>
#include <sstream>

#include "declarations.hpp"
#include "write_output.hpp"
#include "write_txt_tools.hpp"
#include "write_vtu_tools.hpp"




// write_output: write output
// --------------------------

int write_output (CELLS *cells, LINES lines)
{

  // Get tag to distinguish outputs

  std::ostringstream conv_tag_nr;
  conv_tag_nr << tag_nr;
  std::string tag = conv_tag_nr.str();

  tag_nr++;


  if      (INPUT_TYPE == vtu)
  {
    write_vtu_output (tag, cells);
  }
  else if (INPUT_TYPE == txt)
  {
    write_txt_output (tag, cells, lines);
  }


  return (0);

}




// write_output_log: write output info
// -----------------------------------

int write_output_log ()
{

  std::string file_name = output_directory + "output.log";

  FILE *file = fopen(file_name.c_str(), "w");


  if (file == NULL)
  {
    printf ("Error opening file!\n");
    std::cout << file_name + "\n";
    exit (1);
  }

  // fprintf (file, "outputDirectory %s\n", output_directory.c_str());

  fprintf (file, "tag_nr %d\n", tag_nr);

  fclose (file);


  return (0);

}




// write_performance_log: write performance results
// ------------------------------------------------

int write_performance_log (TIMERS timers)
{


  // std::string run_number = RUN_NUMBER;
  // std::string file_name = "tests/performance/performance_" + run_number + ".log";

  std::string file_name = output_directory + "performance.log";

  FILE *file = fopen(file_name.c_str(), "w");


  if (file == NULL)
  {
    printf ("Error opening file!\n");
    std::cout << file_name + "\n";
    exit (1);
  }


  fprintf (file, "time_total       %lE\n", timers.total.duration);
  fprintf (file, "time_level_pop   %lE\n", timers.level_pop.duration);
  fprintf (file, "time_chemistry   %lE\n", timers.chemistry.duration);


  fclose (file);


  return (0);

}
