// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <fstream>

#include "input.hpp"


///  Constructor for Model
///    @param[in] input_file: folder containing input data
////////////////////////////////////////////////////////////

Input ::
    Input (
        const string input_file)
  : input_file (input_file)
{


}   // END OF CONSTRUCTOR





//
/////  get_ncells: reads in the number of cells
///////////////////////////////////////////////
//
//long get_ncells (void)
//{
//  return get_long_from_txt ("nrays.txt");
//}
//
//
//
//
/////  get_nrays: reads in the number of cells
/////  @param[in] inpt_folder: path to folder containing all input data
///////////////////////////////////////////////////////////////////////
//
//long get_nrays (void)
//{
//  return get_long_from_txt ("ncells.txt");
//}
//



///  get_long_from_txt:
///  @param[in] inpt_folder: path to folder containing all input data
/////////////////////////////////////////////////////////////////////

long Input ::
    get_number (
        const string file_name)
{

  long number;

  ifstream file (input_file + file_name + ".txt");

  file >> number;

  file.close();


  return number;

}
