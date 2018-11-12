// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <mpi.h>
#include <omp.h>

#include <iostream>
#include <fstream>
#include <iomanip>
using namespace std;

#include "image.hpp"
#include "folders.hpp"
#include "constants.hpp"
#include "GridTypes.hpp"
#include "mpiTools.hpp"


///  Constructor for IMAGE
//////////////////////////

RAYDATA ::
RAYDATA (const long num_of_cells)
  : ncells (num_of_cells)
{


}   // END OF CONSTRUCTOR




///  print: write out the images
///    @param[in] tag: tag for output file
//////////////////////////////////////////

int IMAGE ::
    print (const string tag) const
{

}
