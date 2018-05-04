// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <fstream>
#include <iostream>
#include <string>

// #include "cells.hpp"

int read_data (/*CELLS *cells,std::string file_name*/)
{
  std::ifstream infile ("thefile.txt");

  int a, b;

  while (infile >> a >> b)
  {
    std::cout << a << b << std::endl;
  }

}



int main (void)
{

  const int  Dimension   = 1;
  const long Nrays       = 2;
  const long Ncells      = 10000;

  // CELLS <Dimension, Nrays> cells (Ncells);

  read_data();

  return(0);

}
