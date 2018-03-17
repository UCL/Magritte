// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <stdio.h>
#include <stdlib.h>

#include <string>

#include <vtkXMLUnstructuredGridReader.h>
#include <vtkUnstructuredGrid.h>
#include <vtkSmartPointer.h>
#include <vtkCellCenters.h>
#include <vtkPointData.h>
#include <vtkCellData.h>
#include <vtkVersion.h>

// #include "../src/parameters.hpp"
#include "data_tools.hpp"


// get_NCELLS_vtu: Count number of grid points in .vtu input file
// --------------------------------------------------------------

long get_NCELLS_vtu (std::string inputfile, std::string grid_type)
{
  long ncells = 0;


  // Read data from .vtu file

  vtkSmartPointer<vtkXMLUnstructuredGridReader> reader
    = vtkSmartPointer<vtkXMLUnstructuredGridReader>::New();

  reader->SetFileName(inputfile.c_str());
  reader->Update();

  vtkUnstructuredGrid* ugrid = reader->GetOutput();


  if (grid_type == "cell_based")
  {
    vtkSmartPointer<vtkCellCenters> cellCentersFilter
      = vtkSmartPointer<vtkCellCenters>::New();

#   if (VTK_MAJOR_VERSION <= 5)
      cellCentersFilter->SetInputConnection(ugrid->GetProducerPort());
#   else
      cellCentersFilter->SetInputData(ugrid);
#   endif

    cellCentersFilter->VertexCellsOn();
    cellCentersFilter->Update();

    ncells = cellCentersFilter->GetOutput()->GetNumberOfPoints();
  }

  else if (grid_type == "point_based")
  {
    ncells = ugrid->GetNumberOfPoints();
  }


  return ncells;

}



//
// // get_nlev: get number of energy levels from data file in LAMBDA/RADEX format
// // ---------------------------------------------------------------------------
//
// int get_nlev (std::string datafile)
// {
//
//   int nlev = 0;   // number of levels
//
//
//   // Open data file
//
//   FILE *file = fopen (datafile.c_str(), "r");
//
//
//   // Skip first 5 lines
//
//   for (int l = 0; l < 5; l++)
//   {
//     fscanf (file, "%*[^\n]\n");
//   }
//
//
//   // Read number of energy levels
//
//   fscanf (file, "%d \n", &nlev);
//
//
//   fclose (file);
//
//
//   return nlev;
//
// }
//
//
//
//
// // get_nrad: get number of radiative transitions from data file in LAMBDA/RADEX format
// // -----------------------------------------------------------------------------------
//
// int get_nrad (std::string datafile)
// {
//
//   int nrad = 0;                     // number of radiative transitions
//
//   int nlev = get_nlev (datafile);   // number of levels
//
//
//   // Open data file
//
//   FILE *file = fopen (datafile.c_str(), "r");
//
//
//   // Skip first 8+nlev lines
//
//   for (int l = 0; l < 8+nlev; l++)
//   {
//     fscanf (file, "%*[^\n]\n");
//   }
//
//
//   // Read number of radiative transitions
//
//   fscanf (file, "%d \n", &nrad);
//
//
//   fclose (file);
//
//
//   return nrad;
//
// }
//
//
//
//
// // get_ncolpar: get number of collision partners from data file in LAMBDA/RADEX format
// // -----------------------------------------------------------------------------------
//
// int get_ncolpar (std::string datafile)
// {
//
//
//   int ncolpar = 0;                  // number of collision partners
//
//   int nlev = get_nlev (datafile);   // number of levels
//   int nrad = get_nrad (datafile);   // number of radiative transitions
//
//
//   // Open data file
//
//   FILE *file = fopen (datafile.c_str(), "r");
//
//
//   // Skip first 11+nlev+nrad lines
//
//   for (int l = 0; l < 11+nlev+nrad; l++)
//   {
//     fscanf (file, "%*[^\n]\n");
//   }
//
//
//   // Read number of collision partners
//
//   fscanf (file, "%d \n", &ncolpar);
//
//
//   fclose (file);
//
//
//   return ncolpar;
//
// }
//
//
//
//
// // get_ncoltran: get number of collisional transitions from data file in LAMBDA/RADEX format
// // -----------------------------------------------------------------------------------------
//
// int get_ncoltran (std::string datafile, int *ncoltran, int *ncolpar, int *cum_ncolpar, int lspec)
// {
//
//
//   int loc_ncoltran = 0;            // number of collision partners
//
//   int nlev = get_nlev(datafile);   // number of levels
//   int nrad = get_nrad(datafile);   // number of radiative transitions
//
//
//   // Open data file
//
//   FILE *file = fopen (datafile.c_str(), "r");
//
//
//   // Skip first 15+nlev+nrad lines
//
//   for (int l = 0; l < 15+nlev+nrad; l++)
//   {
//     fscanf (file, "%*[^\n]\n");
//   }
//
//
//   // Skip collision partners that are already done
//
//   for (int par = 0; par < ncolpar[lspec]; par++)
//   {
//     if (ncoltran[LSPECPAR(lspec,par)] > 0)
//     {
//
//       // Skip next 9+ncoltran lines
//
//       for (int l = 0; l < 9+ncoltran[LSPECPAR(lspec,par)]; l++)
//       {
//         fscanf (file, "%*[^\n]\n");
//       }
//     }
//   }
//
//
//   // Read number of collisional transitions
//
//   fscanf (file, "%d \n", &loc_ncoltran);
//
//
//   fclose (file);
//
//
//   return loc_ncoltran;
//
// }
//
//
//
//
// // get_ncoltemp: get number of collisional temperatures from data file in LAMBDA/RADEX format
// // ------------------------------------------------------------------------------------------
//
// int get_ncoltemp (std::string datafile, int *ncoltran, int *cum_ncolpar, int partner, int lspec)
// {
//
//
//   int ncoltemp = 0;                       // number of collision partners
//
//   int nlev    = get_nlev (datafile);      // number of levels
//   int nrad    = get_nrad (datafile);      // number of radiative transitions
//   int ncolpar = get_ncolpar (datafile);   // number of collision partners
//
//
//   // Open data file
//
//   FILE *file = fopen (datafile.c_str(), "r");
//
//
//   // Skip first 17+nlev+nrad lines
//
//   for (int l = 0; l < 17+nlev+nrad; l++)
//   {
//     fscanf (file, "%*[^\n]\n");
//   }
//
//
//   // Skip collision partners before "partner"
//
//   for (int par = 0; par < partner; par++)
//   {
//
//     // Skip next 9+ncoltran lines
//
//     for (int l = 0; l < 9+ncoltran[LSPECPAR(lspec,par)]; l++)
//     {
//       fscanf (file, "%*[^\n]\n");
//     }
//
//   }
//
//
//   // Read number of collisional temperatures
//
//   fscanf (file, "%d \n", &ncoltemp);
//
//
//   fclose (file);
//
//
//   return ncoltemp;
//
// }
