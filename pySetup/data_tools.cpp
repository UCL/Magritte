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
