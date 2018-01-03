// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <iostream>
#include <string>

#include <vtkXMLUnstructuredGridReader.h>
#include <vtkXMLUnstructuredGridWriter.h>
#include <vtkUnstructuredGrid.h>
#include <vtkSmartPointer.h>
#include <vtkDoubleArray.h>

#include <vtkCellData.h>
#include <vtkVersion.h>

#include "../parameters.hpp"
#include "Magritte_config.hpp"
#include "declarations.hpp"

#include "write_vtu_tools.hpp"




// write_vtu_output: write all physical variables to vtu input grid
// ----------------------------------------------------------------

int write_vtu_output (long ncells, CELL *cell, std::string inputfile)
{


  // Read data from the .vtu file

  vtkSmartPointer<vtkXMLUnstructuredGridReader> reader
    = vtkSmartPointer<vtkXMLUnstructuredGridReader>::New();

  reader->SetFileName(inputfile.c_str());
  reader->Update();

  vtkUnstructuredGrid* ugrid = reader->GetOutput();


  // Reformat Magritte output to append it to the grid

  vtkSmartPointer<vtkDoubleArray> temp_gas
    = vtkSmartPointer<vtkDoubleArray>::New();

  vtkSmartPointer<vtkDoubleArray> temp_dust
    = vtkSmartPointer<vtkDoubleArray>::New();

  vtkSmartPointer<vtkDoubleArray> prev_temp_gas
    = vtkSmartPointer<vtkDoubleArray>::New();

  vtkSmartPointer<vtkDoubleArray> abn
    = vtkSmartPointer<vtkDoubleArray>::New();

  temp_gas->SetNumberOfComponents(1);
  temp_gas->SetNumberOfTuples(NCELLS);
  temp_gas->SetName("temperature_gas");

  temp_dust->SetNumberOfComponents(1);
  temp_dust->SetNumberOfTuples(NCELLS);
  temp_dust->SetName("temperature_dust");

  prev_temp_gas->SetNumberOfComponents(1);
  prev_temp_gas->SetNumberOfTuples(NCELLS);
  prev_temp_gas->SetName("prev_temperature_gas");

  abn->SetNumberOfComponents(NSPEC);
  abn->SetNumberOfTuples(NCELLS);
  abn->SetName("abundance");


  for (long n = 0; n < NCELLS; n++)
  {
    temp_gas ->InsertValue(n, cell[n].temperature.gas);
    temp_dust->InsertValue(n, cell[n].temperature.dust);
    prev_temp_gas->InsertValue(n, cell[n].temperature.gas_prev);


    double abundance[NSPEC];

    for (int spec = 0; spec < NSPEC; spec++)
    {
      abundance[spec] = species[spec].abn[n];
    }

    abn->InsertTuple(n, abundance);

  } // end of n loop over grid points


  // Add new arrays to grid

  ugrid->GetCellData()->AddArray(temp_gas);
  ugrid->GetCellData()->AddArray(temp_dust);
  ugrid->GetCellData()->AddArray(prev_temp_gas);
  ugrid->GetCellData()->AddArray(abn);


  // Write .vtu file

  std::string file_name = output_directory + "grid.vtu";

  vtkSmartPointer<vtkXMLUnstructuredGridWriter> writer
    = vtkSmartPointer<vtkXMLUnstructuredGridWriter>::New();

  writer->SetFileName(file_name.c_str());


# if VTK_MAJOR_VERSION <= 5

    writer->SetInput(ugrid);

# else

    writer->SetInputData(ugrid);

# endif

  writer->Write();


  return (0);

}
