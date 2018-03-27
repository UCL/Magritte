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
#include <vtkCellCenters.h>
#include <vtkDoubleArray.h>
#include <vtkLongArray.h>
#include <vtkCellData.h>
#include <vtkVersion.h>

#include <vtkCellData.h>
#include <vtkVersion.h>

#include "declarations.hpp"
#include "write_vtu_tools.hpp"


// write_vtu_output: write all physical variables to vtu input grid
// ----------------------------------------------------------------

int write_vtu_output (std::string tag, long ncells, CELLS *cells)
{

  // Read data from .vtu file on which we need to append data

  vtkSmartPointer<vtkXMLUnstructuredGridReader> reader
    = vtkSmartPointer<vtkXMLUnstructuredGridReader>::New();

  reader->SetFileName(inputfile.c_str());
  reader->Update();

  vtkUnstructuredGrid *ugrid = reader->GetOutput();


  // Reformat Magritte output to append it to grid

  vtkSmartPointer<vtkLongArray> id
    = vtkSmartPointer<vtkLongArray>::New();

  vtkSmartPointer<vtkDoubleArray> density
    = vtkSmartPointer<vtkDoubleArray>::New();

  vtkSmartPointer<vtkDoubleArray> temp_gas
    = vtkSmartPointer<vtkDoubleArray>::New();

  vtkSmartPointer<vtkDoubleArray> temp_dust
    = vtkSmartPointer<vtkDoubleArray>::New();

  vtkSmartPointer<vtkDoubleArray> temp_gas_prev
    = vtkSmartPointer<vtkDoubleArray>::New();

  vtkSmartPointer<vtkDoubleArray> abn
    = vtkSmartPointer<vtkDoubleArray>::New();


  id->SetNumberOfComponents(1);
  id->SetNumberOfTuples(NCELLS);
  id->SetName("id");

  density->SetNumberOfComponents(1);
  density->SetNumberOfTuples(NCELLS);
  density->SetName("density");

  temp_gas->SetNumberOfComponents(1);
  temp_gas->SetNumberOfTuples(NCELLS);
  temp_gas->SetName("temperature_gas");

  temp_dust->SetNumberOfComponents(1);
  temp_dust->SetNumberOfTuples(NCELLS);
  temp_dust->SetName("temperature_dust");

  temp_gas_prev->SetNumberOfComponents(1);
  temp_gas_prev->SetNumberOfTuples(NCELLS);
  temp_gas_prev->SetName("temperature_gas_prev");

  abn->SetNumberOfComponents(NSPEC);
  abn->SetNumberOfTuples(NCELLS);
  abn->SetName("abundance");


  for (long p = 0; p < NCELLS; p++)
  {
    id           ->InsertValue(p, cells->id[p]);
    density      ->InsertValue(p, cells->density[p]);
    temp_gas     ->InsertValue(p, cells->temperature_gas[p]);
    temp_dust    ->InsertValue(p, cells->temperature_dust[p]);
    temp_gas_prev->InsertValue(p, cells->temperature_gas_prev[p]);


    double abundance[NSPEC];

    for (int s = 0; s < NSPEC; s++)
    {
      abundance[s] = cells->abundance[SINDEX(p,s)];
    }

    abn->InsertTuple(p, abundance);
  }


  // Add new arrays to grid

  ugrid->GetCellData()->AddArray(id);
  ugrid->GetCellData()->AddArray(density);
  ugrid->GetCellData()->AddArray(temp_gas);
  ugrid->GetCellData()->AddArray(temp_dust);
  ugrid->GetCellData()->AddArray(temp_gas_prev);
  ugrid->GetCellData()->AddArray(abn);


  // Write .vtu file

  if (tag != "")
  {
    tag = "_" + tag;
  }

  std::string file_name = output_directory + "grid" + tag + ".vtu";


  vtkSmartPointer<vtkXMLUnstructuredGridWriter> writer
    = vtkSmartPointer<vtkXMLUnstructuredGridWriter>::New();

  writer->SetFileName(file_name.c_str());


# if (VTK_MAJOR_VERSION <= 5)

    writer->SetInput(ugrid);

# else

    writer->SetInputData(ugrid);

# endif


  writer->Write();


  return (0);

}
