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

#include "../parameters.hpp"
#include "Magritte_config.hpp"
#include "declarations.hpp"

#include "read_input.hpp"
#include "initializers.hpp"



// read_txt_input: read .txt input file
// ------------------------------------

int read_txt_input (std::string inputfile, long ncells, CELL *cell)
{

  char buffer[BUFFER_SIZE];   // buffer for a line of data


  // Read input file

  FILE *input = fopen(inputfile.c_str(), "r");


  // For all lines in input file

  for (long n = 0; n < NCELLS; n++)
  {
    fgets (buffer, BUFFER_SIZE, input);

    sscanf (buffer, "%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf",
            &(cell[n].x), &(cell[n].y), &(cell[n].z),
            &(cell[n].vx), &(cell[n].vy), &(cell[n].vz),
            &(cell[n].density));
  }


  fclose(input);


# if (!RESTART)

  initialize_temperature_gas (NCELLS, cell);

# else


  std::string INPUT_DIRECTORY = RESTART_DIRECTORY;


  // Read input temperature files to restart

  std::string tgas_file_name      = INPUT_DIRECTORY + "temperature_gas.txt";
  std::string tdust_file_name     = INPUT_DIRECTORY + "temperature_dust.txt";
  std::string prev_tgas_file_name = INPUT_DIRECTORY + "prev_temperature_gas.txt";

  FILE *tgas      = fopen(tgas_file_name.c_str(), "r");
  FILE *tdust     = fopen(tdust_file_name.c_str(), "r");
  FILE *prev_tgas = fopen(prev_tgas_file_name.c_str(), "r");


  // For all lines in input file

  for (long n = 0; n < NCELLS; n++)
  {
    fgets (buffer, BUFFER_SIZE, tgas);
    sscanf (buffer, "%lf", &(cell[n].temperature.gas));

    fgets (buffer, BUFFER_SIZE, tdust);
    sscanf (buffer, "%lf", &(cell[n].temperature.dust));

    fgets (buffer, BUFFER_SIZE, prev_tgas);
    sscanf (buffer, "%lf", &(cell[n].temperature.gas_prev));
  }

  fclose (tgas);
  fclose (tdust);
  fclose (prev_tgas);

# endif


  return (0);

}




// read_vtu_input: read input file
// -------------------------------

int read_vtu_input (std::string inputfile, long ncells, CELL *cell)
{


  // Read data from the .vtu file

  vtkSmartPointer<vtkXMLUnstructuredGridReader> reader
    = vtkSmartPointer<vtkXMLUnstructuredGridReader>::New();

  reader->SetFileName(inputfile.c_str());
  reader->Update();

  vtkUnstructuredGrid* ugrid = reader->GetOutput();


  // Extract cell centers

  vtkSmartPointer<vtkCellCenters> cellCentersFilter
    = vtkSmartPointer<vtkCellCenters>::New();

# if VTK_MAJOR_VERSION <= 5
  cellCentersFilter->SetInputConnection(ugrid->GetProducerPort());
# else
  cellCentersFilter->SetInputData(ugrid);
# endif
  cellCentersFilter->VertexCellsOn();
  cellCentersFilter->Update();


  for (long n = 0; n < NCELLS; n++)
  {
    double point[3];

    cellCentersFilter->GetOutput()->GetPoint(n, point);

    cell[n].x = point[0];
    cell[n].y = point[1];
    cell[n].z = point[2];
  }


  // Extract cell data

  vtkCellData *cellData = ugrid->GetCellData();

  int nr_of_arrays = cellData->GetNumberOfArrays();


  for (int a = 0; a < nr_of_arrays; a++)
  {
    vtkDataArray* data = cellData->GetArray(a);

    std::string name = data->GetName();


    if (name == "rho")
    {
      for (long n=0; n<NCELLS; n++)
      {
        cell[n].density = data->GetTuple1(n);
      }
    }

    if (name == "v1")
    {
      for (long n=0; n<NCELLS; n++)
      {
        cell[n].vx = data->GetTuple1(n);
      }
    }

    if (name == "v2")
    {
      for (long n=0; n<NCELLS; n++)
      {
        cell[n].vy = data->GetTuple1(n);
      }
    }


#   if (RESTART)

    if (name == "temperature_gas")
    {
      for (long n=0; n<NCELLS; n++)
      {
        cell[n].temperature.gas = data->GetTuple1(n);
      }
    }

    if (name == "temperature_dust")
    {
      for (long n=0; n<NCELLS; n++)
      {
        cell[n].temperature.dust = data->GetTuple1(n);
      }
    }

    if (name == "prev_temperature_gas")
    {
      for (long n=0; n<NCELLS; n++)
      {
        cell[n].temperature.gas_prev = data->GetTuple1(n);
      }
    }

#   endif


  }


  return(0);

}

/*-----------------------------------------------------------------------------------------------*/
