// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include <iostream>
#include <string>

#include <vtkXMLUnstructuredGridReader.h>
#include <vtkUnstructuredGrid.h>
#include <vtkSmartPointer.h>
#include <vtkCellCenters.h>
#include <vtkPointData.h>
#include <vtkCellData.h>
#include <vtkVersion.h>

#include "declarations.hpp"
#include "cells.hpp"


// read_input: read input file
// ---------------------------

int CELLS::read_input (std::string inputfile)
{

  // Read input file

  if      (INPUT_TYPE == vtu)
  {
    read_vtu_input (inputfile);
  }
  else if (INPUT_TYPE == txt)
  {
    read_txt_input (inputfile);
  }


  return(0);

}




// read_txt_input: read .txt input file
// ------------------------------------

int CELLS::read_txt_input (std::string inputfile)
{

  char buffer[BUFFER_SIZE];   // buffer for a line of data


  // Read input file

  FILE *input = fopen(inputfile.c_str(), "r");


  // For all lines in input file

  for (long p = 0; p < ncells; p++)
  {
    fgets (buffer, BUFFER_SIZE, input);

    sscanf (buffer, "%ld\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf",
            &(id[p]), &(x[p]), &(y[p]), &(z[p]),
            &(vx[p]), &(vy[p]), &(vz[p]),
            &(density[p]));
  }


  fclose(input);


# if (RESTART)


  // Get directory containing restart files

  std::string input_directory = RESTART_DIRECTORY;                  // relative
              input_directory = project_folder + input_directory;   // absolute

  std::string tgas_file_name = input_directory + "temperature_gas.txt";

  if (access (tgas_file_name.c_str(), F_OK) != -1)
  {

    // If temperature gas file exists

    FILE *tgas = fopen (tgas_file_name.c_str(), "r");

    for (long p = 0; p < ncells; p++)
    {
      fgets (buffer, BUFFER_SIZE, tgas);
      sscanf (buffer, "%lf", &(temperature_gas[p]));
    }

    fclose (tgas);
  }

  else
  {

    // If there is no temperature gas file

    for (long p = 0; p < ncells; p++)
    {
      temperature_gas[p] = T_CMB;
    }

  }


  std::string tdust_file_name = input_directory + "temperature_dust.txt";

  if (access (tdust_file_name.c_str(), F_OK) != -1)
  {

    // If temperature gas file exists

    FILE *tdust = fopen (tdust_file_name.c_str(), "r");

    for (long p = 0; p < ncells; p++)
    {
      fgets (buffer, BUFFER_SIZE, tdust);
      sscanf (buffer, "%lf", &(temperature_dust[p]));
    }

    fclose (tdust);
  }

  else
  {

    // If there is no temperature gas file

    for (long p = 0; p < ncells; p++)
    {
      temperature_dust[p] = T_CMB;
    }

  }


  std::string tgas_prev_file_name = input_directory + "temperature_gas_prev.txt";

  if (access (tgas_prev_file_name.c_str(), F_OK) != -1)
  {

    // If temperature gas file exists

    FILE *tgas_prev = fopen (tgas_prev_file_name.c_str(), "r");

    for (long p = 0; p < ncells; p++)
    {
      fgets (buffer, BUFFER_SIZE, tgas_prev);
      sscanf (buffer, "%lf", &(temperature_gas_prev[p]));
    }

    fclose (tgas_prev);
  }

  else
  {
    // If there is no temperature gas file

    for (long p = 0; p < ncells; p++)
    {
      temperature_gas_prev[p] = T_CMB;
    }
  }


  std::string abun_file_name = input_directory + "abundances.txt";

  if (access (abun_file_name.c_str(), F_OK) != -1)
  {

    // If temperature gas file exists

    FILE *abun = fopen (abun_file_name.c_str(), "r");

    for (long p = 0; p < ncells; p++)
    {
      for (int s = 0; s < NSPEC; s++)
      {
        fscanf (abun, "%lf", &(abundance[SINDEX(p,s)]));
      }
    }

    fclose (abun);
  }

  else
  {

    // If there is no temperature gas file

    for (long p = 0; p < ncells; p++)
    {
      for (int s = 0; s < NSPEC; s++)
      {
        abundance[SINDEX(p,s)] = 0.0;
      }
    }

  }


# endif


  return (0);

}




// read_vtu_input: read .vtu input file
// ------------------------------------

int CELLS::read_vtu_input (std::string inputfile)
{

  // Read data from the .vtu file

  vtkSmartPointer<vtkXMLUnstructuredGridReader> reader
    = vtkSmartPointer<vtkXMLUnstructuredGridReader>::New();

  reader->SetFileName(inputfile.c_str());
  reader->Update();

  vtkUnstructuredGrid* ugrid = reader->GetOutput();


# if (CELL_BASED)

    // Extract cell centers

    vtkSmartPointer<vtkCellCenters> cellCentersFilter
      = vtkSmartPointer<vtkCellCenters>::New();

#   if (VTK_MAJOR_VERSION <= 5)
      cellCentersFilter->SetInputConnection(ugrid->GetProducerPort());
#   else
      cellCentersFilter->SetInputData(ugrid);
#   endif

    cellCentersFilter->VertexCellsOn();
    cellCentersFilter->Update();

# endif


  for (long p = 0; p < ncells; p++)
  {
    double point[3];

#   if (CELL_BASED)
      cellCentersFilter->GetOutput()->GetPoint(p, point);
#   else
      ugrid->GetPoint(p, point);
#   endif

    x[p] = point[0];
    y[p] = point[1];
    z[p] = point[2];
  }



  // Extract data

# if (CELL_BASED)

    vtkCellData *cellData = ugrid->GetCellData();

    int nr_of_arrays = cellData->GetNumberOfArrays();

# else

    vtkPointData *pointData = ugrid->GetPointData();

    int nr_of_arrays = pointData->GetNumberOfArrays();

    printf("nr of arrays %d\n",nr_of_arrays);

# endif



  for (int a = 0; a < nr_of_arrays; a++)
  {

#   if (CELL_BASED)

      vtkDataArray* data = cellData->GetArray(a);

      std::string name = data->GetName();

      printf("name %s\n",name.c_str());

#   else

      vtkDataArray* data = pointData->GetArray(a);

      std::string name = data->GetName();

      printf("name %s\n",name.c_str());

#   endif

    if (name == NAME_DENSITY)
    {

      for (long p = 0; p < ncells; p++)
      {
        density[p] = data->GetTuple1(p);
      }
    }


    // In case velocity is stored as vector

    if (name == NAME_VELOCITY)
    {
      for (long p = 0; p < ncells; p++)
      {
        double *velocity = new double[3];

        data->GetTuple(p,velocity);

        vx[p] = velocity[0];
        vy[p] = velocity[1];
        vz[p] = velocity[2];
      }
    }


    // In case velocity components are stored individually

    if (name == NAME_VX)
    {
      for (long p = 0; p < ncells; p++)
      {
        vx[p] = data->GetTuple1(p);
      }
    }

    if (name == NAME_VY)
    {
      for (long p = 0; p < ncells; p++)
      {
        vy[p] = data->GetTuple1(p);
      }
    }

    if (name == NAME_VZ)
    {
      for (long p = 0; p < ncells; p++)
      {
        vz[p] = data->GetTuple1(p);
      }
    }


    // Extract chemical abundances

    if (name == NAME_CHEM_ABUNDANCES)
    {
      for (long p = 0; p < ncells; p++)
      {
        double *abundances = new double[NSPEC-2];

        data->GetTuple(p,abundances);

        for (int s = 0; s < NSPEC-2; s++)
        {
          abundance[SINDEX(p,s+1)] = abundances[s];
        }
      }
    }


    // In case temperature informations is available

#   if (RESTART)

    if (name == NAME_TEMPERATURE_GAS)
    {
      for (long p = 0; p < ncells; p++)
      {
        temperature_gas[p] = data->GetTuple1(p);
      }
    }

    if (name == NAME_TEMPERATURE_DUST)
    {
      for (long p = 0; p < ncells; p++)
      {
        temperature_dust[p] = data->GetTuple1(p);
      }
    }

    if (name == NAME_TEMPERATURE_GAS_PREV)
    {
      for (long p = 0; p < ncells; p++)
      {
        temperature_gas_prev[p] = data->GetTuple1(p);
      }
    }

#   endif


  }


  return(0);

}
