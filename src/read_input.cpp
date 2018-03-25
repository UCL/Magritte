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

    sscanf (buffer, "%ld\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf",
            &(cell[n].id), &(cell[n].x), &(cell[n].y), &(cell[n].z),
            &(cell[n].vx), &(cell[n].vy), &(cell[n].vz),
            &(cell[n].density));
  }


  fclose(input);


# if (!RESTART)

  initialize_temperature_gas (NCELLS, cell);

# else


  // Get directory containing restart files

  std::string input_directory = RESTART_DIRECTORY;
  input_directory = project_folder + input_directory;

  std::string tgas_file_name = input_directory + "temperature_gas.txt";

  if( access (tgas_file_name.c_str(), F_OK) != -1)
  {

    // If temperature gas file exists

    FILE *tgas = fopen (tgas_file_name.c_str(), "r");

    for (long n = 0; n < NCELLS; n++)
    {
      fgets (buffer, BUFFER_SIZE, tgas);
      sscanf (buffer, "%lf", &(cell[n].temperature.gas));
    }

    fclose (tgas);
  }

  else
  {

    // If there is no temperature gas file

    for (long n = 0; n < NCELLS; n++)
    {
      cell[n].temperature.gas = T_CMB;
    }

  }


  std::string tdust_file_name = input_directory + "temperature_dust.txt";

  if( access (tdust_file_name.c_str(), F_OK) != -1 )
  {

    // If temperature gas file exists

    FILE *tdust = fopen (tdust_file_name.c_str(), "r");

    for (long n = 0; n < NCELLS; n++)
    {
      fgets (buffer, BUFFER_SIZE, tdust);
      sscanf (buffer, "%lf", &(cell[n].temperature.dust));
    }

    fclose (tdust);
  }

  else
  {

    // If there is no temperature gas file

    for (long n = 0; n < NCELLS; n++)
    {
      cell[n].temperature.dust = T_CMB;
    }

  }


  std::string tgas_prev_file_name = input_directory + "temperature_gas_prev.txt";

  if( access (tgas_prev_file_name.c_str(), F_OK) != -1 )
  {

    // If temperature gas file exists

    FILE *tgas_prev = fopen (tgas_prev_file_name.c_str(), "r");

    for (long n = 0; n < NCELLS; n++)
    {
      fgets (buffer, BUFFER_SIZE, tgas_prev);
      sscanf (buffer, "%lf", &(cell[n].temperature.gas_prev));
    }

    fclose (tgas_prev);
  }

  else
  {
    // If there is no temperature gas file

    for (long n = 0; n < NCELLS; n++)
    {
      cell[n].temperature.gas_prev = T_CMB;
    }
  }


  std::string abun_file_name = input_directory + "abundances.txt";

  if( access (abun_file_name.c_str(), F_OK) != -1 )
  {

    // If temperature gas file exists

    FILE *abun = fopen (abun_file_name.c_str(), "r");

    for (long n = 0; n < NCELLS; n++)
    {
      for (int spec = 0; spec < NSPEC; spec++)
      {
        fscanf (abun, "%lf", &(cell[n].abundance[spec]));
      }
    }

    fclose (abun);
  }

  else
  {

    // If there is no temperature gas file

    for (long n = 0; n < NCELLS; n++)
    {
      for (int spec = 0; spec < NSPEC; spec++)
      {
        cell[n].abundance[spec] = 0.0;
      }
    }

  }


# endif


  return (0);

}




// read_neighbors: read neighbors
// ------------------------------

int read_txt_neighbors (std::string file_name, long ncells, CELL *cell)
{

  char buffer[BUFFER_SIZE];   // buffer for a line of data


  // Read input file

  FILE *file = fopen (file_name.c_str(), "r");


  if (file == NULL)
  {
    printf ("Error opening file!\n");
    std::cout << file_name + "\n";
    exit (1);
  }


  // For all lines in input file

  for (long n = 0; n < NCELLS; n++)
  {
    fscanf (file, "%ld", &(cell[n].n_neighbors));

    for (long r = 0; r < NRAYS; r++)
    {
      fscanf (file, "%ld", &(cell[n].neighbor[r]));
    }

    fgets (buffer, BUFFER_SIZE, file);
  }


  fclose(file);


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


  for (long n = 0; n < NCELLS; n++)
  {
    double point[3];

#   if (CELL_BASED)
      cellCentersFilter->GetOutput()->GetPoint(n, point);
#   else
      ugrid->GetPoint(n, point);
#   endif

    cell[n].x = point[0];
    cell[n].y = point[1];
    cell[n].z = point[2];
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

      for (long n = 0; n < NCELLS; n++)
      {
        cell[n].density = data->GetTuple1(n);
      }
    }


    // In case velocity is stored as vector

    if (name == NAME_VELOCITY)
    {
      for (long n = 0; n < NCELLS; n++)
      {
        double *velocity = new double[3];

        data->GetTuple(n,velocity);

        cell[n].vx = velocity[0];
        cell[n].vy = velocity[1];
        cell[n].vz = velocity[2];
      }
    }


    // In case velocity components are stored individually

    if (name == NAME_VX)
    {
      for (long n=0; n<NCELLS; n++)
      {
        cell[n].vx = data->GetTuple1(n);
      }
    }

    if (name == NAME_VY)
    {
      for (long n=0; n<NCELLS; n++)
      {
        cell[n].vy = data->GetTuple1(n);
      }
    }

    if (name == NAME_VZ)
    {
      for (long n=0; n<NCELLS; n++)
      {
        cell[n].vz = data->GetTuple1(n);
      }
    }


    // Extract chemical abundances

    if (name == NAME_CHEM_ABUNDANCES)
    {
      for (long n = 0; n < NCELLS; n++)
      {
        double *abundances = new double[NSPEC-2];

        data->GetTuple(n,abundances);

        for (int spec = 0; spec < NSPEC-2; spec++)
        {
          cell[n].abundance[spec+1] = abundances[spec];
        }
      }
    }


    // In case temperature informations is available

#   if (RESTART)

    if (name == NAME_TEMPERATURE_GAS)
    {
      for (long n = 0; n < NCELLS; n++)
      {
        cell[n].temperature.gas = data->GetTuple1(n);
      }
    }

    if (name == NAME_TEMPERATURE_DUST)
    {
      for (long n = 0; n < NCELLS; n++)
      {
        cell[n].temperature.dust = data->GetTuple1(n);
      }
    }

    if (name == NAME_TEMPERATURE_GAS_PREV)
    {
      for (long n = 0; n < NCELLS; n++)
      {
        cell[n].temperature.gas_prev = data->GetTuple1(n);
      }
    }

#   endif


  }


  return(0);

}
