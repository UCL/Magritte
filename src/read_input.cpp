/* Frederik De Ceuster - University College London & KU Leuven                                   */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/* read_input: read the input files                                                              */
/*                                                                                               */
/* (NEW)                                                                                         */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/



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


/* read_txt_input: read the .txt input file                                                      */
/*-----------------------------------------------------------------------------------------------*/

int read_txt_input(std::string grid_inputfile, GRIDPOINT *gridpoint)
{


  char buffer[BUFFER_SIZE];                                         /* buffer for a line of data */


  /* Read input file */

  FILE *input = fopen(grid_inputfile.c_str(), "r");


  /* For all lines in the input file */

  for (long n=0; n<NGRID; n++){

    fgets( buffer, BUFFER_SIZE, input );

    sscanf( buffer, "%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf",
            &(gridpoint[n].x), &(gridpoint[n].y), &(gridpoint[n].z),
            &(gridpoint[n].vx), &(gridpoint[n].vy), &(gridpoint[n].vz),
            &(gridpoint[n].density) );
  }


  fclose(input);


  return EXIT_SUCCESS;

}

/*-----------------------------------------------------------------------------------------------*/





/* read_vtu_input: read the input file                                                           */
/*-----------------------------------------------------------------------------------------------*/

int read_vtu_input(std::string grid_inputfile, GRIDPOINT *gridpoint)
{


  /* Read the data from the .vtu file */

  vtkSmartPointer<vtkXMLUnstructuredGridReader> reader
    = vtkSmartPointer<vtkXMLUnstructuredGridReader>::New();

  reader->SetFileName(grid_inputfile.c_str());
  reader->Update();

  vtkUnstructuredGrid* ugrid = reader->GetOutput();


  /* Extract the cell centers */

  vtkSmartPointer<vtkCellCenters> cellCentersFilter
    = vtkSmartPointer<vtkCellCenters>::New();

# if VTK_MAJOR_VERSION <= 5
  cellCentersFilter->SetInputConnection(ugrid->GetProducerPort());
# else
  cellCentersFilter->SetInputData(ugrid);
# endif
  cellCentersFilter->VertexCellsOn();
  cellCentersFilter->Update();


  for (long n=0; n<NGRID; n++){

    double point[3];

    cellCentersFilter->GetOutput()->GetPoint(n, point);

    gridpoint[n].x = point[0];
    gridpoint[n].y = point[1];
    gridpoint[n].z = point[2];
  }


  /* Extract the cell data */

  vtkCellData *cellData = ugrid->GetCellData();

  int nr_of_arrays = cellData->GetNumberOfArrays();


  for (int a = 0; a < nr_of_arrays; a++){

    vtkDataArray* data = cellData->GetArray(a);

    std::string name = data->GetName();


    if (name == "rho"){
    for (long n=0; n<NGRID; n++){

      gridpoint[n].density = data->GetTuple1(n);
    }
    }

    if (name == "v1"){
    for (long n=0; n<NGRID; n++){

      gridpoint[n].vx = data->GetTuple1(n);
    }
    }

    if (name == "v2"){
    for (long n=0; n<NGRID; n++){

      gridpoint[n].vy = data->GetTuple1(n);
    }
    }

    if (name == "v3"){
    for (long n=0; n<NGRID; n++){

      gridpoint[n].vz = data->GetTuple1(n);
    }
    }

  }


  return EXIT_SUCCESS;

}

/*-----------------------------------------------------------------------------------------------*/
