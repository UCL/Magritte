#include <iostream>
#include <string>

#include <vtkXMLUnstructuredGridReader.h>
#include <vtkUnstructuredGrid.h>
#include <vtkSmartPointer.h>
#include <vtkCellCenters.h>
#include <vtkPointData.h>
#include <vtkCellData.h>

// #include "../src/declarations.hpp"
// #include "../src/ray_tracing.hpp"

#define NGRID ngrid



int main()
{


  std::string grid_inputfile = "Aori_0001.vtu";


  /* Read the data from the .vtu file */

  vtkSmartPointer<vtkXMLUnstructuredGridReader> reader
    = vtkSmartPointer<vtkXMLUnstructuredGridReader>::New();

  reader->SetFileName(grid_inputfile.c_str());
  reader->Update();

  vtkUnstructuredGrid* ugrid = reader->GetOutput();


  /* Filter out the cell centers */

  vtkSmartPointer<vtkCellCenters> cellCentersFilter
    = vtkSmartPointer<vtkCellCenters>::New();

# if VTK_MAJOR_VERSION <= 5
  cellCentersFilter->SetInputConnection(ugrid->GetProducerPort());
# else
  cellCentersFilter->SetInputData(ugrid);
# endif
  cellCentersFilter->VertexCellsOn();
  cellCentersFilter->Update();


  /* Extract the number of cell centers = number of grid points */

  long ngrid = cellCentersFilter->GetOutput()->GetNumberOfPoints();


  // GRIDPOINT *gridpoint = new GRIDPOINT[ngrid];

  // double *x = new double[ngrid];
  // double *y = new double[ngrid];
  // double *z = new double[ngrid];
  //
  // double *vx = new double[ngrid];
  // double *vy = new double[ngrid];
  // double *vz = new double[ngrid];
  //
  // double *density = new double[ngrid];


  /* Extract the locations of the cell centers */

  for (long n=0; n<ngrid; n++){

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
    for (long n=0; n<ngrid; n++){

      gridpoint[n].density = data->GetTuple1(n);
    }
    }

    if (name == "v1"){
    for (long n=0; n<ngrid; n++){

      gridpoint[n].vx = data->GetTuple1(n);
    }
    }

    if (name == "v2"){
    for (long n=0; n<ngrid; n++){

      gridpoint[n].vy = data->GetTuple1(n);
    }
    }

    if (name == "v3"){
    for (long n=0; n<ngrid; n++){

      gridpoint[n].vz = data->GetTuple1(n);
    }
    }

  }


  /* Trace rays through the grid */

  EVALPOINT *evalpoint = new EVALPOINT[ngrid];

  long *key = new long[NGRID*NGRID];

  long *raytot = new long[NGRID*NRAYS];

  long *cum_raytot = new long[NGRID*NRAYS];


  // for (gridp=0; gridp<ngrid; gridp++){
  //
  //   get_local_evalpoint(gridpoint, evalpoint, key, raytot, cum_raytot, gridp);
  // }

  /* Write grid in .txt format */

  std::string file_name = "grid.txt";


  FILE *file = fopen(file_name.c_str(), "w");

  if (file == NULL){

    printf("Error opening file!\n");
    std::cout << file_name + "\n";
    exit(1);
  }

  for (long n=0; n<ngrid; n++){

    fprintf( file, "%ld\t%lE\t%lE\t%lE\t%lE\t%lE\t%lE\t%lE\n",
             n,
             gridpoint[n].x, gridpoint[n].y, gridpoint[n].z,
             gridpoint[n].vx, gridpoint[n].vy, gridpoint[n].vz,
             gridpoint[n].density );
  }

  fclose(file);


  return EXIT_SUCCESS;

}
