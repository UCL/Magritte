#include <iostream>
#include <string>

// #include <vtkXMLUnstructuredGridReader.h>
// #include <vtkUnstructuredGrid.h>
// #include <vtkSmartPointer.h>
// #include <vtkCellCenters.h>
// #include <vtkPointData.h>
// #include <vtkCellData.h>

#include "../parameters.hpp"
#include "../src/Magritte_config.hpp"
#include "../src/declarations.hpp"

#include "../src/definitions.hpp"
#include "../src/ray_tracing.hpp"
#include "../src/read_input.hpp"
#include "../src/write_txt_tools.hpp"

#include "../setup/setup_data_tools.hpp"

// #define NCELLS ncells



int main()
{


  // std::string inputfile = "Aori_0001.vtu";

  CELL cell[NCELLS];

  double temperature_gas[NCELLS];
  double temperature_dust[NCELLS];
  double pre_temperature_gas[NCELLS];


  read_vtu_input( "../" + inputfile, NCELLS, cell, temperature_gas, temperature_dust,
                  pre_temperature_gas );

  output_directory = "";

  write_grid("", cell);


//   /* Read the data from the .vtu file */
//
//   vtkSmartPointer<vtkXMLUnstructuredGridReader> reader
//     = vtkSmartPointer<vtkXMLUnstructuredGridReader>::New();
//
//   reader->SetFileName(inputfile.c_str());
//   reader->Update();
//
//   vtkUnstructuredGrid* ugrid = reader->GetOutput();
//
//
//   /* Filter out the cell centers */
//
//   vtkSmartPointer<vtkCellCenters> cellCentersFilter
//     = vtkSmartPointer<vtkCellCenters>::New();
//
// # if VTK_MAJOR_VERSION <= 5
//   cellCentersFilter->SetInputConnection(ugrid->GetProducerPort());
// # else
//   cellCentersFilter->SetInputData(ugrid);
// # endif
//   cellCentersFilter->VertexCellsOn();
//   cellCentersFilter->Update();
//
//
//   /* Extract the number of cell centers = number of grid points */
//
//   long ncells = cellCentersFilter->GetOutput()->GetNumberOfPoints();
//
//
//   // CELL *cell = new CELL[ncells];
//
//   // double *x = new double[ncells];
//   // double *y = new double[ncells];
//   // double *z = new double[ncells];
//   //
//   // double *vx = new double[ncells];
//   // double *vy = new double[ncells];
//   // double *vz = new double[ncells];
//   //
//   // double *density = new double[ncells];
//
//
//   /* Extract the locations of the cell centers */
//
//   for (long n=0; n<ncells; n++){
//
//     double point[3];
//
//     cellCentersFilter->GetOutput()->GetPoint(n, point);
//
//     cell[n].x = point[0];
//     cell[n].y = point[1];
//     cell[n].z = point[2];
//   }
//
//
//   /* Extract the cell data */
//
//   vtkCellData *cellData = ugrid->GetCellData();
//
//   int nr_of_arrays = cellData->GetNumberOfArrays();
//
//
//   for (int a = 0; a < nr_of_arrays; a++){
//
//     vtkDataArray* data = cellData->GetArray(a);
//
//     std::string name = data->GetName();
//
//
//     if (name == "rho"){
//     for (long n=0; n<ncells; n++){
//
//       cell[n].density = data->GetTuple1(n);
//     }
//     }
//
//     if (name == "v1"){
//     for (long n=0; n<ncells; n++){
//
//       cell[n].vx = data->GetTuple1(n);
//     }
//     }
//
//     if (name == "v2"){
//     for (long n=0; n<ncells; n++){
//
//       cell[n].vy = data->GetTuple1(n);
//     }
//     }
//
//     if (name == "v3"){
//     for (long n=0; n<ncells; n++){
//
//       cell[n].vz = data->GetTuple1(n);
//     }
//     }
//
//   }
//
//
//   /* Trace rays through the grid */
//
//   EVALPOINT *evalpoint = new EVALPOINT[ncells];
//
//   long *key = new long[NCELLS*NCELLS];
//
//   long *raytot = new long[NCELLS*NRAYS];
//
//   long *cum_raytot = new long[NCELLS*NRAYS];
//
//
//   // for (gridp=0; gridp<ncells; gridp++){
//   //
//   //   find_evalpoints(cell, evalpoint, key, raytot, cum_raytot, gridp);
//   // }
//
//   /* Write grid in .txt format */

  // std::string file_name = "grid.txt";

  //
  // FILE *file = fopen(file_name.c_str(), "w");
  //
  // if (file == NULL){
  //
  //   printf("Error opening file!\n");
  //   std::cout << file_name + "\n";
  //   exit(1);
  // }
  //
  // for (long n=0; n<ncells; n++){
  //
  //   fprintf( file, "%ld\t%lE\t%lE\t%lE\t%lE\t%lE\t%lE\t%lE\n",
  //            n,
  //            cell[n].x, cell[n].y, cell[n].z,
  //            cell[n].vx, cell[n].vy, cell[n].vz,
  //            cell[n].density );
  // }
  //
  // fclose(file);
  //
  //
  // return(0);

}
