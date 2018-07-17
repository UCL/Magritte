#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>
using namespace std;
#include <mpi.h>

#include "folders.hpp"


/// TIMER: struct for precise process timing
////////////////////////////////////////////

class TIMER
{
  private:

    string name;

    int world_size;
    int world_rank;

    chrono::duration <double> interval;
    chrono::high_resolution_clock::time_point initial;

	  vector<double> times;


  public:


	  ///  Constructor for TIMER
	  //////////////////////////

	  TIMER (string timer_name)
	  {
      name = timer_name;

	  	MPI_Comm_size (MPI_COMM_WORLD, &world_size);
	  	MPI_Comm_rank (MPI_COMM_WORLD, &world_rank);

	  	times.resize (world_size);
	  }


	  ///  start: start timer i.e. set initial time stamp
	  ///////////////////////////////////////////////////

    void start ()
    {
      initial = chrono::high_resolution_clock::now();
    }


	  ///  stop: stop timer and calculate interval for every process
	  //////////////////////////////////////////////////////////////

    void stop ()
    {
      interval = chrono::high_resolution_clock::now() - initial;

	  	times[world_rank] = interval.count();

	  	MPI_Allgather (MPI_IN_PLACE,
	  			           0,
	  								 MPI_DATATYPE_NULL,
	  								 times.data(),
	  								 1,
	  								 MPI_DOUBLE,
	  								 MPI_COMM_WORLD);
    }


	  ///  print_to_file: let rank 0 process print times for every rank to file
	  ///    @param[in] file_name: name of the file to print to
	  /////////////////////////////////////////////////////////////////////////


	  void print_to_file ()
	  {
	  	if (world_rank == 0)
	  	{
        string file_name = output_folder + "timer_" + name + ".txt";

	  		ofstream outFile (file_name, ios_base::app);

	  	  for (int w = 0; w < world_size; w++)
	  	  {
          outFile << world_size << "\t" << w << "\t" << times[w] << endl;
	  	  }

	  		outFile.close();
	  	}
	  }


	  ///  print: let rank 0 process print times for every rank
	  /////////////////////////////////////////////////////////

    void print ()
    {
	  	if (world_rank == 0)
	  	{
        cout << "Timer" << name << ":" << endl;

	  	  for (int w = 0; w < world_size; w++)
	  	  {
          cout << "rank[" << w << "]: time = " << times[w] << " seconds" << endl;
	  	  }
	  	}
    }

};
