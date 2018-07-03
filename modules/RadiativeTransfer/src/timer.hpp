#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>
using namespace std;
#include <mpi.h>


/// TIMER: struct for precise process timing
////////////////////////////////////////////

struct TIMER               
{

  chrono::duration <double> interval;
  chrono::high_resolution_clock::time_point initial;

	vector<double> times;


	///  Constructor for TIMER
	//////////////////////////

	TIMER ()
	{
		int world_size;
		MPI_Comm_size (MPI_COMM_WORLD, &world_size);

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

		int world_rank;
		MPI_Comm_rank (MPI_COMM_WORLD, &world_rank);

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


	void print_to_file (string file_name)
	{
		int world_size;
		MPI_Comm_size (MPI_COMM_WORLD, &world_size);

		int world_rank;
		MPI_Comm_rank (MPI_COMM_WORLD, &world_rank);

		if (world_rank == 0)
		{
			ofstream outFile (file_name);

		  for (int w = 0; w < world_size; w++)
		  {
        outFile << w << "\t" << times[w] << endl;
		  }

			outFile.close();
		}
	}


	///  print: let rank 0 process print times for every rank
	/////////////////////////////////////////////////////////

  void print ()
  {
		int world_size;
		MPI_Comm_size (MPI_COMM_WORLD, &world_size);

		int world_rank;
		MPI_Comm_rank (MPI_COMM_WORLD, &world_rank);
		
		if (world_rank == 0)
		{
		  for (int w = 0; w < world_size; w++)
		  {
        cout << "rank[" << w << "]: time = " << times[w] << " seconds" << endl;
		  }
		}
  }

  void print (std::string text)
  {
    cout << text << " time  = " << interval.count() << " seconds" << endl;
  }

};
