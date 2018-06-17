// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <string>
#include <fstream>
#include <iostream>
using namespace std;
#include <Eigen/Core>
using namespace Eigen;

#include "catch.hpp"

#include "../src/frequencies.hpp"
#include "../src/constants.hpp"
#include "../src/GridTypes.hpp"
#include "../../Lines/src/linedata.hpp"

#define EPS 1.0E-7


///  relative_error: returns relative error between A and B
///////////////////////////////////////////////////////////

double relative_error (double A, double B)
{
  return 2.0 * fabs(A-B) / fabs(A+B);
}


///  setup_linedata: setup simple line data for testing
///////////////////////////////////////////////////////

int setup_linedata (LINEDATA& linedata)
{

	linedata.nlspec    =  5;
	linedata.nrad.resize (linedata.nlspec);
	linedata.nrad      = {1, 1, 1, 1, 1};
	linedata.irad.resize (linedata.nlspec);
	
	for (int i=0; i<linedata.nlspec; i++) linedata.irad[i].resize (linedata.nrad[i]);
	linedata.irad      = {{1}, {1}, {1}, {1}, {1}};
	linedata.jrad.resize (linedata.nlspec);
	for (int i=0; i<linedata.nlspec; i++) linedata.jrad[i].resize (linedata.nrad[i]);
	linedata.jrad      = {{0}, {0}, {0}, {0}, {0}};

	linedata.frequency.resize (linedata.nlspec);

	linedata.frequency[0].resize (2,2);
	Matrix2d freqs0;
  freqs0 << 0.0, 3.0, 3.0, 0.0;
	linedata.frequency[0] = freqs0;

	linedata.frequency[1].resize (2,2);
	Matrix2d freqs1;
  freqs1 << 0.0, 2.0, 2.0, 0.0;
	linedata.frequency[1] = freqs1;

	linedata.frequency[2].resize (2,2);
	Matrix2d freqs2;
  freqs2 << 0.0, 1.0, 1.0, 0.0;
	linedata.frequency[2] = freqs2;
	
	linedata.frequency[3].resize (2,2);
	Matrix2d freqs3;
  freqs3 << 0.0, 5.0, 5.0, 0.0;
	linedata.frequency[3] = freqs3;

	linedata.frequency[4].resize (2,2);
	Matrix2d freqs4;
  freqs4 << 0.0, 4.0, 4.0, 0.0;
	linedata.frequency[4] = freqs4;


  return (0);

}




TEST_CASE ("FREQUENCIES Constructor")
{
	long ncells = 1;

	LINEDATA linedata;

	FREQUENCIES frequencies (ncells, linedata);

	
	CHECK (true);
}




TEST_CASE ("Reset")
{
	long ncells = 10;

	LINEDATA linedata;
	
	setup_linedata (linedata);

	FREQUENCIES frequencies (ncells, linedata);
	
	TEMPERATURE temperature (ncells);

	for (long p = 0; p < ncells; p ++)
	{
  	temperature.gas[p] = 100000.0;
	}

	frequencies.reset (linedata, temperature);


 	SECTION ("Ordering")
	{

		vector<double> freqs (frequencies.nfreq);
		long index = 0;

	  for (int f = 0; f < frequencies.nfreq_red; f++)
	  {
		  for (int lane = 0; lane < n_vector_lanes; lane++)
 		  {
        freqs[index] = frequencies.all[0][f].getlane(lane);
				index++;
		  }
	  }


	  for (int f = 1; f < frequencies.nfreq; f++)
	  {
      CHECK (freqs[f-1] < freqs[f]);
	  }
	}


	SECTION ("Frequency values")
	{
    for (int l = 0; l < linedata.nlspec; l++)
	  {
	  	for (int k = 0; k < linedata.nrad[l]; k++)
	  	{
	  		int i = linedata.irad[l][k];
	  		int j = linedata.jrad[l][k];

	  		long nr = frequencies.nr_line[0][l][k][NR_LINE_CENTER];
	  		
				long    f = nr / n_vector_lanes;   
				long lane = nr % n_vector_lanes;   

	  		double freq     = frequencies.all[0][f].getlane(lane);
	  		double freq_ref = linedata.frequency[l](i,j);

	  		CHECK (relative_error(freq, freq_ref) == Approx(0.0).epsilon(EPS));
	  	}
	  }	
	}
	

}
