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
#include "tools.hpp"

#include "frequencies.hpp"
#include "constants.hpp"
#include "GridTypes.hpp"

#include "test_data/linedata_config.hpp"
#include "Lines/src/linedata.hpp"

#define EPS 1.0E-7


TEST_CASE ("FREQUENCIES Constructor")
{
	const long ncells = 1;

	LINEDATA linedata;

	FREQUENCIES frequencies (ncells, linedata);

	for (int l = 0; l < linedata.nlspec; l++)
	{
		for (int k = 0; k < linedata.nrad[l]; k++)
		{
      cout << linedata.frequency[l][k] << endl;
    }
	}

	CHECK (true);
}




TEST_CASE ("Reset")
{
//	long ncells = 10;
//
//	LINEDATA linedata;
//
//	setup_linedata (linedata);
//
//	FREQUENCIES frequencies (ncells, linedata);
//
//	TEMPERATURE temperature (ncells);
//
//	for (long p = 0; p < ncells; p ++)
//	{
//  	temperature.gas[p] = 100000.0;
//	}
//
//	frequencies.reset (linedata, temperature);
//
//
// 	SECTION ("Ordering")
//	{
//
//		vector<double> freqs (frequencies.nfreq);
//		long index = 0;
//
//	  for (int f = 0; f < frequencies.nfreq_red; f++)
//	  {
//		  for (int lane = 0; lane < n_simd_lanes; lane++)
// 		  {
//        freqs[index] = frequencies.all[0][f].getlane(lane);
//				index++;
//		  }
//	  }
//
//
//	  for (int f = 1; f < frequencies.nfreq; f++)
//	  {
//      CHECK (freqs[f-1] < freqs[f]);
//	  }
//	}
//
//
//	SECTION ("Frequency values")
//	{
//    for (int l = 0; l < linedata.nlspec; l++)
//	  {
//	  	for (int k = 0; k < linedata.nrad[l]; k++)
//	  	{
//	  		int i = linedata.irad[l][k];
//	  		int j = linedata.jrad[l][k];
//
//	  		long nr = frequencies.nr_line[0][l][k][NR_LINE_CENTER];
//
//				long    f = nr / n_simd_lanes;
//				long lane = nr % n_simd_lanes;
//
//	  		double freq     = frequencies.all[0][f].getlane(lane);
//	  		double freq_ref = linedata.frequency[l](i,j);
//
//	  		CHECK (relative_error(freq, freq_ref) == Approx(0.0).epsilon(EPS));
//	  	}
//	  }
//	}
//

}
