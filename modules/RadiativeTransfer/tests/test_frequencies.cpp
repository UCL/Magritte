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
#include "folders.hpp"
#include "constants.hpp"
#include "GridTypes.hpp"
#include "lines.hpp"

//#include "test_data/linedata_config.hpp"
#include "Lines/src/linedata.hpp"
#include "Lines/src/linedata.cpp"

#define EPS 1.0E-7


TEST_CASE ("FREQUENCIES Constructor")
{

  const long ncells = 2;

  const string linedata_folder = Magritte_folder + "tests/test_data/linedata/";


  LINEDATA linedata (linedata_folder);

  cout << "nlspec = " << linedata.nlspec << endl;

  FREQUENCIES frequencies (ncells, linedata);
  TEMPERATURE temperature (ncells);

  temperature.gas[0] = 1.0E1;
  temperature.gas[1] = 1.0E5;

  frequencies.reset (linedata, temperature);


  SECTION ("Line frequencies")
  {

    for (long p = 0; p < ncells; p++)
    {
      for (int l = 0; l < linedata.nlspec; l++)
      {
        for (int k = 0; k < linedata.nrad[l]; k++)
        {
          long index = frequencies.nr_line[p][l][k][0];

          CHECK (linedata.frequency[l][k] == frequencies.nu[p][index]);
        }
      }
    }

  }


  SECTION ("Ordering of variables that are supposed to be ordered")
  {

    for (long p = 0; p < ncells; p++)
    {
      for (long f = 1; f < frequencies.nfreq_red; f++)
      {
        CHECK (frequencies.nu[p][f] > frequencies.nu[p][f-1]);
      }

      for (long fl = 1; fl < frequencies.nlines; fl++)
      {
        CHECK (frequencies.line[fl] > frequencies.line[fl-1]);
      }
    }
  }


  SECTION ("line_index")
  {

    LINES lines (ncells, linedata);

    for (long p = 0; p < ncells; p++)
    {
      for (int l = 0; l < linedata.nlspec; l++)
      {
        for (int k = 0; k < linedata.nrad[l]; k++)
        {
          for (long fl = 0; fl < frequencies.nlines; fl++)
          {
            if (frequencies.line[fl] == linedata.frequency[l][k])
            {
              const long line_index = frequencies.line_index[fl];

              CHECK(lines.index(p,l,k) == lines.index(p,line_index));
            }
          }
        }
      }
    }

  }


}



TEST_CASE ("Ordering of variables that are supposed to be ordered")
{
  const long ncells = 2;

  const string linedata_folder = Magritte_folder + "tests/test_data/linedata/";


  LINEDATA linedata (linedata_folder);

  FREQUENCIES frequencies (ncells, linedata);
  TEMPERATURE temperature (ncells);

  temperature.gas[0] = 1.0E1;
  temperature.gas[1] = 1.0E5;

  frequencies.reset (linedata, temperature);

  for (long p = 0; p < ncells; p++)
  {
    for (long f = 1; f < frequencies.nfreq_red; f++)
    {
      CHECK (frequencies.nu[p][f] > frequencies.nu[p][f-1]);
    }

    for (long f = 1; f < frequencies.nlines; f++)
    {
      CHECK (frequencies.line[f] > frequencies.line[f-1]);
    }
  }

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
//        freqs[index] = frequencies.nu[0][f].getlane(lane);
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
//	  		double freq     = frequencies.nu[0][f].getlane(lane);
//	  		double freq_ref = linedata.frequency[l](i,j);
//
//	  		CHECK (relative_error(freq, freq_ref) == Approx(0.0).epsilon(EPS));
//	  	}
//	  }
//	}
//

}
