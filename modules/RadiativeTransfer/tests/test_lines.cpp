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

#include "lines.hpp"
#include "folders.hpp"
#include "Lines/src/linedata.hpp"
#include "GridTypes.hpp"

#define EPS 1.0E-7

//#define __VTURB_DEFINED__
//const double v_turb = 0;
//const double V_TURB_OVER_C_ALL_SQUARED = v_turb * v_turb / C_SQUARED;   // (v_turb / c)^2

#include "constants.hpp"

TEST_CASE ("Constructor")
{

  const long ncells = 1;

  const string linedata_folder = Magritte_folder + "tests/test_data/linedata/";

  LINEDATA linedata (linedata_folder);

  LINES lines (ncells, linedata);

}




TEST_CASE ("add_emissivity_and_opacity function")
{

  const long ncells = 3;
//  const long p      = 0;



  TEMPERATURE temperature (ncells);

  for (long p = 0; p < ncells; p++)
  {
    temperature.gas[p] = 100.0;
  }

  const string linedata_folder = Magritte_folder + "tests/test_data/linedata/";

  LINEDATA linedata (linedata_folder);


  LINES lines (ncells, linedata);


  for (long p = 0; p < ncells; p++)
  {
    for (int l = 0; l < lines.nlspec; l++)
    {
      for (int k = 0; k < lines.nrad[l]; k++)
      {
        const long ind = lines.index(p,l,k);

        lines.emissivity[ind] = 1.0;
        lines.opacity[ind] = 1.0;
      }
    }
  }


  FREQUENCIES frequencies (ncells, linedata);

  frequencies.reset (linedata, temperature);


  ofstream ETA ("/home/frederik/dumpster/ETA.txt");
  ofstream CHI ("/home/frederik/dumpster/CHI.txt");

  for (long p = 0; p < ncells; p++)
  {
    long lnotch = 0;
//	long p = 2;
    for (long f = 0; f < frequencies.nfreq_red/1; f++)
    {
      vReal freq_scaled  = 1.000002 * frequencies.nu[p][f];

      vReal eta = 0.0;
      vReal chi = 0.0;

      lines.add_emissivity_and_opacity (frequencies, temperature, freq_scaled,
                                        lnotch, p, eta, chi);

#     if (GRID_SIMD)
        for (int lane = 0; lane < n_simd_lanes; lane++)
        {
          ETA << eta.getlane(lane) << endl;
          CHI << chi.getlane(lane) << endl;
        }
#     else
        ETA << eta << endl;
        CHI << chi << endl;
#     endif
    }
  }
	
  ETA.close();
  CHI.close();

  CHECK (true);
}
