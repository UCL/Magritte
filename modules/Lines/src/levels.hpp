// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __LEVELS_HPP_INCLUDED__
#define __LEVELS_HPP_INCLUDED__


#include <string>
#include <vector>
using namespace std;
#include <Eigen/Core>
using namespace Eigen;

#include "linedata.hpp"
#include "RadiativeTransfer/src/types.hpp"
#include "RadiativeTransfer/src/cells.hpp"
#include "RadiativeTransfer/src/lines.hpp"
#include "RadiativeTransfer/src/temperature.hpp"
#include "RadiativeTransfer/src/frequencies.hpp"
#include "RadiativeTransfer/src/radiation.hpp"
#include "RadiativeTransfer/src/scattering.hpp"
//#include "RadiativeTransfer/src/RadiativeTransfer.hpp"

#define MAX_NITERATIONS 100


struct LEVELS
{
	
  const long ncells;                ///< number of cells
	
  const int nlspec;                 ///< number of species producing lines

  const Int1 nlev;                  ///< number of levels per species
  const Int1 nrad;                  ///< number of radiative transitions per species


  bool some_not_converged;          ///< true when there are unconverged species

  Bool1            not_converged;   ///< true when species is not converged
  Double1 fraction_not_converged;   ///< fraction of levels that is not converged

  VectorXd2 population;             ///< level population (most recent)

  Double3 J_eff;                    ///< effective mean intensity

  Double2   population_tot;         ///< total level population (sum over levels)

  VectorXd2 population_prev1;       ///< level populations 1 iteration back
  VectorXd2 population_prev2;       ///< level populations 2 iterations back
  VectorXd2 population_prev3;       ///< level populations 3 iterations back


  LEVELS                           (
      const long      num_of_cells,
      const LINEDATA &linedata     );   ///< Constructor


  int iteration_using_LTE            (
      const LINEDATA    &linedata,
      const SPECIES     &species,
      const TEMPERATURE &temperature,
            LINES       &lines       );


  int update_using_LTE               (
      const LINEDATA    &linedata,
      const SPECIES     &species,
      const TEMPERATURE &temperature,
      const long         p,
      const int          l           );


  int update_using_Ng_acceleration ();


  int iteration_using_statistical_equilibrium (
      const LINEDATA    &linedata,
      const SPECIES     &species,
      const TEMPERATURE &temperature,
      const FREQUENCIES &frequencies,
      const RADIATION   &radiation,
            LINES       &lines                );


  int update_using_statistical_equilibrium (
      const MatrixXd &R,
      const long      p,
      const int       l                    );


  // Communication with Radiative Transfer module
	
  int calc_line_emissivity_and_opacity (
      const LINEDATA &linedata,
            LINES    &lines,
      const long      p,
      const int       l                ) const;

  int calc_J_eff                     (
      const FREQUENCIES &frequencies,
      const TEMPERATURE &temperature,
      const RADIATION   &radiation,
      const long         p,
      const int          l           );

  // Convergence

  int check_for_convergence (
      const long p,
      const int  l          );


  // Print
	
  int print (
      const string tag) const;


  template <int Dimension, long Nrays>
  int compute_all                                (
      const CELLS<Dimension, Nrays> &cells,
      const LINEDATA                &linedata,
      const SPECIES                 &species,
      const TEMPERATURE             &temperature,
      const FREQUENCIES             &frequencies,
            RADIATION               &radiation   );

};




template <int Dimension, long Nrays>
int LEVELS ::
    compute_all (
        const CELLS<Dimension, Nrays> &cells,
        const LINEDATA                &linedata,
        const SPECIES                 &species,
        const TEMPERATURE             &temperature,
        const FREQUENCIES             &frequencies,
              RADIATION               &radiation   )
{

  LINES lines (cells.ncells, linedata);

  const long nfreq_scat = 1;
  SCATTERING scattering (Nrays, nfreq_scat, frequencies.nfreq_red);


  // Initialize levels, emissivities and opacities with LTE values

  iteration_using_LTE (linedata, species, temperature, lines);

  const string tag_0 = "_0";

  lines.print (tag_0);
  print (tag_0);


  // Initialize the number of iterations

  int niterations = 0;   


  // Iterate as long as some levels are not converged

  while (some_not_converged)
  {
    niterations++;


    // Print number of current iteration

    cout << "Starting iteration ";
    cout << niterations << endl;


    // Perform an Ng acceleration step every 4th iteration

    if (niterations%4 == 0)
    {
      update_using_Ng_acceleration ();
    }


    // Get radiation field from Radiative Transfer

    //MPI_TIMER timer_RT ("RT");
    //timer_RT.start ();

    radiation.compute_mean_intensity (cells, temperature, frequencies, lines, scattering);

    //RadiativeTransfer<Dimension, Nrays>(
    //    cells,
    //    temperature,
    //    frequencies,
    //    lines,
    //    scattering,
    //    radiation   );

    //timer_RT.stop ();
    //timer_RT.print_to_file ();


    for (int l = 0; l < nlspec; l++)
    {
      fraction_not_converged[l] = 0.0;
    }


    iteration_using_statistical_equilibrium (
        linedata,
        species,
        temperature,
        frequencies,
        radiation,
        lines                               );


    const string tag_n = "_" + to_string (niterations);

    radiation.print(tag_n);
    lines.print (tag_n);
    print (tag_n);


    // Allow 1% to be not converged

    for (int l = 0; l < nlspec; l++)
    {
      if (fraction_not_converged[l] < 0.01)
      {
        not_converged[l] = false;
      }
    }


    // If some are not converged

    some_not_converged = false;

    for (int l = 0; l < nlspec; l++)
    {
      if (not_converged[l])
      {
        some_not_converged = true;
      }
    }


    // Limit the number of iteration

    if (niterations >= MAX_NITERATIONS)
    {
      some_not_converged = false;
    }


    // Print status of convergence

    for (int l = 0; l < nlspec; l++)
    {
      cout << "fraction_not_converged = ";
      cout << fraction_not_converged[l] << endl;
    }


  } // end of while loop of iterations



  // Print convergence stats

  cout << "Converged after ";
  cout << niterations;
  cout << " iterations" << endl;


  return (0);
}


#endif // __LEVELS_HPP_INCLUDED__
