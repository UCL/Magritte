// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <iostream>
using std::cout;
using std::endl;

#include "catch.hpp"

#include "Io/cpp/io_cpp_text.hpp"
#include "Simulation/simulation.hpp"

#define EPS 1.0E-15

// Allow access to private variables
#define private public


TEST_CASE ("Simulation :: get_eta_and_chi")
{

  // Read a test model file

  const string name = "/home/frederik/Dropbox/GitHub/Magritte-code/Benchmarks/0_analytical_models/models/model_1_1D_velocity_gradient/";

  IoText io (name);

  Simulation simulation;

  simulation.parameters.set_pop_prec            (1.0E-6);
  simulation.parameters.set_use_scattering      (false);
  simulation.parameters.n_off_diag = 0;

  simulation.read (io);

  simulation.compute_spectral_discretisation ();
  simulation.compute_boundary_intensities    ();
  simulation.compute_LTE_level_populations   ();


  // Print all frequencies

  for (long f = 0; f < simulation.parameters.nfreqs_red(); f++)
  {
    cout << simulation.radiation.frequencies.nu[0][f] << endl;
  }


  vReal freq_scaled = simulation.lines.line[1];

  long p      = 0;
  long lnotch = 0;

  vReal eta;
  vReal chi;

  freq_scaled = simulation.lines.line[1];
  simulation.get_eta_and_chi (freq_scaled, p, lnotch, eta, chi);
  freq_scaled = simulation.lines.line[1];
  simulation.get_eta_and_chi (freq_scaled, p, lnotch, eta, chi);
  freq_scaled = simulation.lines.line[0];
  simulation.get_eta_and_chi (freq_scaled, p, lnotch, eta, chi);

  cout << "Final lnotch = " << lnotch << endl;

  CHECK(true);

}
