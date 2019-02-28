// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include "lineProducingSpecies.hpp"


const string LineProducingSpecies::prefix = "Lines/LineProducingSpecies_";


///  read: read in data structure
///    @param[in] io: io object
///    @param[in] l: nr of line producing species
///    @paran[in] parameters: model parameters object
/////////////////////////////////////////////////////

int LineProducingSpecies ::
    read (
        const Io         &io,
        const long        l,
              Parameters &parameters)
{

  linedata.read (io, l);


  fraction_not_converged = 0.0;
           not_converged = true;


  ncells = parameters.ncells ();


  population_prev1.resize (ncells*linedata.nlev);
  population_prev2.resize (ncells*linedata.nlev);
  population_prev3.resize (ncells*linedata.nlev);
    population_tot.resize (ncells*linedata.nlev);
        population.resize (ncells*linedata.nlev);


  J_line.resize (ncells);
  J_star.resize (ncells);

  for (long p = 0; p < ncells; p++)
  {
    J_line[p].resize (linedata.nrad);
    J_star[p].resize (linedata.nrad);
  }


  return (0);

}




///  write: write out data structure
///    @param[in] io: io object
///    @param[in] l: nr of line producing species
/////////////////////////////////////////////////

int LineProducingSpecies ::
    write (
        const Io  &io,
        const long l  ) const
{

  linedata.write (io, l);


  const string prefix_l = prefix + std::to_string (l) + "/";


  Double2 pops (ncells, Double1 (linedata.nlev));

  OMP_PARALLEL_FOR (p, ncells)
  {
    for (long i = 0; i < linedata.nlev; i++)
    {
      pops[p][i] = population (index (p, i));
    }
  }

  io.write_array (prefix_l+"population", pops);



  return (0);

}
