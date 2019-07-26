// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include "lineProducingSpecies.hpp"
#include "Tools/logger.hpp"
#include "Tools/debug.hpp"


const string LineProducingSpecies::prefix = "Lines/LineProducingSpecies_";


///  Reader for the LineProducingSpecies data from the Io object
///    @param[in] io         : io object
///    @param[in] l          : nr of line producing species
///    @param[in] parameters : model parameters object
////////////////////////////////////////////////////////////////

int LineProducingSpecies ::
    read (
        const Io         &io,
        const long        l,
              Parameters &parameters)
{

  cout << "Reading lineProducingSpecies" << endl;


  linedata.read (io, l);

  quadrature.read (io, l, parameters);


  ncells = parameters.ncells ();
  nquads = parameters.nquads();


  lambda.resize (ncells);
    Jeff.resize (ncells);
    Jlin.resize (ncells);

  for (long p = 0; p < ncells; p++)
  {
    lambda[p].resize (linedata.nrad);
      Jeff[p].resize (linedata.nrad);
      Jlin[p].resize (linedata.nrad);
  }


  nr_line.resize (ncells);

  for (long p = 0; p < ncells; p++)
  {
    nr_line[p].resize (linedata.nrad);

    for (int k = 0; k < linedata.nrad; k++)
    {
      nr_line[p][k].resize (nquads);
    }
  }


  population_prev1.resize (ncells*linedata.nlev);
  population_prev2.resize (ncells*linedata.nlev);
  population_prev3.resize (ncells*linedata.nlev);
    population_tot.resize (ncells*linedata.nlev);
        population.resize (ncells*linedata.nlev);


  const string prefix_l = prefix + std::to_string (l) + "/";


  io.read_list (prefix_l+"population_tot", population_tot);


  read_populations (io, l, "");


  Double2 pops_prev1 (ncells, Double1 (linedata.nlev));
  Double2 pops_prev2 (ncells, Double1 (linedata.nlev));
  Double2 pops_prev3 (ncells, Double1 (linedata.nlev));

  int err_prev1 = io.read_array (prefix_l+"population_prev1", pops_prev1);
  int err_prev2 = io.read_array (prefix_l+"population_prev2", pops_prev2);
  int err_prev3 = io.read_array (prefix_l+"population_prev3", pops_prev3);


  OMP_PARALLEL_FOR (p, ncells)
  {
    for (long i = 0; i < linedata.nlev; i++)
    {
      if (err_prev1 == 0) {population_prev1 (index (p, i)) = pops_prev1[p][i];}
      if (err_prev2 == 0) {population_prev2 (index (p, i)) = pops_prev2[p][i];}
      if (err_prev3 == 0) {population_prev3 (index (p, i)) = pops_prev3[p][i];}
    }
  }


  return (0);

}




///  Writer for the LineProducingSpecies data to the Io object
///    @param[in] io : io object
///    @param[in] l  : nr of line producing species
//////////////////////////////////////////////////////////////

int LineProducingSpecies ::
    write (
        const Io  &io,
        const long l  ) const
{

  cout << "Writing lineProducingSpecies" << endl;


  linedata.write (io, l);

  quadrature.write (io, l);


  write_populations (io, l, "");


  const string prefix_l = prefix + std::to_string (l) + "/";


  io.write_list (prefix_l+"population_tot", population_tot);


  Double2 pops_prev1 (ncells, Double1 (linedata.nlev));
  Double2 pops_prev2 (ncells, Double1 (linedata.nlev));
  Double2 pops_prev3 (ncells, Double1 (linedata.nlev));


  OMP_PARALLEL_FOR (p, ncells)
  {
    for (long i = 0; i < linedata.nlev; i++)
    {
      pops_prev1[p][i] = population_prev1 (index (p, i));
      pops_prev2[p][i] = population_prev2 (index (p, i));
      pops_prev3[p][i] = population_prev3 (index (p, i));
    }
  }

  io.write_array (prefix_l+"population_prev1", pops_prev1);
  io.write_array (prefix_l+"population_prev2", pops_prev2);
  io.write_array (prefix_l+"population_prev3", pops_prev3);


  return (0);

}




///  Reader for the level populations from the Io object
///    @param[in] io  : io object
///    @param[in] l   : number of line producing species
///    @param[in] tag : extra info tag
////////////////////////////////////////////////////////

int LineProducingSpecies ::
    read_populations (
        const Io     &io,
        const long    l,
        const string  tag)
{

  const string prefix_l = prefix + std::to_string (l) + "/";

  Double2 pops (ncells, Double1 (linedata.nlev));

  int err = io.read_array (prefix_l+"population"+tag, pops);


  if (err == 0)
  {
    OMP_PARALLEL_FOR (p, ncells)
    {
      for (long i = 0; i < linedata.nlev; i++)
      {
        population (index (p, i)) = pops[p][i];
      }
    }
  }

  //io.read_array (prefix_l+"Jlin"+tag, Jlin);
  //io.read_array (prefix_l+"Jeff"+tag, Jeff);


  return (0);

}




///  Writer for the level populations to the Io object
///    @param[in] io  : io object
///    @param[in] l   : number of line producing species
///    @param[in] tag : extra info tag
////////////////////////////////////////////////////////

int LineProducingSpecies ::
    write_populations (
        const Io     &io,
        const long    l,
        const string  tag) const
{

  const string prefix_l = prefix + std::to_string (l) + "/";

  Double2 pops (ncells, Double1 (linedata.nlev));


  OMP_PARALLEL_FOR (p, ncells)
  {
    for (long i = 0; i < linedata.nlev; i++)
    {
      pops[p][i] = population (index (p, i));
    }
  }

  io.write_array (prefix_l+"population"+tag, pops);

  //io.write_array (prefix_l+"Jlin"+tag, Jlin);
  //io.write_array (prefix_l+"Jeff"+tag, Jeff);


  return (0);

}




///  Initializer for the Lambda operator
////////////////////////////////////////

int LineProducingSpecies ::
    initialize_Lambda ()
{

  OMP_PARALLEL_FOR (p, ncells)
  {
    for (long k = 0; k < linedata.nrad; k++)
    {
      lambda[p][k].Ls.clear();
      lambda[p][k].nr.clear();
    }
  }


  return (0);

}




///  Gatherer for the Lambda's from the MPI distributed processes
/////////////////////////////////////////////////////////////////

int LineProducingSpecies ::
    gather_Lambda ()
{



  return (0);

}
