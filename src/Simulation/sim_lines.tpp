///  compute_LTE_level_populations: sets level populations to LTE values
////////////////////////////////////////////////////////////////////////

int Simulation ::
    compute_LTE_level_populations ()
{

  // Initialize levels, emissivities and opacities with LTE values
  lines.iteration_using_LTE (
      chemistry.species.abundance,
      thermodynamics.temperature.gas);


  return (0);

}




///  compute_level_populations: computes level populations self-consistenly with
///  the radiation field assuming statistical equilibrium
////////////////////////////////////////////////////////////////////////////////

int Simulation ::
    compute_level_populations (
        const Io   &io        )
{

  const bool use_Ng_acceleration = true;
  const long max_niterations     = 100;

  compute_level_populations_opts (io, use_Ng_acceleration, max_niterations);

  return (0);

}




///  compute_level_populations
//////////////////////////////

int Simulation ::
    compute_level_populations_opts (
        const Io   &io,
        const bool  use_Ng_acceleration,
        const long  max_niterations     )
{

  // Check spectral discretisation setting

  if (specDiscSetting != LineSet)
  {
    logger.write ("Error: Spectral discretisation was not set for Lines!");

    return (-1);
  }


  // Compute level populations

  // Write out initial level populations
  for (int l = 0; l < parameters.nlspecs(); l++)
  {
    const string tag = "_rank_" + str_MPI_comm_rank() + "_iteration_0";

    lines.lineProducingSpecies[l].write_populations (io, l, tag);
  }

  // Initialize the number of iterations
  long iteration        = 0;
  long iteration_normal = 0;

  // Initialize errors
  error_mean.clear ();
  error_max.clear ();

  // Initialize some_not_converged
  bool some_not_converged = true;


  // Iterate as long as some levels are not converged
  while (some_not_converged && (iteration < max_niterations))
  {

    iteration++;

    logger.write ("Starting iteration ", iteration);


    // Start assuming convergence
    some_not_converged = false;


    if (use_Ng_acceleration && (iteration_normal == 4))
    {
      lines.iteration_using_Ng_acceleration (
          parameters.pop_prec()             );

      iteration_normal = 0;
    }

    else
    {
      logger.write ("Computing the radiation field...");

      compute_radiation_field ();

      calc_Jeff ();

      lines.iteration_using_statistical_equilibrium (
          chemistry.species.abundance,
          thermodynamics.temperature.gas,
          parameters.pop_prec()                     );

      iteration_normal++;
    }


    for (int l = 0; l < parameters.nlspecs(); l++)
    {
      error_mean.push_back (lines.lineProducingSpecies[l].relative_change_mean);
      error_max.push_back  (lines.lineProducingSpecies[l].relative_change_max);

      if (lines.lineProducingSpecies[l].fraction_not_converged > 0.005)
      {
        some_not_converged = true;
      }

      const double fnc = lines.lineProducingSpecies[l].fraction_not_converged;

      logger.write ("Already ", 100 * (1.0 - fnc), " % converged!");

      const string tag = "_rank_" + str_MPI_comm_rank() + "_iteration_" + to_string (iteration);

      lines.lineProducingSpecies[l].write_populations (io, l, tag);
    }


  } // end of while loop of iterations


  // Print convergence stats
  logger.write ("Converged after ", iteration, " iterations");


  return (0);

}






///  calc_J_eff: calculate the effective mean intensity in a line
///    @param[in] p: number of the cell under consideration
///    @param[in] l: number of the line producing species under consideration
/////////////////////////////////////////////////////////////////////////////

void Simulation ::
    calc_Jeff ()
{

  for (LineProducingSpecies &lspec : lines.lineProducingSpecies)
  {
    OMP_PARALLEL_FOR (p, parameters.ncells())
    {
      for (int k = 0; k < lspec.linedata.nrad; k++)
      {
        const Long1 freq_nrs = lspec.nr_line[p][k];

        lspec.Jeff[p][k] = 0.0;
        lspec.Jlin[p][k] = 0.0;


        // Integrate over the line

        for (long z = 0; z < parameters.nquads(); z++)
        {
          const double JJ = radiation.get_J (p,freq_nrs[z]);

          lspec.Jeff[p][k] += lspec.quadrature.weights[z] * JJ;
          lspec.Jlin[p][k] += lspec.quadrature.weights[z] * JJ;
        }


        // Subtract the approximated part

        for (long m = 0; m < lspec.lambda.get_size(p,k); m++)
        {
          const long I = lspec.index (lspec.lambda.get_nr(p,k,m), lspec.linedata.irad[k]);

          lspec.Jeff[p][k] -= HH_OVER_FOUR_PI * lspec.lambda.get_Ls(p,k,m) * lspec.population[I];
        }
      }
    }
  }


}
