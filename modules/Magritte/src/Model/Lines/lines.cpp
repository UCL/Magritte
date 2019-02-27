// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <iostream>

#include "lines.hpp"
#include "Tools/Parallel/wrap_omp.hpp"
#include "Functions/heapsort.hpp"


const string Lines::prefix = "Lines/";


///  read: read in data structure
///    @param[in] io: io object
///    @paran[in] parameters: model parameters object
/////////////////////////////////////////////////////

int Lines ::
    read (
        const Io         &io,
              Parameters &parameters)
{

  // Data

  io.read_length (prefix+"Linedata", nlspecs);


  parameters.set_nlspecs (nlspecs);


  linedata.resize (nlspecs);

  for (int l = 0; l < nlspecs; l++)
  {
    linedata[l].read (io, l);
  }


  nrad_cum.resize (nlspecs, 0);

  for (int l = 1; l < nlspecs; l++)
  {
    nrad_cum[l] = nrad_cum[l-1] + linedata[l-1].nrad;
  }


  io.read_length (prefix+"quadrature_weights", nquads);


  parameters.set_nquads (nquads);


  quadrature_weights.resize (nquads);
  quadrature_roots.resize   (nquads);

  io.read_list (prefix+"quadrature_weights", quadrature_weights);
  io.read_list (prefix+"quadrature_roots",   quadrature_roots  );


  // Lines

  nlines = 0;

  for (int l = 0; l < nlspecs; l++)
  {
    nlines += linedata[l].nrad;
  }


  parameters.set_nlines (nlines);


  line      .resize (nlines);
  line_index.resize (nlines);

  long index = 0;

  for (int l = 0; l < nlspecs; l++)
  {
    for (int k = 0; k < linedata[l].nrad; k++)
    {
      line      [index] = linedata[l].frequency[k];
      line_index[index] = index;
      index++;
    }
  }

  // Sort line frequencies
  heapsort (line, line_index);


  ncells = parameters.ncells();


  nr_line.resize (ncells);

  for (long p = 0; p < ncells; p++)
  {
    nr_line[p].resize (nlspecs);

    for (int l = 0; l < nlspecs; l++)
    {
      nr_line[p][l].resize (linedata[l].nrad);

      for (int k = 0; k < linedata[l].nrad; k++)
      {
        nr_line[p][l][k].resize (nquads);
      }
    }
  }


  emissivity.resize (ncells*nlines);
     opacity.resize (ncells*nlines);


  // Levels

  fraction_not_converged.resize (nlspecs);
           not_converged.resize (nlspecs);


  for (int l = 0; l < nlspecs; l++)
  {
    fraction_not_converged[l] = 0.0;
             not_converged[l] = true;
  }


  population_prev1.resize (ncells);
  population_prev2.resize (ncells);
  population_prev3.resize (ncells);
    population_tot.resize (ncells);
        population.resize (ncells);
            J_line.resize (ncells);
            J_star.resize (ncells);


  for (long p = 0; p < ncells; p++)
  {
    population_prev1[p].resize (nlspecs);
    population_prev2[p].resize (nlspecs);
    population_prev3[p].resize (nlspecs);
      population_tot[p].resize (nlspecs);
          population[p].resize (nlspecs);
              J_line[p].resize (nlspecs);
              J_star[p].resize (nlspecs);


    for (int l = 0; l < nlspecs; l++)
    {
      population_prev1[p][l].resize (linedata[l].nlev);
      population_prev2[p][l].resize (linedata[l].nlev);
      population_prev3[p][l].resize (linedata[l].nlev);
            population[p][l].resize (linedata[l].nlev);
                J_line[p][l].resize (linedata[l].nrad);
                J_star[p][l].resize (linedata[l].nrad);
    }
  }


  return (0);

}




///  write: write out data structure
///    @param[in] io: io object
////////////////////////////////////

int Lines ::
    write (
        const Io &io) const
{

  for (int l = 0; l < linedata.size(); l++)
  {
    linedata[l].write (io, l);


    Double2 pops (ncells, Double1 (linedata[l].nlev));

    OMP_PARALLEL_FOR (p, ncells)
    {
      for (long i = 0; i < linedata[l].nlev; i++)
      {
        pops[p][i] = population[p][l](i);
      }
    }

    const string name = prefix + "population_" + std::to_string (l);

    io.write_array (name, pops);
  }

  io.write_list (prefix+"quadrature_weights", quadrature_weights);
  io.write_list (prefix+"quadrature_roots",   quadrature_roots  );


  return (0);

}




int Lines ::
    gather_emissivities_and_opacities ()

#if (MPI_PARALLEL)

{

  // Get number of processes
  const int comm_size = MPI_comm_size ();


  // Extract the buffer lengths and displacements

  int *buffer_lengths = new int[comm_size];
  int *displacements  = new int[comm_size];


  for (int w = 0; w < world_size; w++)
  {
    long start = ( w   *ncells)/comm_size;
    long stop  = ((w+1)*ncells)/comm_size;

    long ncells_red_w = stop - start;

    buffer_lengths[w] = ncells_red_w * nrad_tot;
  }

  displacements[0] = 0;

  for (int w = 1; w < comm_size; w++)
  {
    displacements[w] = buffer_lengths[w-1];
  }


  // Call MPI to gather the emissivity data

  int ierr_em =	MPI_Allgatherv (
                  MPI_IN_PLACE,            // pointer to data to be send (here in place)
                  0,                       // number of elements in the send buffer
                  MPI_DATATYPE_NULL,       // type of the send data
                  emissivity.data(),       // pointer to the data to be received
                  buffer_lengths,          // number of elements in receive buffer
                  displacements,           // displacements between data blocks
	                MPI_DOUBLE,              // type of the received data
                  MPI_COMM_WORLD);

  assert (ierr_em == 0);


  // Call MPI to gather the opacity data

  int ierr_op = MPI_Allgatherv (
                  MPI_IN_PLACE,            // pointer to data to be send (here in place)
                  0,                       // number of elements in the send buffer
                  MPI_DATATYPE_NULL,       // type of the send data
              	  opacity.data(),          // pointer to the data to be received
              	  buffer_lengths,          // number of elements in receive buffer
                  displacements,           // displacements between data blocks
              	  MPI_DOUBLE,              // type of the received data
                  MPI_COMM_WORLD);

  assert (ierr_op == 0);


  delete [] buffer_lengths;
  delete [] displacements;


  return (0);

}

#else

{

  return (0);

}

#endif
