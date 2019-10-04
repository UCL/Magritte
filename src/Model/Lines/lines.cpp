// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include "lines.hpp"
#include "Tools/Parallel/wrap_omp.hpp"
#include "Tools/Parallel/wrap_mpi.hpp"
#include "Tools/logger.hpp"
#include "Functions/heapsort.hpp"


const string Lines::prefix = "Lines/";


///  Reader for the Lines data
///    @param[in] io         : io object to read with
///    @param[in] parameters : model parameters object
//////////////////////////////////////////////////////

int Lines ::
    read (
        const Io         &io,
              Parameters &parameters)
{

  cout << "Reading lines" << endl;


  // Data

  io.read_length (prefix+"LineProducingSpecies_", nlspecs);


  parameters.set_nlspecs (nlspecs);


  lineProducingSpecies.resize (nlspecs);

  for (int l = 0; l < nlspecs; l++)
  {
    lineProducingSpecies[l].read (io, l, parameters);
  }


  nrad_cum.resize (nlspecs, 0);

  for (int l = 1; l < nlspecs; l++)
  {
    nrad_cum[l] = nrad_cum[l-1] + lineProducingSpecies[l-1].linedata.nrad;
  }




  // Lines

  nlines = 0;

  for (int l = 0; l < nlspecs; l++)
  {
    nlines += lineProducingSpecies[l].linedata.nrad;
  }


  parameters.set_nlines (nlines);


  line      .resize (nlines);
  line_index.resize (nlines);

  long index = 0;

  for (int l = 0; l < nlspecs; l++)
  {
    const Linedata linedata = lineProducingSpecies[l].linedata;

    for (int k = 0; k < linedata.nrad; k++)
    {
      line      [index] = linedata.frequency[k];
      line_index[index] = index;
      index++;
    }
  }

  // Sort line frequencies
  heapsort (line, line_index);


  ncells = parameters.ncells();


  emissivity.resize (ncells*nlines);
     opacity.resize (ncells*nlines);


  return (0);

}




///  Writer for the Lines data
///    @param[in] io: io object to write with
/////////////////////////////////////////////

int Lines ::
    write (
        const Io &io) const
{

  cout << "Writing lines" << endl;


  for (int l = 0; l < lineProducingSpecies.size(); l++)
  {
    lineProducingSpecies[l].write (io, l);
  }


  return (0);

}




int Lines ::
    iteration_using_LTE (
        const Double2 &abundance,
        const Double1 &temperature)
{

  for (LineProducingSpecies &lspec : lineProducingSpecies)
  {
    lspec.update_using_LTE (abundance, temperature);
  }


  set_emissivity_and_opacity ();

  //gather_emissivities_and_opacities ();


  return (0);

}



int Lines ::
    iteration_using_Ng_acceleration (
        const double pop_prec       )
{

  for (LineProducingSpecies &lspec : lineProducingSpecies)
  {
    lspec.update_using_Ng_acceleration ();

    lspec.check_for_convergence (pop_prec);
  }


  set_emissivity_and_opacity ();

  //gather_emissivities_and_opacities ();


  return (0);

}




int Lines ::
    iteration_using_statistical_equilibrium (
        const Double2 &abundance,
        const Double1 &temperature,
        const double   pop_prec             )
{

  for (LineProducingSpecies &lspec : lineProducingSpecies)
  {
    lspec.update_using_statistical_equilibrium (abundance, temperature);

    lspec.check_for_convergence (pop_prec);
  }


  set_emissivity_and_opacity ();

  //gather_emissivities_and_opacities ();


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


  for (int w = 0; w < comm_size; w++)
  {
    long start = ( w   *ncells)/comm_size;
    long stop  = ((w+1)*ncells)/comm_size;

    long ncells_red_w = stop - start;

    buffer_lengths[w] = ncells_red_w * nlines;
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
