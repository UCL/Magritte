// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include "model.hpp"
#include "Io/io.hpp"
#include "Tools/logger.hpp"
#include "Tools/Parallel/wrap_mpi.hpp"


///  Constructor for Model
//////////////////////////

Model :: Model ()
{
    cout << "Constructing" << endl;

    logger.write("Constructing model.");
#   if (MPI_PARALLEL)
        if (!MPI_initialized())
        {
            MPI_Init (NULL, NULL);
            logger.write("MPI initialized.");
        }
#   endif
}




///  Destructor for Model
/////////////////////////

//Model :: ~Model ()
//{
//#   if (MPI_PARALLEL)
//        if (MPI_initialized())
//        {
//            MPI_Finalize ();
//            logger.write("MPI finalized.");
//        }
//#   endif
//}




///  Reader for the Model data
///    @param[in] io : io object to read with
/////////////////////////////////////////////

void Model :: read (const Io &io)
{
    logger.write ("                                           ");
    logger.write ("-------------------------------------------");
    logger.write ("  Reading Model...                         ");
    logger.write ("-------------------------------------------");
    logger.write (" model file = " + io.io_file                );
    logger.write ("-------------------------------------------");

    parameters    .read (io);
    geometry      .read (io, parameters);
    thermodynamics.read (io, parameters);
    chemistry     .read (io, parameters);
    lines         .read (io, parameters);
    radiation     .read (io, parameters);

    logger.write ("                                           ");
    logger.write ("-------------------------------------------");
    logger.write ("  Model read, parameters:                  ");
    logger.write ("-------------------------------------------");
    logger.write ("  ncells     = ", parameters.ncells     ()  );
    logger.write ("  nrays      = ", parameters.nrays      ()  );
    logger.write ("  nrays_red  = ", parameters.nrays_red  ()  );
    logger.write ("  nboundary  = ", parameters.nboundary  ()  );
    logger.write ("  nfreqs     = ", parameters.nfreqs     ()  );
    logger.write ("  nfreqs_red = ", parameters.nfreqs_red ()  );
    logger.write ("  nspecs     = ", parameters.nspecs     ()  );
    logger.write ("  nlspecs    = ", parameters.nlspecs    ()  );
    logger.write ("  nlines     = ", parameters.nlines     ()  );
    logger.write ("  nquads     = ", parameters.nquads     ()  );
    logger.write ("-------------------------------------------");
    logger.write ("                                           ");
}




///  Writer for the Model data
///    @param[in] io : io object to write with
//////////////////////////////////////////////

void Model :: write (const Io &io)
{
    if (MPI_comm_rank () == 0)
    {
        logger.write ("                                           ");
        logger.write ("-------------------------------------------");
        logger.write ("  Writing Model...                         ");
        logger.write ("-------------------------------------------");
        logger.write ("                                           ");

        parameters    .write (io);
        geometry      .write (io);
        thermodynamics.write (io);
        chemistry     .write (io);
        lines         .write (io);
        radiation     .write (io);

        logger.write ("                                           ");
        logger.write ("-------------------------------------------");
        logger.write ("  Model written.                           ");
        logger.write ("-------------------------------------------");
        logger.write ("                                           ");
    }
}
