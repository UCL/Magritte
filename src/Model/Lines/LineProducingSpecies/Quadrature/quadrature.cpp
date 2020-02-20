// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include "quadrature.hpp"
#include "Tools/logger.hpp"


const string Quadrature::prefix = "Lines/LineProducingSpecies_";


///  read: read in data structure
///    @param[in] io: io object
///    @param[in] parameters: model parameters object
/////////////////////////////////////////////////////

void Quadrature :: read (const Io &io, const int l, Parameters &parameters)
{
    cout << "Reading quadrature..." << endl;

    const string prefix_l = prefix + std::to_string (l) + "/Quadrature/";

    io.read_length (prefix_l+"weights", nquads);

    parameters.set_nquads (nquads);

    weights.resize (nquads);
    roots  .resize (nquads);

    io.read_list (prefix_l+"weights", weights);
    io.read_list (prefix_l+"roots",   roots  );
}




///  write: write out data structure
///    @param[in] io: io object
////////////////////////////////////

void Quadrature :: write (const Io &io, const int l) const
{
    cout << "Writing quadrature..." << endl;

    const string prefix_l = prefix + std::to_string (l) + "/Quadrature/";

    io.write_list (prefix_l+"weights", weights);
    io.write_list (prefix_l+"roots",   roots  );
}
