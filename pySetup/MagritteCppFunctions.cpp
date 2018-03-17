// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <boost/python.hpp>
#include "data_tools.cpp"


BOOST_PYTHON_MODULE (MagritteCppFunctions)
{
    using namespace boost::python;
    // From data_tools.cpp
    def("getNCELLSvtu", get_NCELLS_vtu);
}
