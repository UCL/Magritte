// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include "parameters.hpp"
#include "Tools/types.hpp"
#include "Tools/logger.hpp"


void Parameters :: read (const Io &io)
{
    cout << "Reading parameters..." << endl;


    size_t n;

    if   (io.read_number(".ncells",     n) == 0) {set_ncells    (n);}
    else                                         {cout << "Failed read ncells!"     << endl;}
    if   (io.read_number(".nrays",      n) == 0) {set_nrays     (n);}
    else                                         {cout << "Failed read nrays!"      << endl;}
    if   (io.read_number(".nrays_red",  n) == 0) {set_nrays_red (n);}
    else                                         {cout << "Failed read nrays_red!"  << endl;}
    if   (io.read_number(".order_min",  n) == 0) {set_order_min (n);}
    else                                         {cout << "Failed read order_min!"  << endl;}
    if   (io.read_number(".order_max",  n) == 0) {set_order_max (n);}
    else                                         {cout << "Failed read order_max!"  << endl;}
    if   (io.read_number(".nboundary",  n) == 0) {set_nboundary (n);}
    else                                         {cout << "Failed read nboundary!"  << endl;}
    if   (io.read_number(".nfreqs",     n) == 0) {set_nfreqs    (n);}
    else                                         {cout << "Failed read nfreqs!"     << endl;}
    if   (io.read_number(".nfreqs_red", n) == 0) {set_nfreqs_red(n);}
    else                                         {cout << "Failed read nfreqs_red!" << endl;}
    if   (io.read_number(".nspecs",     n) == 0) {set_nspecs    (n);}
    else                                         {cout << "Failed read nspecs!"     << endl;}
    if   (io.read_number(".nlspecs",    n) == 0) {set_nlspecs   (n);}
    else                                         {cout << "Failed read nlspecs!"    << endl;}
    if   (io.read_number(".nlines",     n) == 0) {set_nlines    (n);}
    else                                         {cout << "Failed read nlines!"     << endl;}
    if   (io.read_number(".nquads",     n) == 0) {set_nquads    (n);}
    else                                         {cout << "Failed read nquads!"     << endl;}


    double d;

    if   (io.read_number(".pop_prec", d) == 0) {set_pop_prec(d);}
    else                                       {cout << "Failed read pop_prec!" << endl;}


    bool b;

    if   (io.read_bool(".use_scattering",       b) == 0) {set_use_scattering      (b);}
    else                                                 {cout << "Failed read use_scattering!"       << endl;}
    if   (io.read_bool(".use_Ng_acceleration",  b) == 0) {set_use_Ng_acceleration (b);}
    else                                                 {cout << "Failed read use_Ng_acceleration!"  << endl;}
    if   (io.read_bool(".spherical_symmetry",   b) == 0) {set_spherical_symmetry  (b);}
    else                                                 {cout << "Failed read spherical_symmetry!"   << endl;}
    if   (io.read_bool(".adaptive_ray_tracing", b) == 0) {set_adaptive_ray_tracing(b);}
    else                                                 {cout << "Failed read adaptive_ray_tracing!" << endl;}
}




void Parameters :: write (const Io &io) const
{
    cout << "Writing parameters..." << endl;

    try         {io.write_number (".ncells",     ncells    () );}
    catch (...) {cout << "Failed write ncells!"         << endl;}
    try         {io.write_number (".nrays",      nrays     () );}
    catch (...) {cout << "Failed write nrays!"          << endl;}
//    try         {io.write_number (".nrays_red",  nrays_red () );}   // nrays_red depends on nprocs,
//    catch (...) {cout << "Failed write nrays_red!"      << endl;}   // so should be allowed to vary
    try         {io.write_number (".order_min",  order_min () );}
    catch (...) {cout << "Failed write order_min!"      << endl;}
    try         {io.write_number (".order_max",  order_max () );}
    catch (...) {cout << "Failed write order_max!"      << endl;}
    try         {io.write_number (".nboundary",  nboundary () );}
    catch (...) {cout << "Failed write nboundary!"      << endl;}
    try         {io.write_number (".nfreqs",     nfreqs    () );}
    catch (...) {cout << "Failed write nfreqs!"         << endl;}
    try         {io.write_number (".nfreqs_red", nfreqs_red() );}
    catch (...) {cout << "Failed write nfreqs_red!"     << endl;}
    try         {io.write_number (".nspecs",     nspecs    () );}
    catch (...) {cout << "Failed write nspecs!"         << endl;}
    try         {io.write_number (".nlspecs",    nlspecs   () );}
    catch (...) {cout << "Failed write nlspecs!"        << endl;}
    try         {io.write_number (".nlines",     nlines    () );}
    catch (...) {cout << "Failed write nlines!"         << endl;}
    try         {io.write_number (".nquads",     nquads    () );}
    catch (...) {cout << "Failed write nquads!"         << endl;}

    try         {io.write_number (".pop_prec", pop_prec() );}
    catch (...) {cout << "Failed write pop_prec!"    << endl;}

    try         {io.write_bool (".use_scattering",        use_scattering      () );}
    catch (...) {cout << "Failed write use_scattering!"                    << endl;}
    try         {io.write_bool (".use_Ng_acceleration",   use_Ng_acceleration () );}
    catch (...) {cout << "Failed write use_Ng_acceleration!"               << endl;}
    try         {io.write_bool (".spherical_symmetry",    spherical_symmetry  () );}
    catch (...) {cout << "Failed write spherical_symmetry!"                << endl;}
    try         {io.write_bool (".adaptive_ray_tracing",  adaptive_ray_tracing() );}
    catch (...) {cout << "Failed write adaptive_ray_tracing!"              << endl;}
}
