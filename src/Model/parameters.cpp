// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include "parameters.hpp"
#include "Tools/logger.hpp"


int Parameters ::
    read (
        const Io &io)
{

  write_to_log ("Reading parameters");


  long dummy;

  try {io.read_number (".ncells",     dummy); set_ncells     (dummy);} catch (...) { }
  try {io.read_number (".nrays",      dummy); set_nrays      (dummy);} catch (...) { }
  try {io.read_number (".nrays_red",  dummy); set_nrays_red  (dummy);} catch (...) { }
  try {io.read_number (".nboundary",  dummy); set_nboundary  (dummy);} catch (...) { }
  try {io.read_number (".nfreqs",     dummy); set_nfreqs     (dummy);} catch (...) { }
  try {io.read_number (".nfreqs_red", dummy); set_nfreqs_red (dummy);} catch (...) { }
  try {io.read_number (".nspecs",     dummy); set_nspecs     (dummy);} catch (...) { }
  try {io.read_number (".nlspecs",    dummy); set_nlspecs    (dummy);} catch (...) { }
  try {io.read_number (".nlines",     dummy); set_nlines     (dummy);} catch (...) { }
  try {io.read_number (".nquads",     dummy); set_nquads     (dummy);} catch (...) { }
  try {io.read_number (".max_iter",   dummy); set_max_iter   (dummy);} catch (...) { }

}


int Parameters ::
    write (
        const Io &io) const
{

  write_to_log ("Writing parameters");


  try {io.write_number (".ncells",     ncells     () );} catch (...) { }
  try {io.write_number (".nrays",      nrays      () );} catch (...) { }
  try {io.write_number (".nrays_red",  nrays_red  () );} catch (...) { }
  try {io.write_number (".nboundary",  nboundary  () );} catch (...) { }
  try {io.write_number (".nfreqs",     nfreqs     () );} catch (...) { }
  try {io.write_number (".nfreqs_red", nfreqs_red () );} catch (...) { }
  try {io.write_number (".nspecs",     nspecs     () );} catch (...) { }
  try {io.write_number (".nlspecs",    nlspecs    () );} catch (...) { }
  try {io.write_number (".nlines",     nlines     () );} catch (...) { }
  try {io.write_number (".nquads",     nquads     () );} catch (...) { }
  try {io.write_number (".max_iter",   max_iter   () );} catch (...) { }


}
