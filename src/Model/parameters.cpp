// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include "parameters.hpp"
#include "Tools/types.hpp"
#include "Tools/logger.hpp"


int Parameters ::
    read (
        const Io &io)
{

  cout << "Reading parameters" << endl;


  long n;

  try {io.read_number (".ncells",     n); set_ncells     (n);} catch (...) { }
  try {io.read_number (".ncameras",   n); set_ncameras   (n);} catch (...) { }
  try {io.read_number (".nrays",      n); set_nrays      (n);} catch (...) { }
  try {io.read_number (".nrays_red",  n); set_nrays_red  (n);} catch (...) { }
  try {io.read_number (".nboundary",  n); set_nboundary  (n);} catch (...) { }
  try {io.read_number (".nfreqs",     n); set_nfreqs     (n);} catch (...) { }
  try {io.read_number (".nfreqs_red", n); set_nfreqs_red (n);} catch (...) { }
  try {io.read_number (".nspecs",     n); set_nspecs     (n);} catch (...) { }
  try {io.read_number (".nlspecs",    n); set_nlspecs    (n);} catch (...) { }
  try {io.read_number (".nlines",     n); set_nlines     (n);} catch (...) { }
  try {io.read_number (".nquads",     n); set_nquads     (n);} catch (...) { }


  double d;

  try {io.read_number (".pop_prec", d); set_pop_prec (d);} catch (...) { }


  bool b;

  try {io.read_bool (".use_scattering",      b); set_use_scattering      (b);} catch (...) { }
  try {io.read_bool (".use_Ng_acceleration", b); set_use_Ng_acceleration (b);} catch (...) { }


  return (0);

}


int Parameters ::
    write (
        const Io &io) const
{

  cout << "Writing parameters" << endl;


  try {io.write_number (".ncells",     ncells     () );} catch (...) { }
  try {io.write_number (".ncameras",   ncameras   () );} catch (...) { }
  try {io.write_number (".nrays",      nrays      () );} catch (...) { }
  try {io.write_number (".nrays_red",  nrays_red  () );} catch (...) { }
  try {io.write_number (".nboundary",  nboundary  () );} catch (...) { }
  try {io.write_number (".nfreqs",     nfreqs     () );} catch (...) { }
  try {io.write_number (".nfreqs_red", nfreqs_red () );} catch (...) { }
  try {io.write_number (".nspecs",     nspecs     () );} catch (...) { }
  try {io.write_number (".nlspecs",    nlspecs    () );} catch (...) { }
  try {io.write_number (".nlines",     nlines     () );} catch (...) { }
  try {io.write_number (".nquads",     nquads     () );} catch (...) { }

  try {io.write_number (".pop_prec", pop_prec () );} catch (...) { }

  try {io.write_bool (".use_scattering",      use_scattering      () );} catch (...) { }
  try {io.write_bool (".use_Ng_acceleration", use_Ng_acceleration () );} catch (...) { }


  return (0);

}
