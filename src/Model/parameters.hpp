// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __PARAMETERS_HPP_INCLUDED__
#define __PARAMETERS_HPP_INCLUDED__


#include "Io/io.hpp"
#include "Tools/setOnce.hpp"


///  Parameters: secure structure for the model parameters
//////////////////////////////////////////////////////////

class Parameters
{

  private:

      SetOnce <long> ncells_;       ///< number of cells
      SetOnce <long> nrays_;        ///< number of rays (originating from each cell)
      SetOnce <long> nrays_red_;    ///< number of rays reduced
      SetOnce <long> nboundary_;    ///< number of points on the boundary
      SetOnce <long> nfreqs_;       ///< number of frequency bins
      SetOnce <long> nfreqs_red_;   ///< number of frequency bins reduced
      SetOnce <long> nspecs_;       ///< number of chemical species
      SetOnce <long> nlspecs_;      ///< number of line producing species
      SetOnce <long> nlines_;       ///< number of line transitions
      SetOnce <long> nquads_;       ///< number of frequency quadrature points
      SetOnce <long> max_iter_;     ///< maximum number of iterations

      SetOnce <double> pop_prec_;     ///< required precision for populations



  public:

      inline void set_ncells     (const long value) {    ncells_.set (value);}
      inline void set_nrays      (const long value) {     nrays_.set (value);}
      inline void set_nrays_red  (const long value) { nrays_red_.set (value);}
      inline void set_nboundary  (const long value) { nboundary_.set (value);}
      inline void set_nfreqs     (const long value) {    nfreqs_.set (value);}
      inline void set_nfreqs_red (const long value) {nfreqs_red_.set (value);}
      inline void set_nspecs     (const long value) {    nspecs_.set (value);}
      inline void set_nlspecs    (const long value) {   nlspecs_.set (value);}
      inline void set_nlines     (const long value) {    nlines_.set (value);}
      inline void set_nquads     (const long value) {    nquads_.set (value);}
      inline void set_max_iter   (const long value) {  max_iter_.set (value);}

      inline void set_pop_prec   (const double value) {pop_prec_.set (value);}

      inline long ncells     (void) const {return     ncells_.get ();}
      inline long nrays      (void) const {return      nrays_.get ();}
      inline long nrays_red  (void) const {return  nrays_red_.get ();}
      inline long nboundary  (void) const {return  nboundary_.get ();}
      inline long nfreqs     (void) const {return     nfreqs_.get ();}
      inline long nfreqs_red (void) const {return nfreqs_red_.get ();}
      inline long nspecs     (void) const {return     nspecs_.get ();}
      inline long nlspecs    (void) const {return    nlspecs_.get ();}
      inline long nlines     (void) const {return     nlines_.get ();}
      inline long nquads     (void) const {return     nquads_.get ();}
      inline long max_iter   (void) const {return   max_iter_.get ();}

      inline double pop_prec (void) const {return   pop_prec_.get ();}

      long r;
      long o;
      long f;


      // Io
      int read (
          const Io &io);

      int write (
          const Io &io) const;

};


#endif // __PARAMETERS_HPP_INCLUDED__