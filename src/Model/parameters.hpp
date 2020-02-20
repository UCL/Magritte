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

      SetOnce <size_t> ncells_;       ///< number of cells
      SetOnce <size_t> nrays_;        ///< number of rays (originating from each cell)
      SetOnce <size_t> nrays_red_;    ///< number of rays reduced
      SetOnce <size_t> nboundary_;    ///< number of points on the boundary
      SetOnce <size_t> nfreqs_;       ///< number of frequency bins
      SetOnce <size_t> nfreqs_red_;   ///< number of frequency bins reduced
      SetOnce <size_t> nspecs_;       ///< number of chemical species
      SetOnce <size_t> nlspecs_;      ///< number of line producing species
      SetOnce <size_t> nlines_;       ///< number of line transitions
      SetOnce <size_t> nquads_;       ///< number of frequency quadrature points

      SetOnce <double> pop_prec_;   ///< required precision for populations

      SetOnce <bool>   use_scattering_;        ///< true if scattering is used
      SetOnce <bool>   use_Ng_acceleration_;   ///< true if Ng accelera


  public:

      inline void set_ncells     (const size_t value) {    ncells_.set (value);}
      inline void set_nrays      (const size_t value) {     nrays_.set (value);}
      inline void set_nrays_red  (const size_t value) { nrays_red_.set (value);}
      inline void set_nboundary  (const size_t value) { nboundary_.set (value);}
      inline void set_nfreqs     (const size_t value) {    nfreqs_.set (value);}
      inline void set_nfreqs_red (const size_t value) {nfreqs_red_.set (value);}
      inline void set_nspecs     (const size_t value) {    nspecs_.set (value);}
      inline void set_nlspecs    (const size_t value) {   nlspecs_.set (value);}
      inline void set_nlines     (const size_t value) {    nlines_.set (value);}
      inline void set_nquads     (const size_t value) {    nquads_.set (value);}

      inline void set_pop_prec   (const double value) {pop_prec_.set (value);}

      inline void set_use_scattering      (const bool value) {use_scattering_     .set (value);}
      inline void set_use_Ng_acceleration (const bool value) {use_Ng_acceleration_.set (value);}

      inline size_t ncells     () const {return     ncells_.get ();}
      inline size_t nrays      () const {return      nrays_.get ();}
      inline size_t nrays_red  () const {return  nrays_red_.get ();}
      inline size_t nboundary  () const {return  nboundary_.get ();}
      inline size_t nfreqs     () const {return     nfreqs_.get ();}
      inline size_t nfreqs_red () const {return nfreqs_red_.get ();}
      inline size_t nspecs     () const {return     nspecs_.get ();}
      inline size_t nlspecs    () const {return    nlspecs_.get ();}
      inline size_t nlines     () const {return     nlines_.get ();}
      inline size_t nquads     () const {return     nquads_.get ();}

      inline double pop_prec () const {return   pop_prec_.get ();}

      inline bool use_scattering      () const {return use_scattering_     .get ();}
      inline bool use_Ng_acceleration () const {return use_Ng_acceleration_.get ();}


      long n_off_diag = 0;

      double max_width_fraction = 0.5;


      // Io
      void read  (const Io &io);
      void write (const Io &io) const;

};


#endif // __PARAMETERS_HPP_INCLUDED__
