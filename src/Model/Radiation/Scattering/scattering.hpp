// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __SCATTERING_HPP_INCLUDED__
#define __SCATTERING_HPP_INCLUDED__


#include "Tools/types.hpp"
#include "Tools/Parallel/wrap_Grid.hpp"


struct Scattering
{

  public:

	    Double1 opacity_scat;    ///< scattering opacity (p,f)


	    // Precalculate phase function for all frequencies

	    vReal3 phase;      ///< scattering phase function (r1,r2,f)


      // Io
      int read (
          const Io         &io,
                Parameters &parameters);

      int write (
          const Io &io) const;


      //inline int add_opacity (
      //    vReal& chi) const;

  private:

	    long nrays;         ///< number of rays
	    long nfreqs_scat;   ///< number of frequencies in scattering data
	    long nfreqs_red;    ///< number of frequencies

      static const string prefix;

};


#endif // __SCATTERING_HPP_INCLUDED__
