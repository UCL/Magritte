// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include "scattering.hpp"
#include "Tools/Parallel/wrap_Grid.hpp"
#include "Tools/logger.hpp"


///  Constructor for SCATTERING
///    @param[in] num_of_rays: number of rays
///    @param[in] num_of_freq_scat: number of frequencies in scattering table
/////////////////////////////////////////////////////////////////////////////

int Scattering ::
    read (
        const Io &io,
              Parameters &parameters)
{

  write_to_log ("Reading scattering");


  nrays      = parameters.nrays ();
  nfreqs_red = parameters.nfreqs_red ();

  nfreqs_scat = 1;


  // Size and initialize scattering opacity

	opacity_scat.resize (nfreqs_scat);

	for (long f = 0; f < nfreqs_scat; f++)
	{
		opacity_scat[f] = 0.0;
	}


  // Size and initialize scattering phase function

	phase.resize (nrays);

	for (long r1 = 0; r1 < nrays; r1++)
	{
	  phase[r1].resize (nrays);

		for (long r2 = 0; r2 < nrays; r2++)
		{
	    phase[r1][r2].resize (nfreqs_red);

			for (long f = 0; f < nfreqs_red; f++)
			{
				phase[r1][r2][f] = 0.0;
			}
		}
	}


}   // END OF CONSTRUCTOR




///  add_opacity: adds the scattering contribution to the opacity
///    @param[in] frequencies: freqiencies at which to evaluate the opacity
///    @param[in/out] chi: opacity to which to add the scattering contribution
//////////////////////////////////////////////////////////////////////////////

//inline int Scattering ::
//    add_opacity (
//        vReal& chi) const
//{
//  return (0);
//}
