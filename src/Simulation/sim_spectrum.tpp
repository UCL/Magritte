// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include "Functions/heapsort.hpp"


///  Computer for spectral (=frequency) discretisation
//////////////////////////////////////////////////////

int Simulation ::
    compute_spectral_discretisation ()
{

  OMP_PARALLEL_FOR (p, parameters.ncells())
  {
    Double1 freqs (parameters.nfreqs());
    Long1   nmbrs (parameters.nfreqs());
    long    index0 = 0;
    long    index1 = 0;


    // Add the line frequencies (over the profile)
    for (int l = 0; l < parameters.nlspecs(); l++)
    {
      const double inverse_mass = lines.lineProducingSpecies[l].linedata.inverse_mass;

      for (int k = 0; k < lines.lineProducingSpecies[l].linedata.nrad; k++)
      {
        const double freqs_line = lines.line[index0];
        const double width      = freqs_line * thermodynamics.profile_width (inverse_mass, p);

        for (long z = 0; z < parameters.nquads(); z++)
        {
          const double root = lines.lineProducingSpecies[l].quadrature.roots[z];

          freqs[index1] = freqs_line + width * root;
          nmbrs[index1] = index1;

          index1++;
        }

        index0++;
      }
    }


    /*
     *  Set other frequencies...
     */


    // Add extra frequency bins around line to improve spectrum
    //
    // for (int l = 0; l < linedata.nlspec; l++)
    // {
    //   for (int k = 0; k < linedata.nrad[l]; k++)
    //   {
    //     const double freq_line = linedata.frequency[l][k];
    //     const double width     = profile_width (temperature.gas[p],
    //                                             temperature.vturb2[p],
    //                                             freq_line);
    //
    //     double factor = 1.0;
    //
    //     for (long e = 0; e < nbins; e++)
    //     {
    //       freqs[index1] = freq_line + width*LOWER * factor;
    //       nmbrs[index1] = index1;
    //
    //       index1++;
    //
    //       freqs[index1] = freq_line + width*UPPER * factor;
    //       nmbrs[index1] = index1;
    //
    //       index1++;
    //
    //       factor += 0.7;
    //     }
    //   }
    // }
    //
    //
    // // Add linspace for background
    //
    // // Find freqmax and freqmin
    //
    // long freqmax = 0;
    //
    // for (long f = 0; f < nfreq; f++)
    // {
    //   if (freqs[f] > freqmax)
    //   {
    //     freqmax = freqs[f];
    //   }
    // }
    //
    //
    // long freqmin = freqmax;
    //
    // for (long f = 0; f < nfreq; f++)
    // {
    //   if ( (freqs[f] < freqmin) && (freqs[f] != 0.0) )
    //   {
    //     freqmin = freqs[f];
    //   }
    // }
    //
    //
    // for (int i = 0; i < ncont; i++)
    // {
    //   freqs[index1] = (freqmax-freqmin) / ncont * i + freqmin;
    //   nmbrs[index1] = index1;
    //
    //   index1++;
    // }

    // Sort frequencies
    heapsort (freqs, nmbrs);


    // Set all frequencies nu
    for (long fl = 0; fl < parameters.nfreqs(); fl++)
    {
#     if (GRID_SIMD)
        const long    f = newIndex (fl);
        const long lane = laneNr   (fl);
        radiation.frequencies.nu[p][f].putlane (freqs[fl], lane);
#     else
        radiation.frequencies.nu[p][fl] = freqs[fl];
#     endif
    }


#   if (GRID_SIMD)

      // Remove possible first zeros

      for (int lane = n_simd_lanes-2; lane >= 0; lane--)
      {
        if (radiation.frequencies.nu[p][0].getlane(lane) <= 0.0)
        {
          const double freq = radiation.frequencies.nu[p][0].getlane(lane+1);

          radiation.frequencies.nu[p][0].putlane(0.9*freq, lane);
        }
      }

#   endif


    // Create lookup table for the frequency corresponding to each line
    Long1 nmbrs_inverted (parameters.nfreqs());

    for (long fl = 0; fl < parameters.nfreqs(); fl++)
    {
      nmbrs_inverted[nmbrs[fl]] = fl;

      radiation.frequencies.appears_in_line_integral[fl] = false;;
      radiation.frequencies.corresponding_l_for_spec[fl] = parameters.nfreqs();
      radiation.frequencies.corresponding_k_for_tran[fl] = parameters.nfreqs();
      radiation.frequencies.corresponding_z_for_line[fl] = parameters.nfreqs();
    }

    long index2 = 0;

    for (int l = 0; l < parameters.nlspecs(); l++)
    {
      for (int k = 0; k < lines.lineProducingSpecies[l].nr_line[p].size(); k++)
      {
        for (long z = 0; z < lines.lineProducingSpecies[l].nr_line[p][k].size(); z++)
        {
          lines.lineProducingSpecies[l].nr_line[p][k][z] = nmbrs_inverted[index2];

          radiation.frequencies.appears_in_line_integral[index2] = true;
          radiation.frequencies.corresponding_l_for_spec[index2] = l;
          radiation.frequencies.corresponding_k_for_tran[index2] = k;
          radiation.frequencies.corresponding_z_for_line[index2] = z;

          index2++;
        }
      }
    }


  } // end of OMP_PARALLEL_FOR (p, parameters.ncells())


  // Set spectral discretisation setting
  specDiscSetting = LineSet;


  return (0);

}




///  Computer for spectral (=frequency) discretisation
///  Gives same frequency bins to each point
///    @param[in] width : corresponding line width for frequency bins
/////////////////////////////////////////////////////////////////////

int Simulation ::
    compute_spectral_discretisation_image (
        const double width                )
{


  OMP_PARALLEL_FOR (p, parameters.ncells())
  {
    Double1 freqs (parameters.nfreqs());
    Long1   nmbrs (parameters.nfreqs());
    long    index0 = 0;
    long    index1 = 0;


    // Add the line frequencies (over the profile)
    for (int l = 0; l < parameters.nlspecs(); l++)
    {
      for (int k = 0; k < lines.lineProducingSpecies[l].linedata.nrad; k++)
      {
        const double freqs_line = lines.line[index0];

        for (long z = 0; z < parameters.nquads(); z++)
        {
          const double root = lines.lineProducingSpecies[l].quadrature.roots[z];

          freqs[index1] = freqs_line + freqs_line * width * root;
          nmbrs[index1] = index1;

          index1++;
        }

        index0++;
      }
    }


    // Sort frequencies
    heapsort (freqs, nmbrs);


    // Set all frequencies nu
    for (long fl = 0; fl < parameters.nfreqs(); fl++)
    {
#     if (GRID_SIMD)
        const long    f = newIndex (fl);
        const long lane = laneNr   (fl);
        radiation.frequencies.nu[p][f].putlane (freqs[fl], lane);
#     else
        radiation.frequencies.nu[p][fl] = freqs[fl];
#     endif
    }


#   if (GRID_SIMD)

      // Remove possible first zeros

      for (int lane = n_simd_lanes-2; lane >= 0; lane--)
      {
        if (radiation.frequencies.nu[p][0].getlane(lane) <= 0.0)
        {
          const double freq = radiation.frequencies.nu[p][0].getlane(lane+1);

          radiation.frequencies.nu[p][0].putlane(0.9*freq, lane);
        }
      }

#   endif


    // Create lookup table for the frequency corresponding to each line
    Long1 nmbrs_inverted (parameters.nfreqs());

    for (long fl = 0; fl < parameters.nfreqs(); fl++)
    {
      nmbrs_inverted[nmbrs[fl]] = fl;

      radiation.frequencies.appears_in_line_integral[fl] = false;;
      radiation.frequencies.corresponding_l_for_spec[fl] = parameters.nfreqs();
      radiation.frequencies.corresponding_k_for_tran[fl] = parameters.nfreqs();
      radiation.frequencies.corresponding_z_for_line[fl] = parameters.nfreqs();
    }

    long index2 = 0;

    for (int l = 0; l < parameters.nlspecs(); l++)
    {
      for (int k = 0; k < lines.lineProducingSpecies[l].nr_line[p].size(); k++)
      {
        for (long z = 0; z < lines.lineProducingSpecies[l].nr_line[p][k].size(); z++)
        {
          lines.lineProducingSpecies[l].nr_line[p][k][z] = nmbrs_inverted[index2];

          radiation.frequencies.appears_in_line_integral[index2] = true;
          radiation.frequencies.corresponding_l_for_spec[index2] = l;
          radiation.frequencies.corresponding_k_for_tran[index2] = k;
          radiation.frequencies.corresponding_z_for_line[index2] = z;

          index2++;
        }
      }
    }


  } // end of OMP_PARALLEL_FOR (p, parameters.ncells())


  // Set spectral discretisation setting
  specDiscSetting = ImageSet;


  return (0);

}
