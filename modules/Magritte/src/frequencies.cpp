// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <algorithm>
#include <fstream>
#include <iostream>
#include <iomanip>

#include "frequencies.hpp"
#include "GridTypes.hpp"
#include "constants.hpp"
#include "ompTools.hpp"
#include "temperature.hpp"
#include "heapsort.hpp"
#include "profile.hpp"
#include "Lines/src/linedata.hpp"


///  Constructor for FREQUENCIES
///    @param[in] num_of_cells: number of cells
///    @param[in] linedata: data structure containing the line data
///////////////////////////////////////////////////////////////////

Frequencies ::
    Frequencies (
        const long number_of_cells,
        const long number_of_line_species,
        const long number_of_freqs )
  : ncells     (number_of_cells),
    nlspecs    (number_of_line_species),
    nrad       (get_nrad (linedata_folder)),
    nlines     (count_nlines (linedata)),
    nfreqs     (count_nfreq (nlines, nbins, ncont)),
    nfreqs_red (reduced (nfreq))

{

  // Size and initialize all, order and deorder

       nu.resize (ncells);
  nr_line.resize (ncells);

# pragma omp parallel   \
  shared (linedata)     \
  default (none)
  {

  for (long p = OMP_start(ncells); p < OMP_stop(ncells); p++)
  {
         nu[p].resize (nfreqs_red);
    nr_line[p].resize (nlspecs);

    for (int l = 0; l < nlspecs; l++)
    {
      nr_line[p][l].resize (nrad[l]);

      for (int k = 0; k < nrad[l]; k++)
      {
        nr_line[p][l][k].resize (N_QUADRATURE_POINTS);
      }
    }

  }
  } // end of pragma omp parallel


  line      .resize (nlines);
  line_index.resize (nlines);



}   // END OF CONSTRUCTOR




///  read: read in the data file
////////////////////////////////

int Frequencies ::
    read (
        const string input_folder)
{

  // Read line frequecies

  ifstream lineFrequencyFile (input_folder + "linedata/frequency.txt");

  long index = 0;

  for (int l = 0; l < nlspecs; l++)
  {
    for (int k = 0; k < nrad[l]; k++)
    {
      lineFrequencyFile >> line[index];

      line_index[index] = index;
      index++;
    }
  }

  infile.close();


  return (0);

}

///  setup:
////////////////////////////////

int Frequencies ::
    setup ()
{

  // frequencies.nu has to be initialized (for unused entries)

# pragma omp parallel   \
  default (none)
  {
  for (long p = OMP_start(ncells); p < OMP_stop(ncells); p++)
  {
    for (long f = 0; f < nfreqs_red; f++)
    {
       nu[p][f] = 0.0;
    }
  }
  }


  // Sort line frequencies

  heapsort (line, line_index);


  return (0);

}



int Frequencies ::
    write (
        const string output_folder,
        const string tag           ) const
{

  // Print all frequencies (nu)

  const string file_name = output_folder + "frequencies_nu" + tag + ".txt";

  ofstream outputFile (file_name);

  outputFile << scientific << setprecision(16);


  for (long p = 0; p < ncells; p++)
  {
    for (long f = 0; f < nfreqs_red; f++)
    {
#     if (GRID_SIMD)
        for (int lane = 0; lane < n_simd_lanes; lane++)
        {
          outputFile << nu[p][f].getlane(lane) << "\t";
        }
#     else
        outputFile << nu[p][f] << "\t";
#     endif
    }

    outputFile << endl;
  }

  outputFile.close ();


  // Print line frequency numbers

  const string file_name_lnr = output_folder + "frequencies_line_nr" + tag + ".txt";

  ofstream outputFile_lnr (file_name_lnr);

  for (long p = 0; p < ncells; p++)
  {
    for (int l = 0; l < nr_line[p].size(); l++)
    {
      for (int k = 0; k < nr_line[p][l].size(); k++)
      {
        outputFile_lnr << nr_line[p][l][k][NR_LINE_CENTER] << "\t";
      }
    }

    outputFile_lnr << endl;
  }

  outputFile_lnr.close ();


  return (0);

}




///  count_nlines: count the number of lines
///    @param[in] linedata: data structure containing the line data
///////////////////////////////////////////////////////////////////

long Frequencies ::
     count_nlines (const LINEDATA& linedata)
{

  long index = 0;

  for (int l = 0; l < linedata.nlspec; l++)
  {
    index += nrad[l];
  }


  return index;

}




///  count_nfreq: count the number of frequencies
///    @param[in] linedata: data structure containing the line data
///////////////////////////////////////////////////////////////////

long Frequencies ::
     count_nfreq (const long nlines,
                  const long nbins,
                  const long ncont )
{

  long index = 0;


  // Count line frequencies

  index += nlines * N_QUADRATURE_POINTS;

  // Add extra frequency bins around lines to get nicer spectrum

  index += nlines * 2 * nbins;


  /*
   *  Count other frequencies...
   */

  // Add ncont bins background

  index += ncont;


  // Ensure that nfreq is a multiple of n_simd_lanes

  long nfreq_red_tmp = (index + n_simd_lanes - 1) / n_simd_lanes;

  return nfreq_red_tmp * n_simd_lanes;

}



///  reset: specify the frequencies under consideration given the temperature
///    @param[in] linedata: data structure containing the line data
///    @param[in] temperature: data structure containiing the temperature fields
////////////////////////////////////////////////////////////////////////////////

int Frequencies ::
    reset (
        const Temperature &temperature)
{

# pragma omp parallel                   \
  shared (temperature, H_roots, cout)   \
  default (none)
  {

  for (long p = OMP_start(ncells); p < OMP_stop(mcells); p++)
  {
    long index1 = 0;

    Long1   nmbrs (nfreqs);
    Double1 freqs (nfreqs);


    // Add the line frequencies (over the profile)

    for (int t = 0; t <nlines; t++)
    {
      const double freqs_line = line[t];
      const double width      = profile_width (temperature.gas[p],
                                               temperature.vturb2[p],
                                               freq_line);

      for (long z = 0; z < N_QUADRATURE_POINTS; z++)
      {
        freqs[index1] = freqs_line + width*H_roots[z];
        nmbrs[index1] = index1;

        index1++;
      }
    }


    /*
     *  Set other frequencies...
     */


    // // Add extra frequency bins around line to improve spectrum
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

    for (long fl = 0; fl < nfreqs; fl++)
    {
#     if (GRID_SIMD)
        const long    f = fl / n_simd_lanes;
        const long lane = fl % n_simd_lanes;
        nu[p][f].putlane(freqs[fl], lane);
#     else
        nu[p][fl] = freqs[fl];
#     endif
    }

    // Create lookup table for the frequency corresponding to each line

    Long1 nmbrs_inverted (nfreq);

    for (long fl = 0; fl < nfreqs; fl++)
    {
      nmbrs_inverted[nmbrs[fl]] = fl;
    }


    long index2 = 0;

    for (int l = 0; l < nlspecs; l++)
    {
      for (int k = 0; k < nrad[l]; k++)
      {
        for (long z = 0; z < N_QUADRATURE_POINTS; z++)
        {
          nr_line[p][l][k][z] = nmbrs_inverted[index2];
          index2++;
        }
      }
    }

  }
  } // end of pragma omp parallel


  return (0);

}
