// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <math.h>
#include <vector>
using namespace std;

#include "medium.hpp"
#include "profile.hpp"
#include "Common/src/interpolation.hpp"


MEDIUM :: MEDIUM (long ncells, LINEDATA linedata, long nfreq_c, long nfreq_s)
{

  nfreq_cont = nfreq_c;
 	nfreq_scat = nfreq_s;

	freq_cont.resize (nfreq_cont);
	freq_scat.resize (nfreq_scat);

 	opacity_line.resize (ncells);
 	opacity_cont.resize (ncells);

 	emissivity_line.resize (ncells);
 	emissivity_cont.resize (ncells);

  freq_line.resize (linedata.nlspec);

	for (int l = 0; l < linedata.nlspec; l++)
	{
		freq_line[l].resize (linedata.nrad[l]);

		for (int k = 0; k < linedata.nrad[l]; k++)
		{
      const int i = linedata.irad[l][k];
			const int j = linedata.jrad[l][k];

			freq_line[l][k] = linedata.frequency[l](i,j);
		}
	}


	for (long p = 0; p < ncells; p++)
	{
		opacity_line[p].resize (nfreq_line);

		for (long f = 0; f < nfreq_line; f++)
		{
			   opacity_line[p][f] = 0.0;
			emissivity_line[p][f] = 0.0;
		}

		opacity_cont[p].resize (nfreq_cont);

		for (long f = 0; f < nfreq_cont; f++)
		{
			   opacity_cont[p][f] = 0.0;
			emissivity_cont[p][f] = 0.0;
		}

		opacity_scat[p].resize (nfreq_scat);

		for (long f = 0; f < nfreq_scat; f++)
		{
			opacity_scat[p][f] = 0.0;
		}
	}


}   // END OF CONSTRUCTOR




int MEDIUM :: add_chi_line (TEMPERATURE temperature, long p, vector<double> frequencies, vector<double>& chi)
{
		
	// find which lines are close enough to nu
//	  line[p,]

		const double profile = profile (temperature.gas[p], freq_line, freq);

		for (long y = 0; y < nfreq; y++)
		{
		  const double profile = profile (temperature.gas[p], freq_line, freq);

			chi[f] += opacity[]    * profile; 
			eta[f] += emissivity[] * profile; 
		}

	return (0);
}




int MEDIUM :: add_chi_cont (long p, vector<double> frequencies, vector<double>& chi)
{
  return (0);
}




int MEDIUM :: add_chi_scat (long p, vector<double> frequencies, vector<double>& chi)
{
	return (0);
}

	


int MEDIUM :: add_eta_line (long p, vector<double> frequencies, vector<double>& eta)
{
		
	// find which lines are close enough to nu
//	  line[p,]
//
//		for (long f = 0; f < nfreq; f++)
//		{
//			chi += opacity[] * profile (o, freq, line_freq); 
//		}

	return (0);
}




int MEDIUM :: add_eta_cont (long p, vector<double> frequencies, vector<double>& eta)
{
	return (0);
}
