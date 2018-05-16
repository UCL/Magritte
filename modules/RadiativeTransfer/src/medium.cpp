// Magritte: Muldimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <math.h>

#include "medium.hpp"
#include "declarations.hpp"
#include "interpolation.hpp"

#define V_TURB 10
#define C(c,a) (ncells*(c) + (a))


double MEDIUM :: profile (long o, double freq, double line_freq)
{
  const double width = line_freq/CC * sqrt(2.0*KB*temperature_gas[o]/MP + V_TURB*V_TURB);

  return exp( -pow((freq - line_freq)/width, 2) ) / sqrt(PI) / width;
}




MEDIUM :: MEDIUM (long ncells, long nrays, long nfreq_l, long nfreq_c, long nfreq_s)
{

	nfreq_line = nfreq_l;
  nfreq_cont = nfreq_c;
 	nfreq_scat = nfreq_s;

	freq_line = new double[nfreq_line];
	freq_cont = new double[nfreq_cont];
	freq_scat = new double[nfreq_scat];

 	opacity_line = new double[ncells*nfreq_line];
 	opacity_cont = new double[ncells*nfreq_cont];
 	opacity_scat = new double[ncells*nfreq_scat];

 	emissivity_line = new double[ncells*nfreq_line];
 	emissivity_cont = new double[ncells*nfreq_cont];

	phase_scat = new double[nrays*nrays*nfreq_scat];

	temperature_gas = new double[ncells];

   
	for (long n = 0; n < ncells; n++)
	{
		for (long y = 0; y < nfreq_l; y++)
		{
			opacity_line[nfreq_l*n+y] = 0.0;
			emissivity_line[nfreq_l*n+y] = 0.0;
		}
		for (long y = 0; y < nfreq_c; y++)
		{
			opacity_cont[nfreq_c*n+y] = 0.0;
			emissivity_cont[nfreq_c*n+y] = 0.0;
		}
		for (long y = 0; y < nfreq_s; y++)
		{
			opacity_scat[nfreq_s*n+y] = 0.0;
		}

		temperature_gas[n] = 0.0;
	}

	for (long r1 = 0; r1 < nrays; r1++)
	{
		for (long r2 = 0; r2 < nrays; r2++)
		{
			for (long y = 0; y < nfreq_s; y++)
			{
				phase_scat[nfreq_s*nrays*r1 + nrays*r2 + y] = 0.0;
			}
		}
	}

}   // END OF CONSTRUCTOR




MEDIUM :: ~MEDIUM ()
{

  delete [] freq_line;
  delete [] freq_cont;
  delete [] freq_scat;

	delete [] opacity_line;
	delete [] opacity_cont;
	delete [] opacity_scat;

	delete [] emissivity_line;
	delete [] emissivity_cont;

	delete [] phase_scat;

	delete [] temperature_gas;

}   // END OF DESTRUCTOR




double MEDIUM :: chi_line (long p, double nu)
{
	double chi = 0.0;
		
	// find which lines are close enough to nu
//	  line[p,]

//		for (long f = 0; f < nfreq; f++)
//		{
//			chi += opacity[] * profile (o, freq, line_freq); 
//		}

	return chi;
}




double MEDIUM :: chi_cont (long p, double nu)
{
  return 0.0;
}




double MEDIUM :: chi_scat (long p, double nu)
{
	return 0.0;
}

	


double MEDIUM :: eta_line (long p, double nu)
{
	double chi = 0.0;
		
	// find which lines are close enough to nu
//	  line[p,]
//
//		for (long f = 0; f < nfreq; f++)
//		{
//			chi += opacity[] * profile (o, freq, line_freq); 
//		}

	return chi;
}




double MEDIUM :: eta_cont (long p, double nu)
{
	return 0.0;
}
