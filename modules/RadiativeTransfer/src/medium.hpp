// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __MEDIUM_HPP_INCLUDED__
#define __MEDIUM_HPP_INCLUDED__


struct MEDIUM
{

	long nfreq_line;
	long nfreq_cont;
	long nfreq_scat;

	double *freq_line;
	double *freq_cont;
	double *freq_scat;

	double *temperature_gas;

	double *opacity_line;
	double *opacity_cont;
	double *opacity_scat;

	double *phase_scat;

	double *emissivity_line;
	double *emissivity_cont;


  MEDIUM (long ncells, long nrays, long nfreq_l, long nfreq_c, long nfreq_s);      

	~MEDIUM ();


	double chi_line (long p, double nu);
	double chi_cont (long p, double nu);
  double chi_scat (long p, double nu);

	double eta_line (long p, double nu);
	double eta_cont (long p, double nu);

	double Phi_scat (long r1, long r2, double nu);

  double profile (long o, double freq, double line_freq);

};


#endif // __MEDIUM_HPP_INCLUDED__ 
