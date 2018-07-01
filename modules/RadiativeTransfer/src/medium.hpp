// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __MEDIUM_HPP_INCLUDED__
#define __MEDIUM_HPP_INCLUDED__


#include <vector>
using namespace std;


struct MEDIUM
{

	long nfreq_cont;   ///< total number of continuum frequencies
	long nfreq_scat;   ///< total number of 

	vector<vector<double>> freq_line;
	vector<double> freq_cont;
	vector<double> freq_scat;

	vector<vector<double>> opacity_cont;

	vector<vector<double>> emissivity_cont;


  MEDIUM (long ncells, long nrays, long nfreq_l, long nfreq_c, long nfreq_s);      


	// Opacites

	int add_chi_line (long p, vector<double> frequencies, vector<double>& chi);
	int add_chi_cont (long p, vector<double> frequencies, vector<double>& chi);
  int add_chi_scat (long p, vector<double> frequencies, vector<double>& chi);


	// Emissivities

	int add_eta_line (long p, vector<double> frequencies, vector<double>& eta);
	int add_eta_cont (long p, vector<double> frequencies, vector<double>& eta);


};


#endif // __MEDIUM_HPP_INCLUDED__ 
