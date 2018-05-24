// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <fstream>
#include <iostream>
#include <vector>
using namespace std;

#include "timer.hpp"

#include "../src/solve_ray.cpp"


int main (void)
{

  // Setup

  std::ifstream infile ("example_data/feautrier1.txt");

  long ndep  = 100;
	long nfreq =  50;

  long n;

  vector<double>     S (ndep);
  vector<double>  dtau (ndep);
  vector<double> u_sol (ndep);
	

	for (long i = 0; i < ndep; i++)
  {
  	infile >> n >> S[i] >> dtau[i] >> u_sol[i];
  }


  vector<double>      u (ndep*nfreq);
  vector<double>      v (ndep*nfreq);
	vector<double> u_prev (ndep*nfreq);
	vector<double> v_prev (ndep*nfreq);

  vector<MatrixXd> Lambda (nfreq);

  MatrixXd temp (ndep,ndep);   


  for (long f = 0; f < nfreq; f++)
  {
    Lambda[f] = temp;
	}

	long ndiag = ndep;


	long n_r  = ndep/2;
	long n_ar = ndep-n_r;

  vector<double>    Su_r (n_r*nfreq);
  vector<double>    Sv_r (n_r*nfreq);
  vector<double>  dtau_r (n_r*nfreq);

  vector<double>   Su_ar (n_ar*nfreq);
  vector<double>   Sv_ar (n_ar*nfreq);
  vector<double> dtau_ar (n_ar*nfreq);


  for (long m = 0; m < n_ar; m++)
  {
		for (long f = 0; f < nfreq; f++)
		{
  	    Su_ar[m*nfreq+f] =    S[n_ar-1-m];
    	  Sv_ar[m*nfreq+f] =    S[n_ar-1-m];
    	dtau_ar[m*nfreq+f] = dtau[n_ar-1-m];
		}
	}	
		
  for (long m = 0; m < n_r; m++)
	{
		for (long f = 0; f < nfreq; f++)
		{	
		    Su_r[m*nfreq+f] =    S[n_ar+m];
		    Sv_r[m*nfreq+f] =    S[n_ar+m];
		  dtau_r[m*nfreq+f] = dtau[n_ar+m];
		}
	}


	// Solving Feautrier equations

	TIMER timer;
	timer.start ();

  solve_ray (n_r,  Su_r,  Sv_r,  dtau_r,
			       n_ar, Su_ar, Sv_ar, dtau_ar,
						 ndep, nfreq, u, v, ndiag, Lambda);

	timer.stop ();
	timer.print ();


	return (0);

}
