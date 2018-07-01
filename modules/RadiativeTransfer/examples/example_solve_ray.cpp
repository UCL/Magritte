// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <fstream>
#include <iostream>
#include <vector>
using namespace std;

#include "../src/timer.hpp"
#include "../src/solve_ray.cpp"
#include "../src/GridTypes.hpp"


int main (void)
{

  // Setup

  std::ifstream infile ("example_data/feautrier1.txt");

  const long ndep      = 100;
	const long nfreq     =  50;
	const long nfreq_red = (nfreq + n_simd_lanes - 1) / n_simd_lanes;



  long n;

  vector<double>     S (ndep);
  vector<double>  dtau (ndep);
  vector<double> u_sol (ndep);


	for (long i = 0; i < ndep; i++)
  {
  	infile >> n >> S[i] >> dtau[i] >> u_sol[i];
  }


	//int n_simd_lanes = vReal :: Nsimd();


  vReal2 u      (ndep, vReal1 (nfreq_red));
  vReal2 v      (ndep, vReal1 (nfreq_red));
	vReal2 u_prev (ndep, vReal1 (nfreq_red));
	vReal2 v_prev (ndep, vReal1 (nfreq_red));

  //vector<MatrixXd> Lambda (nfreq);

  //MatrixXd temp (ndep,ndep);   


  //for (long f = 0; f < nfreq; f++)
  //{
  //  Lambda[f] = temp;
	//}

  vReal2 Lambda (ndep, vReal1 (nfreq_red));

	long ndiag = ndep;


	long n_r  = ndep/2;
	long n_ar = ndep-n_r;

  vReal2    Su_r (n_r, vReal1 (nfreq_red));
  vReal2    Sv_r (n_r, vReal1 (nfreq_red));
  vReal2  dtau_r (n_r, vReal1 (nfreq_red));

  vReal2   Su_ar (n_ar, vReal1 (nfreq_red));
  vReal2   Sv_ar (n_ar, vReal1 (nfreq_red));
  vReal2 dtau_ar (n_ar, vReal1 (nfreq_red));


  for (long m = 0; m < n_ar; m++)
  {
		for (long f = 0; f < nfreq_red; f++)
		{
        Su_ar[m][f] =    S[n_ar-1-m];
        Sv_ar[m][f] =    S[n_ar-1-m];
      dtau_ar[m][f] = dtau[n_ar-1-m];
		}
	}	
		

  for (long m = 0; m < n_r; m++)
	{
		for (long f = 0; f < nfreq_red; f++)
		{	
  	    Su_r[m][f] =    S[n_ar+m];
        Sv_r[m][f] =    S[n_ar+m];
      dtau_r[m][f] = dtau[n_ar+m];
		}
	}


	// Solving Feautrier equations

	TIMER timer;
	timer.start ();

	for (int n = 0; n < 500; n++)
	{
    solve_ray (n_r,  Su_r,  Sv_r,  dtau_r,
		  	       n_ar, Su_ar, Sv_ar, dtau_ar,
			  			 ndep, nfreq_red, u, v, ndiag, Lambda);
	}

	timer.stop ();
	timer.print ();


	return (0);

}
