// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <fstream>
#include <iostream>
using namespace std;

#include "../src/timer.hpp"
#include "../src/solve_ray.hpp"
#include "../src/GridTypes.hpp"


int main (void)
{

  cout << "n_simd_lanes = " << n_simd_lanes << endl;

  #if (GRID_SIMD)
    cout << "GRID_SIMD is true" << endl;
  #else
    cout << "GRID_SIMD is false" << endl;
  #endif

  // Setup

  ifstream infile ("example_data/feautrier1.txt");

  const long ndep      = 100;
	const long nfreq     = 140;
	const long nfreq_red = (nfreq + n_simd_lanes - 1) / n_simd_lanes;

  cout << "nfreq_red = " << (nfreq + n_simd_lanes - 1) / n_simd_lanes << endl;



  long n;

  vector<double>     S (ndep);
  vector<double>  dtau (ndep);
  vector<double> u_sol (ndep);


	for (long i = 0; i < ndep; i++)
  {
  	infile >> n >> S[i] >> dtau[i] >> u_sol[i];
  }


	//int n_simd_lanes = vReal :: Nsimd();


  //vector<MatrixXd> Lambda (nfreq);

  //MatrixXd temp (ndep,ndep);


  //for (long f = 0; f < nfreq; f++)
  //{
  //  Lambda[f] = temp;
	//}

  vReal Lambda[ndep];

	long ndiag = ndep;


	long n_r  = ndep/2;
	long n_ar = ndep-n_r;

  vReal   Su_r[n_r];
  vReal   Sv_r[n_r];
  vReal dtau_r[n_r];

  vReal   Su_ar[n_ar];
  vReal   Sv_ar[n_ar];
  vReal dtau_ar[n_ar];


  for (long m = 0; m < n_ar; m++)
  {
      Su_ar[m] =    S[n_ar-1-m];
      Sv_ar[m] =    S[n_ar-1-m];
    dtau_ar[m] = dtau[n_ar-1-m];
	}


  for (long m = 0; m < n_r; m++)
	{
      Su_r[m] =    S[n_ar+m];
      Sv_r[m] =    S[n_ar+m];
    dtau_r[m] = dtau[n_ar+m];
	}


	// Solving Feautrier equations

	TIMER timer ("solve_ray");
	timer.start ();

  vReal u [ndep];
  vReal v [ndep];

  vReal A [ndep];
  vReal C [ndep];
  vReal F [ndep];
  vReal G [ndep];

	vReal B0       ;
  vReal B0_min_C0;
  vReal Bd       ;
	vReal Bd_min_Ad;

	//for (int n = 0; n < 500; n++)
	//{
	  for (long f = 0; f < nfreq_red; f++)
	  {
  //    solve_ray (n_r,  Su_r,  Sv_r,  dtau_r,
	//  	  	       n_ar, Su_ar, Sv_ar, dtau_ar,
  //               A, C, F, G,
	//							 B0, B0_min_C0, Bd, Bd_min_Ad,
	//  		  			 ndep, u, v, ndiag, Lambda);
    }
	//}

	timer.stop ();
	timer.print ();


	return (0);

}
