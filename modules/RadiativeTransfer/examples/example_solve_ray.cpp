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
	const long nfreq_red = (nfreq + n_vector_lanes - 1) / n_vector_lanes;



  long n;

  vector<double>     S (ndep);
  vector<double>  dtau (ndep);
  vector<double> u_sol (ndep);


	for (long i = 0; i < ndep; i++)
  {
  	infile >> n >> S[i] >> dtau[i] >> u_sol[i];
  }


	//int n_vector_lanes = vDouble :: Nsimd();


  vDouble2 u      (ndep, vDouble1 (nfreq_red));
  vDouble2 v      (ndep, vDouble1 (nfreq_red));
	vDouble2 u_prev (ndep, vDouble1 (nfreq_red));
	vDouble2 v_prev (ndep, vDouble1 (nfreq_red));

  //vector<MatrixXd> Lambda (nfreq);

  //MatrixXd temp (ndep,ndep);   


  //for (long f = 0; f < nfreq; f++)
  //{
  //  Lambda[f] = temp;
	//}

  vDouble2 Lambda (ndep, vDouble1 (nfreq_red));

	long ndiag = ndep;


	long n_r  = ndep/2;
	long n_ar = ndep-n_r;

  vDouble2    Su_r (n_r, vDouble1 (nfreq_red));
  vDouble2    Sv_r (n_r, vDouble1 (nfreq_red));
  vDouble2  dtau_r (n_r, vDouble1 (nfreq_red));

  vDouble2   Su_ar (n_ar, vDouble1 (nfreq_red));
  vDouble2   Sv_ar (n_ar, vDouble1 (nfreq_red));
  vDouble2 dtau_ar (n_ar, vDouble1 (nfreq_red));


  for (long m = 0; m < n_ar; m++)
  {
		for (long f = 0; f < nfreq_red; f++)
		{
			for (int lane = 0; lane < n_vector_lanes; lane++)
			{
  	      Su_ar[m][f].putlane (   S[n_ar-1-m], lane);
    	    Sv_ar[m][f].putlane (   S[n_ar-1-m], lane);
    	  dtau_ar[m][f].putlane (dtau[n_ar-1-m], lane);
			}
		}
	}	
		

  for (long m = 0; m < n_r; m++)
	{
		for (long f = 0; f < nfreq_red; f++)
		{	
			for (int lane = 0; lane < n_vector_lanes; lane++)
			{
  	      Su_r[m][f].putlane (   S[n_ar+m], lane);
    	    Sv_r[m][f].putlane (   S[n_ar+m], lane);
    	  dtau_r[m][f].putlane (dtau[n_ar+m], lane);
			}
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
