// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <fstream>
#include <iostream>
#include <string>
#include <chrono>

#include "../src/solve_ray.cpp"


struct TIMER
{
	std::chrono::duration <double> interval;
	std::chrono::high_resolution_clock::time_point initial;

  void start ()
  {
    initial = std::chrono::high_resolution_clock::now();
  }

  void stop ()
  { 
		interval = std::chrono::high_resolution_clock::now() - initial;
  }

	void print ()
	{
		std::cout << "time  = " << interval.count() << " seconds" << std::endl;
	}

	void print (std::string text)
	{
		std::cout << text << " time  = " << interval.count() << " seconds" << std::endl;
	}
};




int main (void)
{

  // Setup

  std::ifstream infile ("example_data/feautrier1.txt");

  long ndep  = 100;
	long nfreq =  50;

  long n;

  double *S     = new double[ndep*nfreq];
  double *dtau  = new double[ndep*nfreq];
  double *u_sol = new double[ndep*nfreq];
	

  for (long i = 0; i < ndep; i++)
  {
  	infile >> n >> S[i] >> dtau[i] >> u_sol[i];
  }


  double *u      = new double[ndep];
  double *v      = new double[ndep];
	double *u_prev = new double[ndep];
	double *v_prev = new double[ndep];

  Eigen::MatrixXd Lambda (ndep,ndep);
	

	long ndiag = ndep;


	long n_r  = n/2;
	long n_ar = ndep-n_r;

  double    *Su_r = new double[n_r];
  double    *Sv_r = new double[n_r];
  double  *dtau_r = new double[n_r];

  double   *Su_ar = new double[n_ar];
  double   *Sv_ar = new double[n_ar];
  double *dtau_ar = new double[n_ar];


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

	TIMER timer;

	timer.start ();

  solve_ray (n_r,  Su_r,  Sv_r,  dtau_r,
			       n_ar, Su_ar, Sv_ar, dtau_ar,
						 ndep,  u, v, ndiag, Lambda);

	timer.stop ();

	timer.print ();


	// Tear down
	
 	delete [] Su_r;
 	delete [] Sv_r;
	delete [] dtau_r;
	delete [] Su_ar;
	delete [] Sv_ar;
	delete [] dtau_ar;

  delete [] u;
  delete [] v;

  delete [] S;
  delete [] dtau;
  delete [] u_sol;


	return (0);

}
