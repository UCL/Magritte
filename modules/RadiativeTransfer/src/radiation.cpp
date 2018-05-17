// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include "radiation.hpp"
#include "interpolation.hpp"


#define RCF(r,c,f) ( nfreq*(ncells*(r) + (c)) + (f) )


RADIATION :: RADIATION (long ncells_loc, long nrays_loc, long nfreq_loc)
{

	ncells = ncells_loc;
	nrays  =  nrays_loc;
	nfreq  =  nfreq_loc;

	frequencies = new double[nfreq];

	U_d = new double[ncells*nrays*nfreq];
	V_d = new double[ncells*nrays*nfreq];


	for (long y = 0; y < nfreq; y++)
	{
		frequencies[y] = 0.0;

  	for (long p = 0; p < ncells; p++)
  	{
	    for (long r = 0; r < nrays; r++)
      {
        U_d[RCF(r,p,y)] = 0.0;
        V_d[RCF(r,p,y)] = 0.0;
      }	
	  }
	}

}   // END OF CONSTRUCTOR




RADIATION :: ~RADIATION ()
{

  delete [] frequencies;

  delete [] U_d;
  delete [] V_d;

}   // END OF DESTRUCTOR




double RADIATION :: U (long p, long r, double nu)
{
  const long start = RCF(r,p,0);
  const long stop  = RCF(r,p,nfreq-1);

 	return interpolate (U_d, frequencies, start, stop, nu);
}




double RADIATION :: V (long p, long r, double nu)
{
  const long start = RCF(r,p,0);
  const long stop  = RCF(r,p,nfreq-1);

 	return interpolate (V_d, frequencies, start, stop, nu);
}
