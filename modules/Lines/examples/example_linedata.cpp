// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <iostream>
using namespace std;

#include "../src/linedata.hpp"


int main (void)
{
	LINEDATA linedata;

//	cout << linedata.irad[0][0] << endl;
//	cout << linedata.num[0] << endl;
//  cout << linedata.ncolpar[0] << endl;


  for (int l = 0; l < linedata.nlspec; l++)
	{
		for (int c = 0; c < linedata.ncolpar[l]; c++)
		{
      cout << linedata.num_col_partner[l][c] << endl;
		}
	}

	cout << "I'm fine" << endl;
	cout << linedata.A[0] << endl;
	cout << linedata.B[0] << endl;
	cout << linedata.C_data[0][0][2] << endl;

	return (0);

}
