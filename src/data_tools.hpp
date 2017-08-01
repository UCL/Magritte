/* Frederik De Ceuster - University College London                                               */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/* Header for data_tools.cpp                                                                     */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/



#ifndef __DATA_TOOLS_HPP_INCLUDED__
#define __DATA_TOOLS_HPP_INCLUDED__



#include <string>
using namespace std;



/* get_NGRID: Count number of grid points in input file input/iNGRID.txt                         */
/*-----------------------------------------------------------------------------------------------*/

long get_NGRID(string grid_inputfile);

/*-----------------------------------------------------------------------------------------------*/



/* get_NSPEC: get the number of species in the data file                                         */
/*-----------------------------------------------------------------------------------------------*/

int get_NSPEC(string spec_datafile);

/*-----------------------------------------------------------------------------------------------*/



/* get_NREAC: get the number of chemical reactions in the data file                              */
/*-----------------------------------------------------------------------------------------------*/

int get_NREAC(string reac_datafile);

/*-----------------------------------------------------------------------------------------------*/



/* get_nlev: get number of energy levels from data file in LAMBDA/RADEX format                   */
/*-----------------------------------------------------------------------------------------------*/

int get_nlev(string datafile);

/*-----------------------------------------------------------------------------------------------*/



/* get_nrad: get number of radiative transitions from data file in LAMBDA/RADEX format           */
/*-----------------------------------------------------------------------------------------------*/

int get_nrad(string datafile);

/*-----------------------------------------------------------------------------------------------*/



/* get_ncolpar: get number of collision partners from data file in LAMBDA/RADEX format           */
/*-----------------------------------------------------------------------------------------------*/

int get_ncolpar(string datafile);

/*-----------------------------------------------------------------------------------------------*/



/* get_ncoltran: get number of collisional transitions from data file in LAMBDA/RADEX format     */
/*-----------------------------------------------------------------------------------------------*/

int get_ncoltran(string datafile, int *ncoltran, int lspec);

/*-----------------------------------------------------------------------------------------------*/



/* get_ncoltemp: get number of collisional temperatures from data file in LAMBDA/RADEX format    */
/*-----------------------------------------------------------------------------------------------*/

int get_ncoltemp(string datafile, int *ncoltran, int partner, int lspec);

/*-----------------------------------------------------------------------------------------------*/



#endif /* __DATA_TOOLS_HPP_INCLUDED__ */

/*-----------------------------------------------------------------------------------------------*/
