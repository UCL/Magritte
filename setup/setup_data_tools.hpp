/* Frederik De Ceuster - University College London & KU Leuven                                   */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/* Header for setup_data_tools.cpp                                                               */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/



#ifndef __SETUP_DATA_TOOLS_HPP_INCLUDED__
#define __SETUP_DATA_TOOLS_HPP_INCLUDED__



#include <string>



/* get_NGRID: Count number of grid points in input file input/iNGRID.txt                         */
/*-----------------------------------------------------------------------------------------------*/

long get_NGRID(std::string grid_inputfile);

/*-----------------------------------------------------------------------------------------------*/



/* get_NSPEC: get the number of species in the data file                                         */
/*-----------------------------------------------------------------------------------------------*/

int get_NSPEC(std::string spec_datafile);

/*-----------------------------------------------------------------------------------------------*/



/* get_NREAC: get the number of chemical reactions in the data file                              */
/*-----------------------------------------------------------------------------------------------*/

int get_NREAC(std::string reac_datafile);

/*-----------------------------------------------------------------------------------------------*/



/* get_nlev: get number of energy levels from data file in LAMBDA/RADEX format                   */
/*-----------------------------------------------------------------------------------------------*/

int get_nlev(std::string datafile);

/*-----------------------------------------------------------------------------------------------*/



/* get_nrad: get number of radiative transitions from data file in LAMBDA/RADEX format           */
/*-----------------------------------------------------------------------------------------------*/

int get_nrad(std::string datafile);

/*-----------------------------------------------------------------------------------------------*/



/* get_ncolpar: get number of collision partners from data file in LAMBDA/RADEX format           */
/*-----------------------------------------------------------------------------------------------*/

int get_ncolpar(std::string datafile);

/*-----------------------------------------------------------------------------------------------*/



/* get_ncoltran: get number of collisional transitions from data file in LAMBDA/RADEX format     */
/*-----------------------------------------------------------------------------------------------*/

int get_ncoltran(std::string datafile, int *ncoltran, int *ncolpar, int *cum_ncolpar, int lspec);

/*-----------------------------------------------------------------------------------------------*/



/* get_ncoltemp: get number of collisional temperatures from data file in LAMBDA/RADEX format    */
/*-----------------------------------------------------------------------------------------------*/

int get_ncoltemp(std::string datafile, int *ncoltran, int *cum_ncolpar, int partner, int lspec);

/*-----------------------------------------------------------------------------------------------*/



#endif /* __SETUP_DATA_TOOLS_HPP_INCLUDED__ */

/*-----------------------------------------------------------------------------------------------*/
