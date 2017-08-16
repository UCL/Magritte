/* Frederik De Ceuster - University College London                                               */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/* Header for read_chemdata.cpp                                                                     */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/



#ifndef __READ_CHEMDATA_HPP_INCLUDED__
#define __READ_CHEMDATA_HPP_INCLUDED__



#include <string>
using namespace std;



/* read_species: read the species from the data file                                             */
/*-----------------------------------------------------------------------------------------------*/

void read_species(string spec_datafile);

/*-----------------------------------------------------------------------------------------------*/



/* read_reactions: read the reactoins from the (CSV) data file                                                  */
/*-----------------------------------------------------------------------------------------------*/

void read_reactions(string reac_datafile);

/*-----------------------------------------------------------------------------------------------*/



#endif /* __READ_CHEMDATA_HPP_INCLUDED__ */

/*-----------------------------------------------------------------------------------------------*/
