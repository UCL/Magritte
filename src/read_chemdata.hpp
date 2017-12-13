/* Frederik De Ceuster - University College London & KU Leuven                                   */
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



/* read_species: read the species from the data file                                             */
/*-----------------------------------------------------------------------------------------------*/

int read_species(std::string spec_datafile, double *initial_abn);

/*-----------------------------------------------------------------------------------------------*/



/* read_reactions: read the reactoins from the (CSV) data file                                                  */
/*-----------------------------------------------------------------------------------------------*/

int read_reactions(std::string reac_datafile);

/*-----------------------------------------------------------------------------------------------*/



#endif /* __READ_CHEMDATA_HPP_INCLUDED__ */

/*-----------------------------------------------------------------------------------------------*/
