/* Frederik De Ceuster - University College London                                               */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/* Header for read_linedata.cpp                                                                     */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/



#ifndef __READ_LINEDATA_HPP_INCLUDED__
#define __READ_LINEDATA_HPP_INCLUDED__



#include <string>
using namespace std;



/* read_linedata: read data files containing the line information in LAMBDA/RADEX format         */
/*-----------------------------------------------------------------------------------------------*/

void read_linedata( string *line_datafile, int *irad, int *jrad, double *energy, double *weight,
                    double *frequency, double *A_coeff, double *B_coeff, double *coltemp,
                    double *C_data, int *icol, int *jcol );

/*-----------------------------------------------------------------------------------------------*/



/* extract_spec_par: extract the species corresponding to the collision partner                  */
/*-----------------------------------------------------------------------------------------------*/

void extract_spec_par(char *buffer, int lspec, int par);

/*-----------------------------------------------------------------------------------------------*/



#endif /* __READ_LINEDATA_HPP_INCLUDED__ */

/*-----------------------------------------------------------------------------------------------*/
