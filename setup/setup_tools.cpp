/* Frederik De Ceuster - University College London                                               */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/* Setup: Read the sizes of the datafile and use these in definitions.hpp                        */
/*                                                                                               */
/* (NEW)                                                                                         */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/



#include <stdio.h>
#include <stdlib.h>

#include <string>
using namespace std;

#include "setup_tools.hpp"



#define BUFFER_SIZE 500



/* get_file: get the input file name from parameters.txt                                         */
/*-----------------------------------------------------------------------------------------------*/

string get_file(int line)
{

  string filename = "../" + get_string(line);               /* relative path w.r.t /setup folder */

  return filename;

}

/*-----------------------------------------------------------------------------------------------*/





/* get_nr: get the input number from parameters.txt                                              */
/*-----------------------------------------------------------------------------------------------*/

double get_nr(int line)
{

  char buffer1[BUFFER_SIZE];
  double buffer2;


  /* Open the parameters.txt file */

  FILE *params = fopen("../parameters.txt", "r");


  /* Skip the lines before the file name */

  for (int l=0; l<line-1; l++){

    fgets(buffer1, BUFFER_SIZE, params);
  }

  fgets(buffer1, BUFFER_SIZE, params);

  sscanf(buffer1, "%lf %*[^\n]\n", &buffer2);

  double nr = buffer2;


  fclose(params);


  return nr;

}

/*-----------------------------------------------------------------------------------------------*/





/* get_string: get the string at the given line in parameters.txt                                */
/*-----------------------------------------------------------------------------------------------*/

string get_string(int line)
{

  char buffer1[BUFFER_SIZE];
  char buffer2[BUFFER_SIZE];


  /* Open the parameters.txt file */

  FILE *params = fopen("../parameters.txt", "r");


  /* Skip the lines before the file name */

  for (int l=0; l<line-1; l++){

    fgets(buffer1, BUFFER_SIZE, params);
  }

  fgets(buffer1, BUFFER_SIZE, params);

  sscanf(buffer1, "%s %*[^\n]\n", buffer2);

  string word = buffer2;


  fclose(params);


  return word;

}

/*-----------------------------------------------------------------------------------------------*/
