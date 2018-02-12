// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <string>
#include <sstream>
#include <iostream>

#include "../parameters.hpp"
#include "Magritte_config.hpp"
#include "declarations.hpp"

#include "read_linedata.hpp"
#include "species_tools.hpp"


// read_linedata: read data files containing line information in LAMBDA/RADEX format
// ---------------------------------------------------------------------------------

int read_linedata (const std::string *line_datafile, LINE_SPECIES *line_species, SPECIES *species)
{

  char buffer[BUFFER_SIZE];            // buffer for a line of data

  char buffer_name[BUFFER_SIZE];       // buffer for name of line producing species

  double buff1, buff2, buff3, buff4;   // buffers to load data


  // For all line producing species

  for (int lspec = 0; lspec < NLSPEC; lspec++)
  {

    // Open data file

    FILE *data = fopen (line_datafile[lspec].c_str(), "r");


    // Skip first line

    fgets (buffer, BUFFER_SIZE, data);


    // Read name of line producing species

    fgets (buffer, BUFFER_SIZE, data);
    sscanf (buffer, "%s %*[^\n]\n", buffer_name);

    std::string str(buffer_name);
    line_species->sym[lspec] = buffer_name;
    line_species->nr[lspec]  = get_species_nr (species, buffer_name);


    // Skip first 5 lines

    for (int l = 0; l < 5; l++)
    {
      fgets (buffer, BUFFER_SIZE, data);
    }


    // Read energy levels

    for (int l = 0; l < nlev[lspec]; l++)
    {
      int n;                               // helper index

      fgets (buffer, BUFFER_SIZE, data);
      sscanf (buffer, "%d %lf %lf %*[^\n]\n", &n, &buff1, &buff2);

      line_species->energy[LSPECLEV(lspec,l)] = HH*CC*buff1;   // energy converted from cm^-1 to erg
      line_species->weight[LSPECLEV(lspec,l)] = buff2;

      // printf("%s\n",buffer);
      // printf( "(read_linedata): level energy and weight are %lE \t %.1f \n",
      //         energy[LSPECLEV(lspec,l)]/CC/HH, weight[LSPECLEV(lspec,l)] );
    }

    // printf( "(read_linedata): \n" );

    // Skip next 3 lines

    for (int l = 0; l < 3; l++)
    {
      fgets (buffer, BUFFER_SIZE, data);
    }


    /* Read transitions and Einstein A coefficients */

    for (int l = 0; l < nrad[lspec]; l++)
    {
      int n;      // helper index

      int i, j;   // level indices

      fgets (buffer, BUFFER_SIZE, data);
      sscanf (buffer, "%d %d %d %lE %lE %*[^\n] \n", &n, &i, &j, &buff1, &buff2);

      line_species->irad[LSPECRAD(lspec,l)] = i-1;   // shift levels 1 down to have usual indexing
      line_species->jrad[LSPECRAD(lspec,l)] = j-1;   // shift levels 1 down to have usual indexing

      line_species->A_coeff[LSPECLEVLEV(lspec,i-1,j-1)] = buff1;

      line_species->frequency[LSPECLEVLEV(lspec,i-1,j-1)] = buff2;

      // printf( "(read_linedata): i, j, A_ij and frequency are %d \t %d \t %lE \t %lE \n",
      //         i-1, j-1, A_coeff[LSPECLEVLEV(lspec,irad[LSPECRAD(lspec,l)],jrad[LSPECRAD(lspec,l)])],
      //                 frequency[LSPECLEVLEV(lspec,irad[LSPECRAD(lspec,l)],jrad[LSPECRAD(lspec,l)])] );
    }



    // Skip next 2 lines

    for (int l = 0; l < 2; l++)
    {
      fgets (buffer, BUFFER_SIZE, data);
    }


    // For each collision partner

    for (int par4 = 0; par4 < ncolpar[lspec]; par4++)
    {

      // Skip next line

      fgets (buffer, BUFFER_SIZE, data);


      // Extract species corresponding to collision partner

      fgets (buffer, BUFFER_SIZE, data);


      // Use function defined below to extract collision partner species

      extract_collision_partner (species, line_species, buffer, lspec, par4);


      // Skip next 5 lines

      for (int l = 0; l < 5; l++)
      {
        fgets (buffer, BUFFER_SIZE, data);
      }


      // Read collision temperatures

      for (int tindex1 = 0; tindex1 < ncoltemp[LSPECPAR(lspec,par4)]; tindex1++){

        fscanf (data, "\t %lf \t", &buff3);

        line_species->coltemp[LSPECPARTEMP(lspec,par4,tindex1)] = buff3;


        // if ( (lspec == 0) && (par4 ==2 )) printf("AIAIAI %1.1lE\n", line_species->coltemp[LSPECPARTEMP(lspec,par4,tindex1)]);

        // printf( "(read_linedata): collisional temperature %*.2lf K\n", MAX_WIDTH,
        //         coltemp[LSPECPARTEMP(lspec,par4,tindex1)] );

      }


      // Go to next line (previous fscanf() did not do that!)

      fscanf (data, "%*[^\n]\n");


      // Read collision rates

      // printf("(read_linedata): C_data\n");

      for (int l = 0; l < ncoltran[LSPECPAR(lspec,par4)]; l++)
      {
        int n;      // helper index

        int i, j;   // level indices

        // Read first 3 entries of line containing transition level indices

        fscanf (data, "%d \t %d \t %d \t", &n, &i, &j);

        line_species->icol[LSPECPARTRAN(lspec,par4,l)] = i-1;   // shift levels 1 down to have usual indexing
        line_species->jcol[LSPECPARTRAN(lspec,par4,l)] = j-1;   // shift levels 1 down to have usual indexing


        // printf( "\t i = %d   j = %d \n",
        //         icol[LSPECPARTRAN(lspec,par4,l)], jcol[LSPECPARTRAN(lspec,par4,l)] );


        // Read rest of line containing C_data

        for (int tindex2 = 0; tindex2 < ncoltemp[LSPECPAR(lspec,par4)]; tindex2++)
        {
          fscanf (data, "%lf", &buff4);

          line_species->C_data[LSPECPARTRANTEMP(lspec,par4,l,tindex2)] = buff4;

          // printf("  %.2lE", C_data[LSPECPARTRANTEMP(lspec,par4,l,tindex2)]);
        }


        // Go to next line (previous fscanf() did not do that!)

        fgets (buffer, BUFFER_SIZE, data);

        // printf("\n");
      }

      // printf("\n");


    } // end of par4 loop over collision partners

    fclose (data);


    // Use data to calculate all coefficients in proper units

    for (int l = 0; l < nrad[lspec]; l++)
    {

      int i = line_species->irad[LSPECRAD(lspec,l)];
      int j = line_species->jrad[LSPECRAD(lspec,l)];

      int l_ij = LSPECLEVLEV(lspec,i,j);
      int l_ji = LSPECLEVLEV(lspec,j,i);


      // Frequency is in GHz, convert to Hz

      line_species->frequency[l_ij] = 1.0E9 * line_species->frequency[l_ij];


      // Energy/frequency of transition is symmetric

      line_species->frequency[l_ji] = line_species->frequency[l_ij];


      // Calculate Einstein B coefficients

      line_species->B_coeff[l_ij] = line_species->A_coeff[l_ij] * pow(CC, 2)
                                   / ( 2.0*HH*pow(line_species->frequency[l_ij] , 3) );

      line_species->B_coeff[l_ji] = line_species->weight[LSPECLEV(lspec,i)]
                                   / line_species->weight[LSPECLEV(lspec,j)] * line_species->B_coeff[l_ij];

/////////////

      // i = line_species[lspec].irad[l];
      // j = line_species[lspec].jrad[l];
      //
      //
      // // Frequency is in GHz, convert to Hz
      //
      // line_species[lspec].frequency[i][j] = 1.0E9 * line_species[lspec].frequency[i][j];
      //
      //
      // // Energy/frequency of transition is symmetric
      //
      // line_species[lspec].frequency[j][i] = line_species[lspec].frequency[i][j];
      //
      //
      // // Calculate Einstein B coefficients
      //
      // line_species[lspec].B[i][j] = line_species[lspec].A[i][j] * pow(CC, 2)
      //                               / (2.0*HH*pow(line_species[lspec].frequency[i][j], 3));
      //
      // line_species[lspec].B[j][i] = line_species[lspec].weight[i] / line_species[lspec].weight[j]
      //                               * B_coeff[l_ij];

/////////////



      // printf( "(read_linedata): A_ij, B_ij and B_ji are  %lE \t %lE \t %lE \n",
      //         line_species.A_coeff[LSPECLEVLEV(lspec,i,j)], line_species.B_coeff[LSPECLEVLEV(lspec,i,j)],
      //         line_species.B_coeff[LSPECLEVLEV(lspec,j,i)] );
    }


    // printf("(read_linedata): intput C_data = \n");
    //
    // for (int par=0; par<ncolpar[lspec]; par++){
    //
    //   for (int ctran=0; ctran<ncoltran[LSPECPAR(lspec,par)]; ctran++){
    //
    //     for (int ctemp=0; ctemp<ncoltemp[LSPECPAR(lspec,par)]; ctemp++){
    //
    //       printf( "  %.2lE", line_species.C_data[LSPECPARTRANTEMP(lspec,par, ctran, ctemp)] );
    //     }
    //
    //     printf("\n");
    //   }
    //
    //   printf("\n");
    // }
  }


  return(0);

}




// extract_collision_partner: extract species corresponding to collision partner
// -----------------------------------------------------------------------------

int extract_collision_partner (SPECIES *species, LINE_SPECIES *line_species, char *buffer, int lspec, int par)
{

  int n;                                        // index

  char buffer2[BUFFER_SIZE];                    // possibly modified copy of buffer

  int cursor, cursor2;                          // index of position in a string

  char string1[10], string2[10], string3[10];   // buffers for symbols of species


  // Addapt for inconsistencies in specification of collision partners

  cursor2 = 0;
  buffer2[cursor2] = buffer[0];
  cursor2++;

  for (cursor = 1; cursor < BUFFER_SIZE/3; cursor++ )
  {
    if ( (buffer[cursor] == '-') && (buffer[cursor-1] != 'o') && (buffer[cursor-1] != 'p') )
    {
      buffer2[cursor2] = ' ';
      cursor2++;
      buffer2[cursor2] = '-';
      cursor2++;
      buffer2[cursor2] = ' ';
      cursor2++;
    }

    else
    {
      buffer2[cursor2] = buffer[cursor];
      cursor2++;
    }
  }

  sscanf (buffer2, "%d %s %s %s %*[^\n] \n", &n, string1, string2, string3);


  // string3 contains name of collision partner

  std::string name = string3;


  // Use one of species_tools to find species nr corresponding to coll. partner

  line_species->partner[LSPECPAR(lspec,par)] = get_species_nr (species, name);


  // Check whether collision partner is ortho- or para- H2 (or something else)

  line_species->ortho_para[LSPECPAR(lspec,par)] = check_ortho_para (name);


  return (0);

}
