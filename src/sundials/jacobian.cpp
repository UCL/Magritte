/* Frederik De Ceuster - University College London                                               */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/* jacobian: Calculates the Jacobian of the rate equations                                       */
/*                                                                                               */
/* ( based on calculate_abundances in 3D-PDR                                                     */
/*   and the cvRobers_dns example that comes with Sundials )                                     */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/



#include <cvode/cvode.h>             /* prototypes for CVODE fcts., consts. */
#include <nvector/nvector_serial.h>  /* serial N_Vector types, fcts., macros */
#include <cvode/cvode_dense.h>       /* prototype for CVDense */
#include <sundials/sundials_dense.h> /* definitions DlsMat DENSE_ELEM */
#include <sundials/sundials_types.h> /* definition of type realtype */

#include "../declarations.hpp"



/*
 * Jacobian routine. Compute J(t,y) = df/dy.
 */


/* User-defined vector and matrix accessor macros: Ith, IJth */

/* These macros are defined in order to write code which exactly matches
   the mathematical problem description given above.

   Ith(v,i) references the ith component of the vector v, where i is in
   the range [1..NEQ] and NEQ is defined below. The Ith macro is defined
   using the N_VIth macro in nvector.h. N_VIth numbers the components of
   a vector starting from 0.

   IJth(A,i,j) references the (i,j)th element of the dense matrix A, where
   i and j are in the range [1..NEQ]. The IJth macro is defined using the
   DENSE_ELEM macro in dense.h. DENSE_ELEM numbers rows and columns of a
   dense matrix starting from 0. */

#define Ith(v,i)    NV_Ith_S(v,i-1)       /* Ith numbers components 1..NEQ */
#define IJth(A,i,j) DENSE_ELEM(A,i-1,j-1) /* IJth numbers rows,cols 1..NEQ */







static int Jac( long int N, realtype t, N_Vector y, N_Vector fy, DlsMat J, void *user_data,
                N_Vector tmp1, N_Vector tmp2, N_Vector tmp3 )
{


  USER_DATA data = (USER_DATA) user_data;
  
  long gridp = data->gp;

  GRIDPOINT *gridpoint = data->gridpointer;

  double n_H = gridpoint[gridp].density;



  /* The Jacobian matrix created by MakeRates begin here... */
  x_e = Ith(y,5+2);
  IJth(J,0+2,0+2) = -reaction[0].k[gridp]*Ith(y,3+2)*n_H;
  IJth(J,1+2,0+2) = +reaction[1].k[gridp]*Ith(y,2+2)*n_H;
  IJth(J,2+2,0+2) = +reaction[1].k[gridp]*Ith(y,1+2)*n_H;
  IJth(J,3+2,0+2) = -reaction[0].k[gridp]*Ith(y,0+2)*n_H;
  IJth(J,0+2,1+2) = +reaction[0].k[gridp]*Ith(y,3+2)*n_H;
  IJth(J,1+2,1+2) = -reaction[1].k[gridp]*Ith(y,2+2)*n_H-reaction[3].k[gridp];
  IJth(J,2+2,1+2) = -reaction[1].k[gridp]*Ith(y,1+2)*n_H;
  IJth(J,3+2,1+2) = +reaction[0].k[gridp]*Ith(y,0+2)*n_H+reaction[2].k[gridp]*Ith(y,4+2)*n_H;
  IJth(J,4+2,1+2) = +reaction[2].k[gridp]*Ith(y,3+2)*n_H;
  IJth(J,0+2,2+2) = +reaction[0].k[gridp]*Ith(y,3+2)*n_H;
  IJth(J,1+2,2+2) = -reaction[1].k[gridp]*Ith(y,2+2)*n_H;
  IJth(J,2+2,2+2) = -reaction[1].k[gridp]*Ith(y,1+2)*n_H;
  IJth(J,3+2,2+2) = +reaction[0].k[gridp]*Ith(y,0+2)*n_H;
  IJth(J,0+2,3+2) = -reaction[0].k[gridp]*Ith(y,3+2)*n_H;
  IJth(J,1+2,3+2) = +reaction[1].k[gridp]*Ith(y,2+2)*n_H+reaction[3].k[gridp];
  IJth(J,2+2,3+2) = +reaction[1].k[gridp]*Ith(y,1+2)*n_H;
  IJth(J,3+2,3+2) = -reaction[0].k[gridp]*Ith(y,0+2)*n_H-reaction[2].k[gridp]*Ith(y,4+2)*n_H-reaction[4].k[gridp];
  IJth(J,4+2,3+2) = -reaction[2].k[gridp]*Ith(y,3+2)*n_H;
  IJth(J,5+2,3+2) = +reaction[5].k[gridp]*x_e*n_H;
  IJth(J,1+2,4+2) = +reaction[3].k[gridp];
  IJth(J,3+2,4+2) = -reaction[2].k[gridp]*Ith(y,4+2)*n_H;
  IJth(J,4+2,4+2) = -reaction[2].k[gridp]*Ith(y,3+2)*n_H;
  IJth(J,3+2,5+2) = +reaction[4].k[gridp];
  IJth(J,5+2,5+2) = -reaction[5].k[gridp]*x_e*n_H;

  return(0);
}
/*-----------------------------------------------------------------------------------------------*/

