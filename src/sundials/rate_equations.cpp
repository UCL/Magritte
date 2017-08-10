/* Frederik De Ceuster - University College London                                               */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/* rate_equations: defines the (chemical) rate equations                                         */
/*                                                                                               */
/* ( based on odes in 3D-PDR                                                                     */
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
 * f routine. Compute function f(t,y) in dy = f(t,y).
 */



static int f(realtype t, N_Vector y, N_Vector ydot, void *user_data)
{

  USER_DATA data = (USER_DATA) user_data;

  long gridp = data->gp;

  GRIDPOINT *gridpoint = data->gridpointer;
 
  double n_H = gridpoint[gridp].density;

  double loss, form;


  x_e = Ith(y,5+2);

  /* The ODEs created by MakeRates begin here... */

  loss = -reaction[0].k[gridp]*Ith(y,3+2)*n_H;
  form = +reaction[1].k[gridp]*Ith(y,1+2)*Ith(y,2+2)*n_H;
  Ith(ydot,0+2) = form+Ith(y,0+2)*loss;

  loss = -reaction[1].k[gridp]*Ith(y,2+2)*n_H-reaction[3].k[gridp];
  form = +reaction[0].k[gridp]*Ith(y,0+2)*Ith(y,3+2)*n_H+reaction[2].k[gridp]*Ith(y,3+2)*Ith(y,4+2)*n_H;
  Ith(ydot,1+2) = form+Ith(y,1+2)*loss;

  loss = -reaction[1].k[gridp]*Ith(y,1+2)*n_H;
  form = +reaction[0].k[gridp]*Ith(y,0+2)*Ith(y,3+2)*n_H;
  Ith(ydot,2+2) = form+Ith(y,2+2)*loss;

  loss = -reaction[0].k[gridp]*Ith(y,0+2)*n_H-reaction[2].k[gridp]*Ith(y,4+2)*n_H-reaction[4].k[gridp];
  form = +reaction[1].k[gridp]*Ith(y,1+2)*Ith(y,2+2)*n_H+reaction[3].k[gridp]*Ith(y,1+2)+reaction[5].k[gridp]*Ith(y,5+2)*x_e*n_H;
  Ith(ydot,3+2) = form+Ith(y,3+2)*loss;

  loss = -reaction[2].k[gridp]*Ith(y,3+2)*n_H;
  form = +reaction[3].k[gridp]*Ith(y,1+2);
  Ith(ydot,4+2) = form+Ith(y,4+2)*loss;

  loss = -reaction[5].k[gridp]*x_e*n_H;
  form = +reaction[4].k[gridp]*Ith(y,3+2);
  Ith(ydot,5+2) = form+Ith(y,5+2)*loss;


  return(0);
}
 /*-----------------------------------------------------------------------------------------------*/

