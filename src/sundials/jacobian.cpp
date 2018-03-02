// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


// User-defined vector and matrix accessor macros: Ith, IJth

#define Ith(v,i)    NV_Ith_S(v,i)         // Ith numbers components 0..NEQ-1
#define IJth(A,i,j) SM_ELEMENT_D(A,i,j)   // IJth numbers rows,cols 0..NEQ-1


// Jac: Computes J(t,y) = df/dy
// ----------------------------

static int Jac (realtype t, N_Vector y, N_Vector fy, SUNMatrix J, void *user_data,
                N_Vector tmp1, N_Vector tmp2, N_Vector tmp3)
{

  USER_DATA data = (USER_DATA) user_data;

  long o = data->gp;

  CELL *cell = data->cellpointer;

  realtype n_H = (realtype) cell[o].density;
  realtype x_e = data->electron_abundance;

  /* The Jacobian matrix created by MakeRates begin here... */
  x_e = Ith(y,5);
  data->electron_abundance = x_e;

  IJth(J,0,0) = -cell[o].rate[0]*Ith(y,3)*n_H;
  IJth(J,1,0) = +cell[o].rate[1]*Ith(y,2)*n_H;
  IJth(J,2,0) = +cell[o].rate[1]*Ith(y,1)*n_H;
  IJth(J,3,0) = -cell[o].rate[0]*Ith(y,0)*n_H;
  IJth(J,0,1) = +cell[o].rate[0]*Ith(y,3)*n_H;
  IJth(J,1,1) = -cell[o].rate[1]*Ith(y,2)*n_H-cell[o].rate[3];
  IJth(J,2,1) = -cell[o].rate[1]*Ith(y,1)*n_H;
  IJth(J,3,1) = +cell[o].rate[0]*Ith(y,0)*n_H+cell[o].rate[2]*Ith(y,4)*n_H;
  IJth(J,4,1) = +cell[o].rate[2]*Ith(y,3)*n_H;
  IJth(J,0,2) = +cell[o].rate[0]*Ith(y,3)*n_H;
  IJth(J,1,2) = -cell[o].rate[1]*Ith(y,2)*n_H;
  IJth(J,2,2) = -cell[o].rate[1]*Ith(y,1)*n_H;
  IJth(J,3,2) = +cell[o].rate[0]*Ith(y,0)*n_H;
  IJth(J,0,3) = -cell[o].rate[0]*Ith(y,3)*n_H;
  IJth(J,1,3) = +cell[o].rate[1]*Ith(y,2)*n_H+cell[o].rate[3];
  IJth(J,2,3) = +cell[o].rate[1]*Ith(y,1)*n_H;
  IJth(J,3,3) = -cell[o].rate[0]*Ith(y,0)*n_H-cell[o].rate[2]*Ith(y,4)*n_H-cell[o].rate[4];
  IJth(J,4,3) = -cell[o].rate[2]*Ith(y,3)*n_H;
  IJth(J,5,3) = +cell[o].rate[5]*x_e*n_H;
  IJth(J,1,4) = +cell[o].rate[3];
  IJth(J,3,4) = -cell[o].rate[2]*Ith(y,4)*n_H;
  IJth(J,4,4) = -cell[o].rate[2]*Ith(y,3)*n_H;
  IJth(J,3,5) = +cell[o].rate[4];
  IJth(J,5,5) = -cell[o].rate[5]*x_e*n_H;

  return(0);
}
/*-----------------------------------------------------------------------------------------------*/

