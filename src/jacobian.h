/* Frederik De Ceuster - University College London                                               */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/* Header for jacobian.c                                                                         */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/



#ifndef __JACOBIAN_H_INCLUDED__
#define __JACOBIAN_H_INCLUDED__



#include <nvector/nvector_serial.h>  /* serial N_Vector types, fcts., macros */

static int Jac( long int N, realtype t, N_Vector y, N_Vector fy, DlsMat J, void *user_data,
                N_Vector tmp1, N_Vector tmp2, N_Vector tmp3 );



#endif /* __JACOBIAN_H_INCLUDED__ */

/*-----------------------------------------------------------------------------------------------*/
