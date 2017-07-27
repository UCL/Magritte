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

  int i, j;                                                                             /* index */

   /* Some temporary changes of variables untill I can generate the equations automatically */

  realtype x[NSPEC];

  realtype *J_temp;
  J_temp = (realtype*) malloc( NSPEC*NSPEC*sizeof(realtype) );
  
  realtype rate[NREAC];


  for (i=0; i<NSPEC; i++){

    x[i] = Ith(y,i+1);

    for (j=0; j<NSPEC; j++){

      J_temp[ i*NSPEC+ j] = 0.0;
    }
  }

  for (j=0; j<NREAC; j++){

    rate[j] = reaction[j].k;
    printf("test: %lE \n", rate[j]);
  }

  realtype n_H = 1.0;

  x_e = x[0]+x[1]+x[3]+x[5]+x[6]+x[7]+x[8]+x[9]+x[10]+x[12]+x[13]+x[14]+x[15]+x[18]+x[19]+x[22]+x[23]+x[26];


  /* The Jacobian matrix created by MakeRates begin here... */
  J_temp[ 0*NSPEC+ 0] = -rate[67]*x[24]*n_H-rate[68]*x[20]*n_H-rate[69]*x[16]*n_H-rate[70]*x[21]*n_H-rate[71]*x[29]*n_H-rate[72]*x[11]*n_H-rate[73]*x[28]*n_H-rate[74]*x[17]*n_H-rate[75]*x[2]*n_H-rate[76]*x[27]*n_H-rate[216]*x_e*n_H-rate[217]*x_e*n_H-rate[266]-rate[267];
  J_temp[ 2*NSPEC+ 0] = -rate[75]*x[0]*n_H;
  J_temp[ 5*NSPEC+ 0] = +rate[48]*x[30]*n_H;
  J_temp[11*NSPEC+ 0] = -rate[72]*x[0]*n_H;
  J_temp[16*NSPEC+ 0] = -rate[69]*x[0]*n_H;
  J_temp[17*NSPEC+ 0] = -rate[74]*x[0]*n_H;
  J_temp[20*NSPEC+ 0] = -rate[68]*x[0]*n_H;
  J_temp[21*NSPEC+ 0] = -rate[70]*x[0]*n_H;
  J_temp[24*NSPEC+ 0] = -rate[67]*x[0]*n_H;
  J_temp[27*NSPEC+ 0] = -rate[76]*x[0]*n_H;
  J_temp[28*NSPEC+ 0] = -rate[73]*x[0]*n_H;
  J_temp[29*NSPEC+ 0] = -rate[71]*x[0]*n_H;
  J_temp[30*NSPEC+ 0] = +rate[48]*x[5]*n_H;
  J_temp[ 1*NSPEC+ 1] = -rate[49]*x[30]*n_H-rate[77]*x[20]*n_H-rate[78]*x[16]*n_H-rate[79]*x[16]*n_H-rate[80]*x[21]*n_H-rate[81]*x[11]*n_H-rate[82]*x[11]*n_H-rate[83]*x[11]*n_H-rate[84]*x[11]*n_H-rate[85]*x[28]*n_H-rate[86]*x[17]*n_H-rate[87]*x[17]*n_H-rate[88]*x[27]*n_H-rate[89]*x[27]*n_H-rate[90]*x[27]*n_H-rate[91]*x[4]*n_H-rate[155]*x[31]*n_H-rate[156]*x[31]*n_H-rate[169]*x[30]*n_H-rate[177]*x[24]*n_H-rate[178]*x[20]*n_H-rate[179]*x[11]*n_H-rate[180]*x[17]*n_H-rate[181]*x[4]*n_H-rate[242]*x_e*n_H;
  J_temp[ 4*NSPEC+ 1] = -rate[91]*x[1]*n_H-rate[181]*x[1]*n_H;
  J_temp[11*NSPEC+ 1] = -rate[81]*x[1]*n_H-rate[82]*x[1]*n_H-rate[83]*x[1]*n_H-rate[84]*x[1]*n_H-rate[179]*x[1]*n_H;
  J_temp[16*NSPEC+ 1] = -rate[78]*x[1]*n_H-rate[79]*x[1]*n_H;
  J_temp[17*NSPEC+ 1] = -rate[86]*x[1]*n_H-rate[87]*x[1]*n_H-rate[180]*x[1]*n_H;
  J_temp[20*NSPEC+ 1] = -rate[77]*x[1]*n_H-rate[178]*x[1]*n_H;
  J_temp[21*NSPEC+ 1] = -rate[80]*x[1]*n_H;
  J_temp[24*NSPEC+ 1] = -rate[177]*x[1]*n_H;
  J_temp[25*NSPEC+ 1] = +rate[296];
  J_temp[27*NSPEC+ 1] = -rate[88]*x[1]*n_H-rate[89]*x[1]*n_H-rate[90]*x[1]*n_H;
  J_temp[28*NSPEC+ 1] = -rate[85]*x[1]*n_H;
  J_temp[30*NSPEC+ 1] = -rate[49]*x[1]*n_H-rate[169]*x[1]*n_H;
  J_temp[31*NSPEC+ 1] = -rate[155]*x[1]*n_H-rate[156]*x[1]*n_H;
  J_temp[ 0*NSPEC+ 2] = -rate[75]*x[2]*n_H;
  J_temp[ 2*NSPEC+ 2] = -rate[75]*x[0]*n_H-rate[147]*x[3]*n_H-rate[166]*x[18]*n_H-rate[184]*x[10]*n_H-rate[190]*x[26]*n_H-rate[198]*x[19]*n_H-rate[210]*x[12]*n_H-rate[213]*x[15]*n_H-rate[288]-rate[311];
  J_temp[ 3*NSPEC+ 2] = -rate[147]*x[2]*n_H;
  J_temp[ 9*NSPEC+ 2] = +rate[248]*x_e*n_H;
  J_temp[10*NSPEC+ 2] = -rate[184]*x[2]*n_H;
  J_temp[12*NSPEC+ 2] = -rate[210]*x[2]*n_H;
  J_temp[15*NSPEC+ 2] = -rate[213]*x[2]*n_H;
  J_temp[18*NSPEC+ 2] = -rate[166]*x[2]*n_H;
  J_temp[19*NSPEC+ 2] = -rate[198]*x[2]*n_H;
  J_temp[26*NSPEC+ 2] = -rate[190]*x[2]*n_H;
  J_temp[ 0*NSPEC+ 3] = +rate[72]*x[11]*n_H;
  J_temp[ 2*NSPEC+ 3] = -rate[147]*x[3]*n_H;
  J_temp[ 3*NSPEC+ 3] = -rate[47]*x[31]*n_H-rate[95]*x[24]*n_H-rate[107]*x[20]*n_H-rate[119]*x[16]*n_H-rate[130]*x[29]*n_H-rate[140]*x[28]*n_H-rate[146]*x[17]*n_H-rate[147]*x[2]*n_H-rate[148]*x[27]*n_H-rate[229]*x_e*n_H-rate[230]*x_e*n_H;
  J_temp[ 5*NSPEC+ 3] = +rate[58]*x[11]*n_H;
  J_temp[ 6*NSPEC+ 3] = +rate[59]*x[30]*n_H+rate[132]*x[11]*n_H;
  J_temp[ 8*NSPEC+ 3] = +rate[133]*x[11]*n_H;
  J_temp[11*NSPEC+ 3] = +rate[58]*x[5]*n_H+rate[72]*x[0]*n_H+rate[132]*x[6]*n_H+rate[133]*x[8]*n_H;
  J_temp[16*NSPEC+ 3] = -rate[119]*x[3]*n_H;
  J_temp[17*NSPEC+ 3] = -rate[146]*x[3]*n_H;
  J_temp[19*NSPEC+ 3] = +rate[258]*x[30]*n_H;
  J_temp[20*NSPEC+ 3] = -rate[107]*x[3]*n_H;
  J_temp[24*NSPEC+ 3] = -rate[95]*x[3]*n_H;
  J_temp[27*NSPEC+ 3] = -rate[148]*x[3]*n_H;
  J_temp[28*NSPEC+ 3] = -rate[140]*x[3]*n_H;
  J_temp[29*NSPEC+ 3] = -rate[130]*x[3]*n_H;
  J_temp[30*NSPEC+ 3] = +rate[59]*x[6]*n_H+rate[258]*x[19]*n_H;
  J_temp[31*NSPEC+ 3] = -rate[47]*x[3]*n_H;
  J_temp[ 1*NSPEC+ 4] = -rate[91]*x[4]*n_H-rate[181]*x[4]*n_H;
  J_temp[ 2*NSPEC+ 4] = +rate[213]*x[15]*n_H;
  J_temp[ 4*NSPEC+ 4] = -rate[7]*x[31]*n_H-rate[14]*x[30]*n_H-rate[18]*x[24]*n_H-rate[23]*x[20]*n_H-rate[31]*x[16]*n_H-rate[91]*x[1]*n_H-rate[100]*x[10]*n_H-rate[101]*x[10]*n_H-rate[114]*x[26]*n_H-rate[115]*x[26]*n_H-rate[124]*x[23]*n_H-rate[168]*x[18]*n_H-rate[176]*x[5]*n_H-rate[181]*x[1]*n_H-rate[204]*x[7]*n_H-rate[206]*x[6]*n_H-rate[209]*x[8]*n_H-rate[212]*x[12]*n_H-rate[214]*x[14]*n_H-rate[290]-rate[291]-rate[313]-rate[314]-rate[319]*x[31]*n_H-rate[325]*x[30]*n_H;
  J_temp[ 5*NSPEC+ 4] = -rate[176]*x[4]*n_H;
  J_temp[ 6*NSPEC+ 4] = -rate[206]*x[4]*n_H;
  J_temp[ 7*NSPEC+ 4] = -rate[204]*x[4]*n_H;
  J_temp[ 8*NSPEC+ 4] = -rate[209]*x[4]*n_H;
  J_temp[10*NSPEC+ 4] = -rate[100]*x[4]*n_H-rate[101]*x[4]*n_H;
  J_temp[12*NSPEC+ 4] = -rate[212]*x[4]*n_H;
  J_temp[14*NSPEC+ 4] = -rate[214]*x[4]*n_H;
  J_temp[15*NSPEC+ 4] = +rate[186]*x[24]*n_H+rate[192]*x[20]*n_H+rate[197]*x[16]*n_H+rate[213]*x[2]*n_H;
  J_temp[16*NSPEC+ 4] = -rate[31]*x[4]*n_H+rate[197]*x[15]*n_H;
  J_temp[18*NSPEC+ 4] = -rate[168]*x[4]*n_H;
  J_temp[20*NSPEC+ 4] = -rate[23]*x[4]*n_H+rate[192]*x[15]*n_H;
  J_temp[23*NSPEC+ 4] = -rate[124]*x[4]*n_H;
  J_temp[24*NSPEC+ 4] = -rate[18]*x[4]*n_H+rate[186]*x[15]*n_H;
  J_temp[26*NSPEC+ 4] = -rate[114]*x[4]*n_H-rate[115]*x[4]*n_H;
  J_temp[28*NSPEC+ 4] = +rate[37]*x[29]*n_H;
  J_temp[29*NSPEC+ 4] = +rate[37]*x[28]*n_H+rate[264]*x[29]*n_H;
  J_temp[30*NSPEC+ 4] = -rate[14]*x[4]*n_H-rate[325]*x[4]*n_H;
  J_temp[31*NSPEC+ 4] = -rate[7]*x[4]*n_H-rate[319]*x[4]*n_H;
  J_temp[ 0*NSPEC+ 5] = +rate[266];
  J_temp[ 1*NSPEC+ 5] = +rate[169]*x[30]*n_H;
  J_temp[ 4*NSPEC+ 5] = -rate[176]*x[5]*n_H;
  J_temp[ 5*NSPEC+ 5] = -rate[48]*x[30]*n_H-rate[50]*x[24]*n_H-rate[52]*x[20]*n_H-rate[54]*x[16]*n_H-rate[56]*x[29]*n_H-rate[58]*x[11]*n_H-rate[60]*x[11]*n_H-rate[61]*x[28]*n_H-rate[63]*x[17]*n_H-rate[65]*x[27]*n_H-rate[154]*x[31]*n_H-rate[170]*x[20]*n_H-rate[171]*x[16]*n_H-rate[172]*x[11]*n_H-rate[173]*x[28]*n_H-rate[174]*x[17]*n_H-rate[175]*x[27]*n_H-rate[176]*x[4]*n_H-rate[215]*x_e*n_H-rate[265];
  J_temp[11*NSPEC+ 5] = -rate[58]*x[5]*n_H-rate[60]*x[5]*n_H-rate[172]*x[5]*n_H;
  J_temp[16*NSPEC+ 5] = -rate[54]*x[5]*n_H-rate[171]*x[5]*n_H;
  J_temp[17*NSPEC+ 5] = -rate[63]*x[5]*n_H-rate[174]*x[5]*n_H;
  J_temp[18*NSPEC+ 5] = +rate[153]*x[30]*n_H+rate[249]*x[31]*n_H+rate[250]*x[31]*n_H;
  J_temp[20*NSPEC+ 5] = -rate[52]*x[5]*n_H-rate[170]*x[5]*n_H;
  J_temp[24*NSPEC+ 5] = -rate[50]*x[5]*n_H;
  J_temp[27*NSPEC+ 5] = -rate[65]*x[5]*n_H-rate[175]*x[5]*n_H;
  J_temp[28*NSPEC+ 5] = -rate[61]*x[5]*n_H-rate[173]*x[5]*n_H;
  J_temp[29*NSPEC+ 5] = -rate[56]*x[5]*n_H;
  J_temp[30*NSPEC+ 5] = -rate[48]*x[5]*n_H+rate[153]*x[18]*n_H+rate[169]*x[1]*n_H+rate[295];
  J_temp[31*NSPEC+ 5] = -rate[154]*x[5]*n_H+rate[249]*x[18]*n_H+rate[250]*x[18]*n_H;
  J_temp[ 0*NSPEC+ 6] = +rate[70]*x[21]*n_H;
  J_temp[ 1*NSPEC+ 6] = +rate[179]*x[11]*n_H;
  J_temp[ 3*NSPEC+ 6] = +rate[47]*x[31]*n_H;
  J_temp[ 4*NSPEC+ 6] = -rate[206]*x[6]*n_H;
  J_temp[ 5*NSPEC+ 6] = +rate[172]*x[11]*n_H;
  J_temp[ 6*NSPEC+ 6] = -rate[46]*x[31]*n_H-rate[59]*x[30]*n_H-rate[127]*x[29]*n_H-rate[132]*x[11]*n_H-rate[135]*x[17]*n_H-rate[137]*x[27]*n_H-rate[206]*x[4]*n_H-rate[226]*x_e*n_H-rate[227]*x_e*n_H;
  J_temp[ 7*NSPEC+ 6] = +rate[199]*x[11]*n_H;
  J_temp[11*NSPEC+ 6] = -rate[132]*x[6]*n_H+rate[163]*x[18]*n_H+rate[172]*x[5]*n_H+rate[179]*x[1]*n_H+rate[199]*x[7]*n_H+rate[205]*x[14]*n_H;
  J_temp[14*NSPEC+ 6] = +rate[205]*x[11]*n_H;
  J_temp[17*NSPEC+ 6] = -rate[135]*x[6]*n_H;
  J_temp[18*NSPEC+ 6] = +rate[163]*x[11]*n_H;
  J_temp[21*NSPEC+ 6] = +rate[70]*x[0]*n_H;
  J_temp[27*NSPEC+ 6] = -rate[137]*x[6]*n_H;
  J_temp[29*NSPEC+ 6] = -rate[127]*x[6]*n_H;
  J_temp[30*NSPEC+ 6] = -rate[59]*x[6]*n_H;
  J_temp[31*NSPEC+ 6] = -rate[46]*x[6]*n_H+rate[47]*x[3]*n_H;
  J_temp[ 1*NSPEC+ 7] = +rate[85]*x[28]*n_H+rate[90]*x[27]*n_H+rate[91]*x[4]*n_H;
  J_temp[ 4*NSPEC+ 7] = -rate[204]*x[7]*n_H+rate[91]*x[1]*n_H+rate[101]*x[10]*n_H;
  J_temp[ 7*NSPEC+ 7] = -rate[57]*x[30]*n_H-rate[104]*x[20]*n_H-rate[126]*x[11]*n_H-rate[128]*x[28]*n_H-rate[160]*x[31]*n_H-rate[187]*x[20]*n_H-rate[193]*x[16]*n_H-rate[199]*x[11]*n_H-rate[200]*x[28]*n_H-rate[201]*x[17]*n_H-rate[203]*x[27]*n_H-rate[204]*x[4]*n_H-rate[247]*x_e*n_H-rate[259]*x[24]*n_H;
  J_temp[10*NSPEC+ 7] = +rate[101]*x[4]*n_H;
  J_temp[11*NSPEC+ 7] = -rate[126]*x[7]*n_H-rate[199]*x[7]*n_H;
  J_temp[14*NSPEC+ 7] = +rate[202]*x[29]*n_H;
  J_temp[16*NSPEC+ 7] = -rate[193]*x[7]*n_H;
  J_temp[17*NSPEC+ 7] = -rate[201]*x[7]*n_H;
  J_temp[18*NSPEC+ 7] = +rate[161]*x[29]*n_H+rate[162]*x[29]*n_H;
  J_temp[20*NSPEC+ 7] = -rate[104]*x[7]*n_H-rate[187]*x[7]*n_H;
  J_temp[24*NSPEC+ 7] = -rate[259]*x[7]*n_H;
  J_temp[27*NSPEC+ 7] = -rate[203]*x[7]*n_H+rate[90]*x[1]*n_H;
  J_temp[28*NSPEC+ 7] = -rate[128]*x[7]*n_H-rate[200]*x[7]*n_H+rate[85]*x[1]*n_H;
  J_temp[29*NSPEC+ 7] = +rate[161]*x[18]*n_H+rate[162]*x[18]*n_H+rate[202]*x[14]*n_H+rate[298];
  J_temp[30*NSPEC+ 7] = -rate[57]*x[7]*n_H;
  J_temp[31*NSPEC+ 7] = -rate[160]*x[7]*n_H;
  J_temp[ 0*NSPEC+ 8] = +rate[71]*x[29]*n_H;
  J_temp[ 1*NSPEC+ 8] = +rate[87]*x[17]*n_H;
  J_temp[ 4*NSPEC+ 8] = -rate[209]*x[8]*n_H;
  J_temp[ 5*NSPEC+ 8] = +rate[56]*x[29]*n_H+rate[173]*x[28]*n_H;
  J_temp[ 7*NSPEC+ 8] = +rate[57]*x[30]*n_H+rate[200]*x[28]*n_H;
  J_temp[ 8*NSPEC+ 8] = -rate[62]*x[30]*n_H-rate[92]*x[24]*n_H-rate[105]*x[20]*n_H-rate[118]*x[16]*n_H-rate[129]*x[29]*n_H-rate[133]*x[11]*n_H-rate[134]*x[11]*n_H-rate[139]*x[28]*n_H-rate[141]*x[17]*n_H-rate[143]*x[27]*n_H-rate[188]*x[20]*n_H-rate[194]*x[16]*n_H-rate[207]*x[17]*n_H-rate[209]*x[4]*n_H-rate[228]*x_e*n_H-rate[285];
  J_temp[11*NSPEC+ 8] = -rate[133]*x[8]*n_H-rate[134]*x[8]*n_H;
  J_temp[14*NSPEC+ 8] = +rate[208]*x[28]*n_H;
  J_temp[16*NSPEC+ 8] = -rate[118]*x[8]*n_H-rate[194]*x[8]*n_H;
  J_temp[17*NSPEC+ 8] = -rate[141]*x[8]*n_H-rate[207]*x[8]*n_H+rate[87]*x[1]*n_H;
  J_temp[18*NSPEC+ 8] = +rate[164]*x[28]*n_H;
  J_temp[20*NSPEC+ 8] = -rate[105]*x[8]*n_H-rate[188]*x[8]*n_H;
  J_temp[24*NSPEC+ 8] = -rate[92]*x[8]*n_H;
  J_temp[27*NSPEC+ 8] = -rate[143]*x[8]*n_H;
  J_temp[28*NSPEC+ 8] = -rate[139]*x[8]*n_H+rate[164]*x[18]*n_H+rate[173]*x[5]*n_H+rate[200]*x[7]*n_H+rate[208]*x[14]*n_H+rate[283];
  J_temp[29*NSPEC+ 8] = -rate[129]*x[8]*n_H+rate[56]*x[5]*n_H+rate[71]*x[0]*n_H;
  J_temp[30*NSPEC+ 8] = -rate[62]*x[8]*n_H+rate[57]*x[7]*n_H;
  J_temp[ 0*NSPEC+ 9] = +rate[75]*x[2]*n_H;
  J_temp[ 2*NSPEC+ 9] = +rate[75]*x[0]*n_H+rate[147]*x[3]*n_H+rate[166]*x[18]*n_H+rate[184]*x[10]*n_H+rate[190]*x[26]*n_H+rate[198]*x[19]*n_H+rate[210]*x[12]*n_H+rate[213]*x[15]*n_H+rate[288]+rate[311];
  J_temp[ 3*NSPEC+ 9] = +rate[147]*x[2]*n_H;
  J_temp[ 9*NSPEC+ 9] = -rate[248]*x_e*n_H;
  J_temp[10*NSPEC+ 9] = +rate[184]*x[2]*n_H;
  J_temp[12*NSPEC+ 9] = +rate[210]*x[2]*n_H;
  J_temp[15*NSPEC+ 9] = +rate[213]*x[2]*n_H;
  J_temp[18*NSPEC+ 9] = +rate[166]*x[2]*n_H;
  J_temp[19*NSPEC+ 9] = +rate[198]*x[2]*n_H;
  J_temp[26*NSPEC+ 9] = +rate[190]*x[2]*n_H;
  J_temp[ 1*NSPEC+10] = +rate[77]*x[20]*n_H+rate[78]*x[16]*n_H+rate[88]*x[27]*n_H+rate[89]*x[27]*n_H+rate[177]*x[24]*n_H;
  J_temp[ 2*NSPEC+10] = -rate[184]*x[10]*n_H;
  J_temp[ 4*NSPEC+10] = -rate[100]*x[10]*n_H-rate[101]*x[10]*n_H;
  J_temp[10*NSPEC+10] = -rate[51]*x[30]*n_H-rate[93]*x[28]*n_H-rate[94]*x[28]*n_H-rate[97]*x[17]*n_H-rate[100]*x[4]*n_H-rate[101]*x[4]*n_H-rate[182]*x[20]*n_H-rate[183]*x[16]*n_H-rate[184]*x[2]*n_H-rate[243]*x_e*n_H-rate[244]*x_e*n_H-rate[245]*x_e*n_H-rate[252]*x[31]*n_H-rate[256]*x[30]*n_H-rate[262]*x[29]*n_H-rate[263]*x[29]*n_H;
  J_temp[14*NSPEC+10] = +rate[185]*x[24]*n_H;
  J_temp[15*NSPEC+10] = +rate[186]*x[24]*n_H;
  J_temp[16*NSPEC+10] = -rate[183]*x[10]*n_H+rate[78]*x[1]*n_H;
  J_temp[17*NSPEC+10] = -rate[97]*x[10]*n_H;
  J_temp[20*NSPEC+10] = -rate[182]*x[10]*n_H+rate[77]*x[1]*n_H;
  J_temp[24*NSPEC+10] = +rate[177]*x[1]*n_H+rate[185]*x[14]*n_H+rate[186]*x[15]*n_H+rate[268]+rate[297]+rate[300];
  J_temp[26*NSPEC+10] = +rate[41]*x[31]*n_H+rate[302];
  J_temp[27*NSPEC+10] = +rate[88]*x[1]*n_H+rate[89]*x[1]*n_H;
  J_temp[28*NSPEC+10] = -rate[93]*x[10]*n_H-rate[94]*x[10]*n_H;
  J_temp[29*NSPEC+10] = -rate[262]*x[10]*n_H-rate[263]*x[10]*n_H;
  J_temp[30*NSPEC+10] = -rate[51]*x[10]*n_H-rate[256]*x[10]*n_H;
  J_temp[31*NSPEC+10] = -rate[252]*x[10]*n_H+rate[41]*x[26]*n_H;
  J_temp[ 0*NSPEC+11] = -rate[72]*x[11]*n_H;
  J_temp[ 1*NSPEC+11] = -rate[81]*x[11]*n_H-rate[82]*x[11]*n_H-rate[83]*x[11]*n_H-rate[84]*x[11]*n_H-rate[179]*x[11]*n_H;
  J_temp[ 2*NSPEC+11] = +rate[147]*x[3]*n_H;
  J_temp[ 3*NSPEC+11] = +rate[95]*x[24]*n_H+rate[107]*x[20]*n_H+rate[119]*x[16]*n_H+rate[140]*x[28]*n_H+rate[146]*x[17]*n_H+rate[147]*x[2]*n_H+rate[148]*x[27]*n_H+rate[230]*x_e*n_H;
  J_temp[ 4*NSPEC+11] = +rate[206]*x[6]*n_H;
  J_temp[ 5*NSPEC+11] = -rate[58]*x[11]*n_H-rate[60]*x[11]*n_H-rate[172]*x[11]*n_H;
  J_temp[ 6*NSPEC+11] = -rate[132]*x[11]*n_H+rate[206]*x[4]*n_H;
  J_temp[ 7*NSPEC+11] = -rate[126]*x[11]*n_H-rate[199]*x[11]*n_H;
  J_temp[ 8*NSPEC+11] = -rate[133]*x[11]*n_H-rate[134]*x[11]*n_H;
  J_temp[11*NSPEC+11] = -rate[3]*x[31]*n_H-rate[22]*x[20]*n_H-rate[28]*x[16]*n_H-rate[36]*x[29]*n_H-rate[39]*x[28]*n_H-rate[45]*x[18]*n_H-rate[58]*x[5]*n_H-rate[60]*x[5]*n_H-rate[72]*x[0]*n_H-rate[81]*x[1]*n_H-rate[82]*x[1]*n_H-rate[83]*x[1]*n_H-rate[84]*x[1]*n_H-rate[126]*x[7]*n_H-rate[132]*x[6]*n_H-rate[133]*x[8]*n_H-rate[134]*x[8]*n_H-rate[136]*x[12]*n_H-rate[138]*x[14]*n_H-rate[163]*x[18]*n_H-rate[172]*x[5]*n_H-rate[179]*x[1]*n_H-rate[199]*x[7]*n_H-rate[205]*x[14]*n_H-rate[280]-rate[281]-rate[282]-rate[308];
  J_temp[12*NSPEC+11] = -rate[136]*x[11]*n_H;
  J_temp[14*NSPEC+11] = -rate[138]*x[11]*n_H-rate[205]*x[11]*n_H;
  J_temp[16*NSPEC+11] = -rate[28]*x[11]*n_H+rate[119]*x[3]*n_H;
  J_temp[17*NSPEC+11] = +rate[35]*x[21]*n_H+rate[146]*x[3]*n_H;
  J_temp[18*NSPEC+11] = -rate[45]*x[11]*n_H-rate[163]*x[11]*n_H;
  J_temp[20*NSPEC+11] = -rate[22]*x[11]*n_H+rate[107]*x[3]*n_H;
  J_temp[21*NSPEC+11] = +rate[11]*x[30]*n_H+rate[32]*x[21]*n_H+rate[33]*x[28]*n_H+rate[35]*x[17]*n_H;
  J_temp[24*NSPEC+11] = +rate[95]*x[3]*n_H;
  J_temp[27*NSPEC+11] = +rate[148]*x[3]*n_H;
  J_temp[28*NSPEC+11] = -rate[39]*x[11]*n_H+rate[33]*x[21]*n_H+rate[140]*x[3]*n_H;
  J_temp[29*NSPEC+11] = -rate[36]*x[11]*n_H;
  J_temp[30*NSPEC+11] = +rate[11]*x[21]*n_H;
  J_temp[31*NSPEC+11] = -rate[3]*x[11]*n_H;
  J_temp[ 0*NSPEC+12] = +rate[73]*x[28]*n_H;
  J_temp[ 1*NSPEC+12] = +rate[180]*x[17]*n_H;
  J_temp[ 2*NSPEC+12] = -rate[210]*x[12]*n_H;
  J_temp[ 3*NSPEC+12] = +rate[140]*x[28]*n_H;
  J_temp[ 4*NSPEC+12] = -rate[212]*x[12]*n_H;
  J_temp[ 5*NSPEC+12] = +rate[61]*x[28]*n_H+rate[174]*x[17]*n_H;
  J_temp[ 7*NSPEC+12] = +rate[201]*x[17]*n_H;
  J_temp[ 8*NSPEC+12] = +rate[62]*x[30]*n_H+rate[139]*x[28]*n_H+rate[207]*x[17]*n_H;
  J_temp[11*NSPEC+12] = -rate[136]*x[12]*n_H;
  J_temp[12*NSPEC+12] = -rate[64]*x[30]*n_H-rate[96]*x[24]*n_H-rate[109]*x[20]*n_H-rate[120]*x[16]*n_H-rate[131]*x[29]*n_H-rate[136]*x[11]*n_H-rate[142]*x[28]*n_H-rate[149]*x[17]*n_H-rate[150]*x[27]*n_H-rate[189]*x[20]*n_H-rate[195]*x[16]*n_H-rate[210]*x[2]*n_H-rate[212]*x[4]*n_H-rate[231]*x_e*n_H-rate[232]*x_e*n_H-rate[233]*x_e*n_H;
  J_temp[14*NSPEC+12] = +rate[211]*x[17]*n_H;
  J_temp[16*NSPEC+12] = -rate[120]*x[12]*n_H-rate[195]*x[12]*n_H;
  J_temp[17*NSPEC+12] = -rate[149]*x[12]*n_H+rate[165]*x[18]*n_H+rate[174]*x[5]*n_H+rate[180]*x[1]*n_H+rate[201]*x[7]*n_H+rate[207]*x[8]*n_H+rate[211]*x[14]*n_H+rate[287];
  J_temp[18*NSPEC+12] = +rate[165]*x[17]*n_H;
  J_temp[20*NSPEC+12] = -rate[109]*x[12]*n_H-rate[189]*x[12]*n_H;
  J_temp[22*NSPEC+12] = +rate[145]*x[28]*n_H;
  J_temp[24*NSPEC+12] = -rate[96]*x[12]*n_H;
  J_temp[27*NSPEC+12] = -rate[150]*x[12]*n_H;
  J_temp[28*NSPEC+12] = -rate[142]*x[12]*n_H+rate[61]*x[5]*n_H+rate[73]*x[0]*n_H+rate[139]*x[8]*n_H+rate[140]*x[3]*n_H+rate[145]*x[22]*n_H;
  J_temp[29*NSPEC+12] = -rate[131]*x[12]*n_H;
  J_temp[30*NSPEC+12] = -rate[64]*x[12]*n_H+rate[62]*x[8]*n_H;
  J_temp[ 0*NSPEC+13] = +rate[74]*x[17]*n_H;
  J_temp[ 3*NSPEC+13] = +rate[130]*x[29]*n_H+rate[146]*x[17]*n_H;
  J_temp[ 5*NSPEC+13] = +rate[63]*x[17]*n_H;
  J_temp[ 6*NSPEC+13] = +rate[135]*x[17]*n_H;
  J_temp[ 8*NSPEC+13] = +rate[134]*x[11]*n_H+rate[141]*x[17]*n_H;
  J_temp[11*NSPEC+13] = +rate[134]*x[8]*n_H+rate[136]*x[12]*n_H;
  J_temp[12*NSPEC+13] = +rate[64]*x[30]*n_H+rate[136]*x[11]*n_H+rate[142]*x[28]*n_H+rate[149]*x[17]*n_H;
  J_temp[13*NSPEC+13] = -rate[98]*x[24]*n_H-rate[111]*x[20]*n_H-rate[121]*x[16]*n_H-rate[234]*x_e*n_H-rate[235]*x_e*n_H-rate[236]*x_e*n_H-rate[237]*x_e*n_H;
  J_temp[16*NSPEC+13] = -rate[121]*x[13]*n_H;
  J_temp[17*NSPEC+13] = +rate[63]*x[5]*n_H+rate[74]*x[0]*n_H+rate[108]*x[26]*n_H+rate[135]*x[6]*n_H+rate[141]*x[8]*n_H+rate[146]*x[3]*n_H+rate[149]*x[12]*n_H+rate[152]*x[22]*n_H;
  J_temp[20*NSPEC+13] = -rate[111]*x[13]*n_H;
  J_temp[22*NSPEC+13] = +rate[152]*x[17]*n_H;
  J_temp[24*NSPEC+13] = -rate[98]*x[13]*n_H;
  J_temp[26*NSPEC+13] = +rate[108]*x[17]*n_H;
  J_temp[28*NSPEC+13] = +rate[142]*x[12]*n_H;
  J_temp[29*NSPEC+13] = +rate[130]*x[3]*n_H;
  J_temp[30*NSPEC+13] = +rate[64]*x[12]*n_H;
  J_temp[ 4*NSPEC+14] = -rate[214]*x[14]*n_H+rate[100]*x[10]*n_H+rate[114]*x[26]*n_H;
  J_temp[ 5*NSPEC+14] = +rate[175]*x[27]*n_H;
  J_temp[ 7*NSPEC+14] = +rate[104]*x[20]*n_H+rate[203]*x[27]*n_H+rate[259]*x[24]*n_H;
  J_temp[10*NSPEC+14] = +rate[93]*x[28]*n_H+rate[100]*x[4]*n_H+rate[262]*x[29]*n_H+rate[263]*x[29]*n_H;
  J_temp[11*NSPEC+14] = -rate[138]*x[14]*n_H-rate[205]*x[14]*n_H;
  J_temp[14*NSPEC+14] = -rate[66]*x[30]*n_H-rate[112]*x[20]*n_H-rate[122]*x[16]*n_H-rate[138]*x[11]*n_H-rate[144]*x[28]*n_H-rate[151]*x[17]*n_H-rate[167]*x[31]*n_H-rate[185]*x[24]*n_H-rate[191]*x[20]*n_H-rate[196]*x[16]*n_H-rate[202]*x[29]*n_H-rate[205]*x[11]*n_H-rate[208]*x[28]*n_H-rate[211]*x[17]*n_H-rate[214]*x[4]*n_H-rate[238]*x_e*n_H;
  J_temp[15*NSPEC+14] = +rate[102]*x[24]*n_H;
  J_temp[16*NSPEC+14] = -rate[122]*x[14]*n_H-rate[196]*x[14]*n_H;
  J_temp[17*NSPEC+14] = -rate[151]*x[14]*n_H-rate[211]*x[14]*n_H;
  J_temp[20*NSPEC+14] = -rate[112]*x[14]*n_H-rate[191]*x[14]*n_H+rate[104]*x[7]*n_H;
  J_temp[24*NSPEC+14] = -rate[185]*x[14]*n_H+rate[102]*x[15]*n_H+rate[259]*x[7]*n_H;
  J_temp[26*NSPEC+14] = +rate[103]*x[29]*n_H+rate[106]*x[28]*n_H+rate[114]*x[4]*n_H;
  J_temp[27*NSPEC+14] = +rate[175]*x[5]*n_H+rate[203]*x[7]*n_H+rate[299];
  J_temp[28*NSPEC+14] = -rate[144]*x[14]*n_H-rate[208]*x[14]*n_H+rate[93]*x[10]*n_H+rate[106]*x[26]*n_H;
  J_temp[29*NSPEC+14] = -rate[202]*x[14]*n_H+rate[103]*x[26]*n_H+rate[262]*x[10]*n_H+rate[263]*x[10]*n_H;
  J_temp[30*NSPEC+14] = -rate[66]*x[14]*n_H;
  J_temp[31*NSPEC+14] = -rate[167]*x[14]*n_H;
  J_temp[ 1*NSPEC+15] = +rate[181]*x[4]*n_H;
  J_temp[ 2*NSPEC+15] = -rate[213]*x[15]*n_H;
  J_temp[ 4*NSPEC+15] = +rate[168]*x[18]*n_H+rate[176]*x[5]*n_H+rate[181]*x[1]*n_H+rate[204]*x[7]*n_H+rate[206]*x[6]*n_H+rate[209]*x[8]*n_H+rate[212]*x[12]*n_H+rate[214]*x[14]*n_H+rate[291]+rate[314];
  J_temp[ 5*NSPEC+15] = +rate[176]*x[4]*n_H;
  J_temp[ 6*NSPEC+15] = +rate[206]*x[4]*n_H;
  J_temp[ 7*NSPEC+15] = +rate[128]*x[28]*n_H+rate[204]*x[4]*n_H;
  J_temp[ 8*NSPEC+15] = +rate[129]*x[29]*n_H+rate[209]*x[4]*n_H;
  J_temp[12*NSPEC+15] = +rate[131]*x[29]*n_H+rate[212]*x[4]*n_H;
  J_temp[14*NSPEC+15] = +rate[214]*x[4]*n_H;
  J_temp[15*NSPEC+15] = -rate[102]*x[24]*n_H-rate[116]*x[20]*n_H-rate[186]*x[24]*n_H-rate[192]*x[20]*n_H-rate[197]*x[16]*n_H-rate[213]*x[2]*n_H-rate[240]*x_e*n_H;
  J_temp[16*NSPEC+15] = -rate[197]*x[15]*n_H;
  J_temp[18*NSPEC+15] = +rate[168]*x[4]*n_H;
  J_temp[20*NSPEC+15] = -rate[116]*x[15]*n_H-rate[192]*x[15]*n_H;
  J_temp[24*NSPEC+15] = -rate[102]*x[15]*n_H-rate[186]*x[15]*n_H;
  J_temp[28*NSPEC+15] = +rate[128]*x[7]*n_H;
  J_temp[29*NSPEC+15] = +rate[129]*x[8]*n_H+rate[131]*x[12]*n_H;
  J_temp[ 0*NSPEC+16] = -rate[69]*x[16]*n_H;
  J_temp[ 1*NSPEC+16] = -rate[78]*x[16]*n_H-rate[79]*x[16]*n_H;
  J_temp[ 3*NSPEC+16] = -rate[119]*x[16]*n_H+rate[130]*x[29]*n_H;
  J_temp[ 4*NSPEC+16] = -rate[31]*x[16]*n_H;
  J_temp[ 5*NSPEC+16] = -rate[54]*x[16]*n_H-rate[171]*x[16]*n_H;
  J_temp[ 6*NSPEC+16] = +rate[227]*x_e*n_H;
  J_temp[ 7*NSPEC+16] = -rate[193]*x[16]*n_H;
  J_temp[ 8*NSPEC+16] = -rate[118]*x[16]*n_H-rate[194]*x[16]*n_H+rate[134]*x[11]*n_H;
  J_temp[10*NSPEC+16] = -rate[183]*x[16]*n_H;
  J_temp[11*NSPEC+16] = -rate[28]*x[16]*n_H+rate[22]*x[20]*n_H+rate[134]*x[8]*n_H+rate[281]+rate[308];
  J_temp[12*NSPEC+16] = -rate[120]*x[16]*n_H-rate[195]*x[16]*n_H;
  J_temp[13*NSPEC+16] = -rate[121]*x[16]*n_H;
  J_temp[14*NSPEC+16] = -rate[122]*x[16]*n_H-rate[196]*x[16]*n_H;
  J_temp[15*NSPEC+16] = -rate[197]*x[16]*n_H;
  J_temp[16*NSPEC+16] = -rate[1]*x[31]*n_H-rate[10]*x[30]*n_H-rate[15]*x[24]*n_H-2*2*rate[24]*x[16]*n_H-rate[25]*x[29]*n_H-rate[26]*x[29]*n_H-rate[27]*x[29]*n_H-rate[28]*x[11]*n_H-rate[29]*x[28]*n_H-rate[30]*x[28]*n_H-rate[31]*x[4]*n_H-rate[42]*x[18]*n_H-rate[54]*x[5]*n_H-rate[69]*x[0]*n_H-rate[78]*x[1]*n_H-rate[79]*x[1]*n_H-rate[118]*x[8]*n_H-rate[119]*x[3]*n_H-rate[120]*x[12]*n_H-rate[121]*x[13]*n_H-rate[122]*x[14]*n_H-rate[123]*x[22]*n_H-rate[158]*x[18]*n_H-rate[171]*x[5]*n_H-rate[183]*x[10]*n_H-rate[193]*x[7]*n_H-rate[194]*x[8]*n_H-rate[195]*x[12]*n_H-rate[196]*x[14]*n_H-rate[197]*x[15]*n_H-rate[272]-rate[273]-rate[303]-rate[304];
  J_temp[18*NSPEC+16] = -rate[42]*x[16]*n_H-rate[158]*x[16]*n_H;
  J_temp[19*NSPEC+16] = +rate[222]*x_e*n_H;
  J_temp[20*NSPEC+16] = +rate[9]*x[30]*n_H+rate[22]*x[11]*n_H;
  J_temp[21*NSPEC+16] = +rate[2]*x[31]*n_H+rate[32]*x[21]*n_H+rate[34]*x[28]*n_H+rate[276]+rate[306];
  J_temp[22*NSPEC+16] = -rate[123]*x[16]*n_H;
  J_temp[24*NSPEC+16] = -rate[15]*x[16]*n_H+rate[255]*x[30]*n_H;
  J_temp[28*NSPEC+16] = -rate[29]*x[16]*n_H-rate[30]*x[16]*n_H+rate[34]*x[21]*n_H;
  J_temp[29*NSPEC+16] = -rate[25]*x[16]*n_H-rate[26]*x[16]*n_H-rate[27]*x[16]*n_H+rate[130]*x[3]*n_H;
  J_temp[30*NSPEC+16] = -rate[10]*x[16]*n_H+rate[9]*x[20]*n_H+rate[255]*x[24]*n_H;
  J_temp[31*NSPEC+16] = -rate[1]*x[16]*n_H+rate[2]*x[21]*n_H;
  J_temp[ 0*NSPEC+17] = -rate[74]*x[17]*n_H;
  J_temp[ 1*NSPEC+17] = -rate[86]*x[17]*n_H-rate[87]*x[17]*n_H-rate[180]*x[17]*n_H;
  J_temp[ 2*NSPEC+17] = +rate[210]*x[12]*n_H;
  J_temp[ 3*NSPEC+17] = -rate[146]*x[17]*n_H;
  J_temp[ 4*NSPEC+17] = +rate[31]*x[16]*n_H+rate[212]*x[12]*n_H;
  J_temp[ 5*NSPEC+17] = -rate[63]*x[17]*n_H-rate[174]*x[17]*n_H;
  J_temp[ 6*NSPEC+17] = -rate[135]*x[17]*n_H;
  J_temp[ 7*NSPEC+17] = -rate[201]*x[17]*n_H;
  J_temp[ 8*NSPEC+17] = -rate[141]*x[17]*n_H-rate[207]*x[17]*n_H;
  J_temp[10*NSPEC+17] = -rate[97]*x[17]*n_H;
  J_temp[11*NSPEC+17] = +rate[39]*x[28]*n_H;
  J_temp[12*NSPEC+17] = -rate[149]*x[17]*n_H+rate[189]*x[20]*n_H+rate[195]*x[16]*n_H+rate[210]*x[2]*n_H+rate[212]*x[4]*n_H;
  J_temp[13*NSPEC+17] = +rate[111]*x[20]*n_H+rate[121]*x[16]*n_H+rate[237]*x_e*n_H;
  J_temp[14*NSPEC+17] = -rate[151]*x[17]*n_H-rate[211]*x[17]*n_H;
  J_temp[16*NSPEC+17] = +rate[30]*x[28]*n_H+rate[31]*x[4]*n_H+rate[121]*x[13]*n_H+rate[195]*x[12]*n_H;
  J_temp[17*NSPEC+17] = -rate[5]*x[31]*n_H-rate[35]*x[21]*n_H-rate[38]*x[29]*n_H-rate[63]*x[5]*n_H-rate[74]*x[0]*n_H-rate[86]*x[1]*n_H-rate[87]*x[1]*n_H-rate[97]*x[10]*n_H-rate[108]*x[26]*n_H-rate[110]*x[26]*n_H-rate[135]*x[6]*n_H-rate[141]*x[8]*n_H-rate[146]*x[3]*n_H-rate[149]*x[12]*n_H-rate[151]*x[14]*n_H-rate[152]*x[22]*n_H-rate[165]*x[18]*n_H-rate[174]*x[5]*n_H-rate[180]*x[1]*n_H-rate[201]*x[7]*n_H-rate[207]*x[8]*n_H-rate[211]*x[14]*n_H-rate[286]-rate[287]-rate[310]-rate[318]*x[31]*n_H-rate[324]*x[30]*n_H;
  J_temp[18*NSPEC+17] = -rate[165]*x[17]*n_H;
  J_temp[20*NSPEC+17] = +rate[111]*x[13]*n_H+rate[189]*x[12]*n_H;
  J_temp[21*NSPEC+17] = -rate[35]*x[17]*n_H+rate[34]*x[28]*n_H;
  J_temp[22*NSPEC+17] = -rate[152]*x[17]*n_H;
  J_temp[26*NSPEC+17] = -rate[108]*x[17]*n_H-rate[110]*x[17]*n_H;
  J_temp[28*NSPEC+17] = +rate[13]*x[30]*n_H+rate[30]*x[16]*n_H+rate[34]*x[21]*n_H+rate[39]*x[11]*n_H+rate[40]*x[28]*n_H+rate[254]*x[31]*n_H;
  J_temp[29*NSPEC+17] = -rate[38]*x[17]*n_H;
  J_temp[30*NSPEC+17] = -rate[324]*x[17]*n_H+rate[13]*x[28]*n_H;
  J_temp[31*NSPEC+17] = -rate[5]*x[17]*n_H-rate[318]*x[17]*n_H+rate[254]*x[28]*n_H;
  J_temp[ 0*NSPEC+18] = +rate[267];
  J_temp[ 1*NSPEC+18] = +rate[49]*x[30]*n_H+rate[83]*x[11]*n_H+rate[86]*x[17]*n_H+rate[155]*x[31]*n_H+rate[156]*x[31]*n_H;
  J_temp[ 2*NSPEC+18] = -rate[166]*x[18]*n_H;
  J_temp[ 4*NSPEC+18] = -rate[168]*x[18]*n_H;
  J_temp[ 5*NSPEC+18] = +rate[154]*x[31]*n_H+rate[265];
  J_temp[ 7*NSPEC+18] = +rate[160]*x[31]*n_H;
  J_temp[ 8*NSPEC+18] = +rate[285];
  J_temp[10*NSPEC+18] = +rate[94]*x[28]*n_H;
  J_temp[11*NSPEC+18] = -rate[45]*x[18]*n_H-rate[163]*x[18]*n_H+rate[83]*x[1]*n_H;
  J_temp[14*NSPEC+18] = +rate[167]*x[31]*n_H;
  J_temp[16*NSPEC+18] = -rate[42]*x[18]*n_H-rate[158]*x[18]*n_H;
  J_temp[17*NSPEC+18] = -rate[165]*x[18]*n_H+rate[86]*x[1]*n_H;
  J_temp[18*NSPEC+18] = -rate[42]*x[16]*n_H-rate[45]*x[11]*n_H-rate[153]*x[30]*n_H-rate[157]*x[20]*n_H-rate[158]*x[16]*n_H-rate[159]*x[21]*n_H-rate[161]*x[29]*n_H-rate[162]*x[29]*n_H-rate[163]*x[11]*n_H-rate[164]*x[28]*n_H-rate[165]*x[17]*n_H-rate[166]*x[2]*n_H-rate[168]*x[4]*n_H-rate[241]*x_e*n_H-rate[249]*x[31]*n_H-rate[250]*x[31]*n_H;
  J_temp[20*NSPEC+18] = -rate[157]*x[18]*n_H;
  J_temp[21*NSPEC+18] = -rate[159]*x[18]*n_H;
  J_temp[26*NSPEC+18] = +rate[271];
  J_temp[28*NSPEC+18] = -rate[164]*x[18]*n_H+rate[94]*x[10]*n_H;
  J_temp[29*NSPEC+18] = -rate[161]*x[18]*n_H-rate[162]*x[18]*n_H;
  J_temp[30*NSPEC+18] = -rate[153]*x[18]*n_H+rate[49]*x[1]*n_H+rate[293];
  J_temp[31*NSPEC+18] = -rate[249]*x[18]*n_H-rate[250]*x[18]*n_H+rate[154]*x[5]*n_H+rate[155]*x[1]*n_H+rate[156]*x[1]*n_H+rate[160]*x[7]*n_H+rate[167]*x[14]*n_H+rate[292];
  J_temp[ 0*NSPEC+19] = +rate[69]*x[16]*n_H;
  J_temp[ 1*NSPEC+19] = +rate[84]*x[11]*n_H;
  J_temp[ 2*NSPEC+19] = -rate[198]*x[19]*n_H;
  J_temp[ 3*NSPEC+19] = +rate[119]*x[16]*n_H;
  J_temp[ 5*NSPEC+19] = +rate[54]*x[16]*n_H+rate[60]*x[11]*n_H;
  J_temp[ 6*NSPEC+19] = +rate[46]*x[31]*n_H+rate[127]*x[29]*n_H;
  J_temp[ 7*NSPEC+19] = +rate[126]*x[11]*n_H;
  J_temp[ 8*NSPEC+19] = +rate[118]*x[16]*n_H;
  J_temp[11*NSPEC+19] = +rate[45]*x[18]*n_H+rate[60]*x[5]*n_H+rate[84]*x[1]*n_H+rate[126]*x[7]*n_H;
  J_temp[12*NSPEC+19] = +rate[120]*x[16]*n_H;
  J_temp[13*NSPEC+19] = +rate[121]*x[16]*n_H;
  J_temp[16*NSPEC+19] = +rate[54]*x[5]*n_H+rate[69]*x[0]*n_H+rate[118]*x[8]*n_H+rate[119]*x[3]*n_H+rate[120]*x[12]*n_H+rate[121]*x[13]*n_H+rate[123]*x[22]*n_H;
  J_temp[18*NSPEC+19] = +rate[45]*x[11]*n_H+rate[159]*x[21]*n_H;
  J_temp[19*NSPEC+19] = -rate[44]*x[31]*n_H-rate[125]*x[29]*n_H-rate[198]*x[2]*n_H-rate[222]*x_e*n_H-rate[223]*x_e*n_H-rate[224]*x_e*n_H-rate[225]*x_e*n_H-rate[246]*x_e*n_H-rate[258]*x[30]*n_H-rate[277]-rate[279];
  J_temp[21*NSPEC+19] = +rate[159]*x[18]*n_H+rate[275]+rate[305];
  J_temp[22*NSPEC+19] = +rate[123]*x[16]*n_H;
  J_temp[23*NSPEC+19] = +rate[55]*x[30]*n_H;
  J_temp[29*NSPEC+19] = -rate[125]*x[19]*n_H+rate[127]*x[6]*n_H;
  J_temp[30*NSPEC+19] = -rate[258]*x[19]*n_H+rate[55]*x[23]*n_H;
  J_temp[31*NSPEC+19] = -rate[44]*x[19]*n_H+rate[46]*x[6]*n_H;
  J_temp[ 0*NSPEC+20] = -rate[68]*x[20]*n_H;
  J_temp[ 1*NSPEC+20] = -rate[77]*x[20]*n_H-rate[178]*x[20]*n_H;
  J_temp[ 2*NSPEC+20] = +rate[190]*x[26]*n_H;
  J_temp[ 3*NSPEC+20] = -rate[107]*x[20]*n_H;
  J_temp[ 4*NSPEC+20] = -rate[23]*x[20]*n_H;
  J_temp[ 5*NSPEC+20] = -rate[52]*x[20]*n_H-rate[170]*x[20]*n_H;
  J_temp[ 7*NSPEC+20] = -rate[104]*x[20]*n_H-rate[187]*x[20]*n_H;
  J_temp[ 8*NSPEC+20] = -rate[105]*x[20]*n_H-rate[188]*x[20]*n_H;
  J_temp[10*NSPEC+20] = -rate[182]*x[20]*n_H;
  J_temp[11*NSPEC+20] = -rate[22]*x[20]*n_H+rate[282];
  J_temp[12*NSPEC+20] = -rate[109]*x[20]*n_H-rate[189]*x[20]*n_H;
  J_temp[13*NSPEC+20] = -rate[111]*x[20]*n_H;
  J_temp[14*NSPEC+20] = -rate[112]*x[20]*n_H-rate[191]*x[20]*n_H+rate[122]*x[16]*n_H;
  J_temp[15*NSPEC+20] = -rate[116]*x[20]*n_H-rate[192]*x[20]*n_H;
  J_temp[16*NSPEC+20] = +rate[1]*x[31]*n_H+2*rate[15]*x[24]*n_H+rate[24]*x[16]*n_H+rate[25]*x[29]*n_H+rate[30]*x[28]*n_H+rate[122]*x[14]*n_H+rate[273]+rate[304];
  J_temp[18*NSPEC+20] = -rate[157]*x[20]*n_H;
  J_temp[19*NSPEC+20] = +rate[223]*x_e*n_H+rate[224]*x_e*n_H;
  J_temp[20*NSPEC+20] = -rate[0]*x[31]*n_H-rate[9]*x[30]*n_H-rate[19]*x[29]*n_H-rate[20]*x[29]*n_H-rate[21]*x[29]*n_H-rate[22]*x[11]*n_H-rate[23]*x[4]*n_H-rate[52]*x[5]*n_H-rate[68]*x[0]*n_H-rate[77]*x[1]*n_H-rate[104]*x[7]*n_H-rate[105]*x[8]*n_H-rate[107]*x[3]*n_H-rate[109]*x[12]*n_H-rate[111]*x[13]*n_H-rate[112]*x[14]*n_H-rate[113]*x[22]*n_H-rate[116]*x[15]*n_H-rate[157]*x[18]*n_H-rate[170]*x[5]*n_H-rate[178]*x[1]*n_H-rate[182]*x[10]*n_H-rate[187]*x[7]*n_H-rate[188]*x[8]*n_H-rate[189]*x[12]*n_H-rate[191]*x[14]*n_H-rate[192]*x[15]*n_H-rate[257]*x[30]*n_H-rate[269]-rate[270]-rate[301]-rate[316]*x[31]*n_H-rate[322]*x[30]*n_H-rate[326]*x[29]*n_H;
  J_temp[21*NSPEC+20] = +rate[278]+rate[307];
  J_temp[22*NSPEC+20] = -rate[113]*x[20]*n_H;
  J_temp[23*NSPEC+20] = +rate[219]*x_e*n_H;
  J_temp[24*NSPEC+20] = +rate[8]*x[30]*n_H+2*rate[15]*x[16]*n_H+rate[16]*x[28]*n_H+rate[251]*x[31]*n_H;
  J_temp[26*NSPEC+20] = +rate[190]*x[2]*n_H;
  J_temp[28*NSPEC+20] = +rate[16]*x[24]*n_H+rate[30]*x[16]*n_H;
  J_temp[29*NSPEC+20] = -rate[19]*x[20]*n_H-rate[20]*x[20]*n_H-rate[21]*x[20]*n_H-rate[326]*x[20]*n_H+rate[25]*x[16]*n_H;
  J_temp[30*NSPEC+20] = -rate[9]*x[20]*n_H-rate[257]*x[20]*n_H-rate[322]*x[20]*n_H+rate[8]*x[24]*n_H;
  J_temp[31*NSPEC+20] = -rate[0]*x[20]*n_H-rate[316]*x[20]*n_H+rate[1]*x[16]*n_H+rate[251]*x[24]*n_H;
  J_temp[ 0*NSPEC+21] = -rate[70]*x[21]*n_H;
  J_temp[ 1*NSPEC+21] = -rate[80]*x[21]*n_H+rate[83]*x[11]*n_H;
  J_temp[ 2*NSPEC+21] = +rate[198]*x[19]*n_H;
  J_temp[ 3*NSPEC+21] = +rate[229]*x_e*n_H;
  J_temp[ 6*NSPEC+21] = +rate[132]*x[11]*n_H+rate[135]*x[17]*n_H+rate[137]*x[27]*n_H+rate[226]*x_e*n_H;
  J_temp[11*NSPEC+21] = +rate[3]*x[31]*n_H+rate[22]*x[20]*n_H+2*rate[28]*x[16]*n_H+rate[36]*x[29]*n_H+rate[39]*x[28]*n_H+rate[83]*x[1]*n_H+rate[132]*x[6]*n_H+rate[136]*x[12]*n_H+rate[138]*x[14]*n_H+rate[280];
  J_temp[12*NSPEC+21] = +rate[136]*x[11]*n_H;
  J_temp[14*NSPEC+21] = +rate[138]*x[11]*n_H;
  J_temp[16*NSPEC+21] = +rate[10]*x[30]*n_H+rate[24]*x[16]*n_H+2*rate[28]*x[11]*n_H+rate[29]*x[28]*n_H;
  J_temp[17*NSPEC+21] = -rate[35]*x[21]*n_H+rate[135]*x[6]*n_H;
  J_temp[18*NSPEC+21] = -rate[159]*x[21]*n_H;
  J_temp[19*NSPEC+21] = +rate[198]*x[2]*n_H+rate[246]*x_e*n_H;
  J_temp[20*NSPEC+21] = +rate[22]*x[11]*n_H+rate[257]*x[30]*n_H;
  J_temp[21*NSPEC+21] = -rate[2]*x[31]*n_H-rate[11]*x[30]*n_H-2*2*rate[32]*x[21]*n_H-rate[33]*x[28]*n_H-rate[34]*x[28]*n_H-rate[35]*x[17]*n_H-rate[70]*x[0]*n_H-rate[80]*x[1]*n_H-rate[159]*x[18]*n_H-rate[275]-rate[276]-rate[278]-rate[305]-rate[306]-rate[307];
  J_temp[27*NSPEC+21] = +rate[137]*x[6]*n_H;
  J_temp[28*NSPEC+21] = -rate[33]*x[21]*n_H-rate[34]*x[21]*n_H+rate[29]*x[16]*n_H+rate[39]*x[11]*n_H;
  J_temp[29*NSPEC+21] = +rate[36]*x[11]*n_H;
  J_temp[30*NSPEC+21] = -rate[11]*x[21]*n_H+rate[10]*x[16]*n_H+rate[257]*x[20]*n_H;
  J_temp[31*NSPEC+21] = -rate[2]*x[21]*n_H+rate[3]*x[11]*n_H;
  J_temp[ 0*NSPEC+22] = +rate[76]*x[27]*n_H;
  J_temp[ 3*NSPEC+22] = +rate[148]*x[27]*n_H;
  J_temp[ 4*NSPEC+22] = +rate[115]*x[26]*n_H+rate[124]*x[23]*n_H;
  J_temp[ 5*NSPEC+22] = +rate[65]*x[27]*n_H;
  J_temp[ 6*NSPEC+22] = +rate[137]*x[27]*n_H;
  J_temp[ 8*NSPEC+22] = +rate[143]*x[27]*n_H;
  J_temp[10*NSPEC+22] = +rate[97]*x[17]*n_H;
  J_temp[11*NSPEC+22] = +rate[138]*x[14]*n_H;
  J_temp[12*NSPEC+22] = +rate[150]*x[27]*n_H;
  J_temp[13*NSPEC+22] = +rate[98]*x[24]*n_H;
  J_temp[14*NSPEC+22] = +rate[66]*x[30]*n_H+rate[112]*x[20]*n_H+rate[122]*x[16]*n_H+rate[138]*x[11]*n_H+rate[144]*x[28]*n_H+rate[151]*x[17]*n_H;
  J_temp[15*NSPEC+22] = +rate[116]*x[20]*n_H;
  J_temp[16*NSPEC+22] = -rate[123]*x[22]*n_H+rate[122]*x[14]*n_H;
  J_temp[17*NSPEC+22] = -rate[152]*x[22]*n_H+rate[97]*x[10]*n_H+rate[110]*x[26]*n_H+rate[151]*x[14]*n_H;
  J_temp[19*NSPEC+22] = +rate[125]*x[29]*n_H;
  J_temp[20*NSPEC+22] = -rate[113]*x[22]*n_H+rate[112]*x[14]*n_H+rate[116]*x[15]*n_H+rate[326]*x[29]*n_H;
  J_temp[22*NSPEC+22] = -rate[99]*x[24]*n_H-rate[113]*x[20]*n_H-rate[123]*x[16]*n_H-rate[145]*x[28]*n_H-rate[152]*x[17]*n_H-rate[239]*x_e*n_H;
  J_temp[23*NSPEC+22] = +rate[117]*x[29]*n_H+rate[124]*x[4]*n_H;
  J_temp[24*NSPEC+22] = -rate[99]*x[22]*n_H+rate[98]*x[13]*n_H;
  J_temp[26*NSPEC+22] = +rate[110]*x[17]*n_H+rate[115]*x[4]*n_H;
  J_temp[27*NSPEC+22] = +rate[65]*x[5]*n_H+rate[76]*x[0]*n_H+rate[137]*x[6]*n_H+rate[143]*x[8]*n_H+rate[148]*x[3]*n_H+rate[150]*x[12]*n_H;
  J_temp[28*NSPEC+22] = -rate[145]*x[22]*n_H+rate[144]*x[14]*n_H;
  J_temp[29*NSPEC+22] = +rate[117]*x[23]*n_H+rate[125]*x[19]*n_H+rate[326]*x[20]*n_H;
  J_temp[30*NSPEC+22] = +rate[66]*x[14]*n_H;
  J_temp[ 0*NSPEC+23] = +rate[68]*x[20]*n_H;
  J_temp[ 1*NSPEC+23] = +rate[82]*x[11]*n_H;
  J_temp[ 3*NSPEC+23] = +rate[107]*x[20]*n_H;
  J_temp[ 4*NSPEC+23] = -rate[124]*x[23]*n_H;
  J_temp[ 5*NSPEC+23] = +rate[52]*x[20]*n_H+rate[171]*x[16]*n_H;
  J_temp[ 7*NSPEC+23] = +rate[193]*x[16]*n_H;
  J_temp[ 8*NSPEC+23] = +rate[105]*x[20]*n_H+rate[194]*x[16]*n_H;
  J_temp[10*NSPEC+23] = +rate[183]*x[16]*n_H+rate[256]*x[30]*n_H;
  J_temp[11*NSPEC+23] = +rate[82]*x[1]*n_H;
  J_temp[12*NSPEC+23] = +rate[109]*x[20]*n_H+rate[195]*x[16]*n_H;
  J_temp[13*NSPEC+23] = +rate[111]*x[20]*n_H;
  J_temp[14*NSPEC+23] = +rate[196]*x[16]*n_H;
  J_temp[15*NSPEC+23] = +rate[197]*x[16]*n_H;
  J_temp[16*NSPEC+23] = +rate[158]*x[18]*n_H+rate[171]*x[5]*n_H+rate[183]*x[10]*n_H+rate[193]*x[7]*n_H+rate[194]*x[8]*n_H+rate[195]*x[12]*n_H+rate[196]*x[14]*n_H+rate[197]*x[15]*n_H+rate[272]+rate[303];
  J_temp[18*NSPEC+23] = +rate[158]*x[16]*n_H;
  J_temp[19*NSPEC+23] = +rate[44]*x[31]*n_H+rate[277];
  J_temp[20*NSPEC+23] = +rate[52]*x[5]*n_H+rate[68]*x[0]*n_H+rate[105]*x[8]*n_H+rate[107]*x[3]*n_H+rate[109]*x[12]*n_H+rate[111]*x[13]*n_H+rate[113]*x[22]*n_H;
  J_temp[22*NSPEC+23] = +rate[113]*x[20]*n_H;
  J_temp[23*NSPEC+23] = -rate[43]*x[31]*n_H-rate[55]*x[30]*n_H-rate[117]*x[29]*n_H-rate[124]*x[4]*n_H-rate[219]*x_e*n_H-rate[220]*x_e*n_H-rate[221]*x_e*n_H-rate[274];
  J_temp[26*NSPEC+23] = +rate[53]*x[30]*n_H;
  J_temp[29*NSPEC+23] = -rate[117]*x[23]*n_H;
  J_temp[30*NSPEC+23] = -rate[55]*x[23]*n_H+rate[53]*x[26]*n_H+rate[256]*x[10]*n_H;
  J_temp[31*NSPEC+23] = -rate[43]*x[23]*n_H+rate[44]*x[19]*n_H;
  J_temp[ 0*NSPEC+24] = -rate[67]*x[24]*n_H;
  J_temp[ 1*NSPEC+24] = -rate[177]*x[24]*n_H+rate[90]*x[27]*n_H;
  J_temp[ 2*NSPEC+24] = +rate[184]*x[10]*n_H;
  J_temp[ 3*NSPEC+24] = -rate[95]*x[24]*n_H;
  J_temp[ 4*NSPEC+24] = -rate[18]*x[24]*n_H;
  J_temp[ 5*NSPEC+24] = -rate[50]*x[24]*n_H;
  J_temp[ 7*NSPEC+24] = -rate[259]*x[24]*n_H;
  J_temp[ 8*NSPEC+24] = -rate[92]*x[24]*n_H;
  J_temp[10*NSPEC+24] = +rate[182]*x[20]*n_H+rate[183]*x[16]*n_H+rate[184]*x[2]*n_H+rate[243]*x_e*n_H+rate[244]*x_e*n_H+rate[245]*x_e*n_H;
  J_temp[12*NSPEC+24] = -rate[96]*x[24]*n_H;
  J_temp[13*NSPEC+24] = -rate[98]*x[24]*n_H;
  J_temp[14*NSPEC+24] = -rate[185]*x[24]*n_H+rate[112]*x[20]*n_H+rate[238]*x_e*n_H;
  J_temp[15*NSPEC+24] = -rate[102]*x[24]*n_H-rate[186]*x[24]*n_H;
  J_temp[16*NSPEC+24] = -rate[15]*x[24]*n_H+rate[183]*x[10]*n_H;
  J_temp[17*NSPEC+24] = +rate[108]*x[26]*n_H;
  J_temp[19*NSPEC+24] = +rate[225]*x_e*n_H;
  J_temp[20*NSPEC+24] = +rate[0]*x[31]*n_H+rate[19]*x[29]*n_H+rate[112]*x[14]*n_H+rate[182]*x[10]*n_H+rate[270]+rate[301]+rate[316]*x[31]*n_H+rate[322]*x[30]*n_H;
  J_temp[22*NSPEC+24] = -rate[99]*x[24]*n_H;
  J_temp[23*NSPEC+24] = +rate[220]*x_e*n_H+rate[221]*x_e*n_H;
  J_temp[24*NSPEC+24] = -rate[8]*x[30]*n_H-rate[15]*x[16]*n_H-rate[16]*x[28]*n_H-rate[17]*x[28]*n_H-rate[18]*x[4]*n_H-rate[50]*x[5]*n_H-rate[67]*x[0]*n_H-rate[92]*x[8]*n_H-rate[95]*x[3]*n_H-rate[96]*x[12]*n_H-rate[98]*x[13]*n_H-rate[99]*x[22]*n_H-rate[102]*x[15]*n_H-rate[177]*x[1]*n_H-rate[185]*x[14]*n_H-rate[186]*x[15]*n_H-rate[251]*x[31]*n_H-rate[255]*x[30]*n_H-rate[259]*x[7]*n_H-rate[260]*x[29]*n_H-rate[261]*x[29]*n_H-rate[268]-rate[297]-rate[300];
  J_temp[26*NSPEC+24] = +rate[108]*x[17]*n_H+rate[218]*x_e*n_H+rate[271];
  J_temp[27*NSPEC+24] = +rate[6]*x[31]*n_H+rate[90]*x[1]*n_H+rate[289]+rate[312];
  J_temp[28*NSPEC+24] = -rate[16]*x[24]*n_H-rate[17]*x[24]*n_H;
  J_temp[29*NSPEC+24] = -rate[260]*x[24]*n_H-rate[261]*x[24]*n_H+rate[19]*x[20]*n_H;
  J_temp[30*NSPEC+24] = -rate[8]*x[24]*n_H-rate[255]*x[24]*n_H+rate[322]*x[20]*n_H;
  J_temp[31*NSPEC+24] = -rate[251]*x[24]*n_H+rate[0]*x[20]*n_H+rate[6]*x[27]*n_H+rate[316]*x[20]*n_H;
  J_temp[ 1*NSPEC+25] = +rate[49]*x[30]*n_H+rate[77]*x[20]*n_H+rate[78]*x[16]*n_H+rate[79]*x[16]*n_H+rate[80]*x[21]*n_H+rate[81]*x[11]*n_H+rate[82]*x[11]*n_H+rate[83]*x[11]*n_H+rate[84]*x[11]*n_H+rate[85]*x[28]*n_H+rate[86]*x[17]*n_H+rate[87]*x[17]*n_H+rate[88]*x[27]*n_H+rate[89]*x[27]*n_H+rate[90]*x[27]*n_H+rate[91]*x[4]*n_H+rate[155]*x[31]*n_H+rate[156]*x[31]*n_H+rate[169]*x[30]*n_H+rate[177]*x[24]*n_H+rate[178]*x[20]*n_H+rate[179]*x[11]*n_H+rate[180]*x[17]*n_H+rate[181]*x[4]*n_H+rate[242]*x_e*n_H;
  J_temp[ 4*NSPEC+25] = +rate[91]*x[1]*n_H+rate[181]*x[1]*n_H;
  J_temp[11*NSPEC+25] = +rate[81]*x[1]*n_H+rate[82]*x[1]*n_H+rate[83]*x[1]*n_H+rate[84]*x[1]*n_H+rate[179]*x[1]*n_H;
  J_temp[16*NSPEC+25] = +rate[78]*x[1]*n_H+rate[79]*x[1]*n_H;
  J_temp[17*NSPEC+25] = +rate[86]*x[1]*n_H+rate[87]*x[1]*n_H+rate[180]*x[1]*n_H;
  J_temp[20*NSPEC+25] = +rate[77]*x[1]*n_H+rate[178]*x[1]*n_H;
  J_temp[21*NSPEC+25] = +rate[80]*x[1]*n_H;
  J_temp[24*NSPEC+25] = +rate[177]*x[1]*n_H;
  J_temp[25*NSPEC+25] = -rate[296];
  J_temp[27*NSPEC+25] = +rate[88]*x[1]*n_H+rate[89]*x[1]*n_H+rate[90]*x[1]*n_H;
  J_temp[28*NSPEC+25] = +rate[85]*x[1]*n_H;
  J_temp[30*NSPEC+25] = +rate[49]*x[1]*n_H+rate[169]*x[1]*n_H;
  J_temp[31*NSPEC+25] = +rate[155]*x[1]*n_H+rate[156]*x[1]*n_H;
  J_temp[ 0*NSPEC+26] = +rate[67]*x[24]*n_H;
  J_temp[ 1*NSPEC+26] = +rate[79]*x[16]*n_H+rate[80]*x[21]*n_H+rate[81]*x[11]*n_H+rate[178]*x[20]*n_H;
  J_temp[ 2*NSPEC+26] = -rate[190]*x[26]*n_H;
  J_temp[ 3*NSPEC+26] = +rate[95]*x[24]*n_H;
  J_temp[ 4*NSPEC+26] = -rate[114]*x[26]*n_H-rate[115]*x[26]*n_H;
  J_temp[ 5*NSPEC+26] = +rate[50]*x[24]*n_H+rate[170]*x[20]*n_H;
  J_temp[ 7*NSPEC+26] = +rate[187]*x[20]*n_H;
  J_temp[ 8*NSPEC+26] = +rate[92]*x[24]*n_H+rate[188]*x[20]*n_H;
  J_temp[10*NSPEC+26] = +rate[51]*x[30]*n_H+rate[182]*x[20]*n_H+rate[252]*x[31]*n_H;
  J_temp[11*NSPEC+26] = +rate[81]*x[1]*n_H;
  J_temp[12*NSPEC+26] = +rate[96]*x[24]*n_H+rate[189]*x[20]*n_H;
  J_temp[14*NSPEC+26] = +rate[191]*x[20]*n_H;
  J_temp[15*NSPEC+26] = +rate[192]*x[20]*n_H;
  J_temp[16*NSPEC+26] = +rate[42]*x[18]*n_H+rate[79]*x[1]*n_H;
  J_temp[17*NSPEC+26] = -rate[108]*x[26]*n_H-rate[110]*x[26]*n_H;
  J_temp[18*NSPEC+26] = +rate[42]*x[16]*n_H+rate[157]*x[20]*n_H;
  J_temp[19*NSPEC+26] = +rate[279];
  J_temp[20*NSPEC+26] = +rate[157]*x[18]*n_H+rate[170]*x[5]*n_H+rate[178]*x[1]*n_H+rate[182]*x[10]*n_H+rate[187]*x[7]*n_H+rate[188]*x[8]*n_H+rate[189]*x[12]*n_H+rate[191]*x[14]*n_H+rate[192]*x[15]*n_H+rate[269];
  J_temp[21*NSPEC+26] = +rate[80]*x[1]*n_H;
  J_temp[22*NSPEC+26] = +rate[99]*x[24]*n_H;
  J_temp[23*NSPEC+26] = +rate[43]*x[31]*n_H+rate[274];
  J_temp[24*NSPEC+26] = +rate[50]*x[5]*n_H+rate[67]*x[0]*n_H+rate[92]*x[8]*n_H+rate[95]*x[3]*n_H+rate[96]*x[12]*n_H+rate[99]*x[22]*n_H;
  J_temp[26*NSPEC+26] = -rate[41]*x[31]*n_H-rate[53]*x[30]*n_H-rate[103]*x[29]*n_H-rate[106]*x[28]*n_H-rate[108]*x[17]*n_H-rate[110]*x[17]*n_H-rate[114]*x[4]*n_H-rate[115]*x[4]*n_H-rate[190]*x[2]*n_H-rate[218]*x_e*n_H-rate[271]-rate[302];
  J_temp[28*NSPEC+26] = -rate[106]*x[26]*n_H;
  J_temp[29*NSPEC+26] = -rate[103]*x[26]*n_H;
  J_temp[30*NSPEC+26] = -rate[53]*x[26]*n_H+rate[51]*x[10]*n_H;
  J_temp[31*NSPEC+26] = -rate[41]*x[26]*n_H+rate[43]*x[23]*n_H+rate[252]*x[10]*n_H;
  J_temp[ 0*NSPEC+27] = -rate[76]*x[27]*n_H;
  J_temp[ 1*NSPEC+27] = -rate[88]*x[27]*n_H-rate[89]*x[27]*n_H-rate[90]*x[27]*n_H;
  J_temp[ 3*NSPEC+27] = -rate[148]*x[27]*n_H;
  J_temp[ 4*NSPEC+27] = +rate[18]*x[24]*n_H+rate[23]*x[20]*n_H+rate[31]*x[16]*n_H+rate[101]*x[10]*n_H+rate[214]*x[14]*n_H;
  J_temp[ 5*NSPEC+27] = -rate[65]*x[27]*n_H-rate[175]*x[27]*n_H;
  J_temp[ 6*NSPEC+27] = -rate[137]*x[27]*n_H;
  J_temp[ 7*NSPEC+27] = -rate[203]*x[27]*n_H;
  J_temp[ 8*NSPEC+27] = -rate[143]*x[27]*n_H;
  J_temp[10*NSPEC+27] = +rate[94]*x[28]*n_H+rate[101]*x[4]*n_H;
  J_temp[11*NSPEC+27] = +rate[205]*x[14]*n_H;
  J_temp[12*NSPEC+27] = -rate[150]*x[27]*n_H;
  J_temp[14*NSPEC+27] = +rate[167]*x[31]*n_H+rate[185]*x[24]*n_H+rate[191]*x[20]*n_H+rate[196]*x[16]*n_H+rate[202]*x[29]*n_H+rate[205]*x[11]*n_H+rate[208]*x[28]*n_H+rate[211]*x[17]*n_H+rate[214]*x[4]*n_H;
  J_temp[16*NSPEC+27] = +rate[26]*x[29]*n_H+rate[27]*x[29]*n_H+rate[31]*x[4]*n_H+rate[123]*x[22]*n_H+rate[196]*x[14]*n_H;
  J_temp[17*NSPEC+27] = +rate[152]*x[22]*n_H+rate[211]*x[14]*n_H;
  J_temp[20*NSPEC+27] = +rate[20]*x[29]*n_H+rate[21]*x[29]*n_H+rate[23]*x[4]*n_H+rate[113]*x[22]*n_H+rate[191]*x[14]*n_H;
  J_temp[22*NSPEC+27] = +rate[99]*x[24]*n_H+rate[113]*x[20]*n_H+rate[123]*x[16]*n_H+rate[145]*x[28]*n_H+rate[152]*x[17]*n_H+rate[239]*x_e*n_H;
  J_temp[24*NSPEC+27] = +rate[17]*x[28]*n_H+rate[18]*x[4]*n_H+rate[99]*x[22]*n_H+rate[185]*x[14]*n_H+rate[260]*x[29]*n_H+rate[261]*x[29]*n_H;
  J_temp[27*NSPEC+27] = -rate[6]*x[31]*n_H-rate[65]*x[5]*n_H-rate[76]*x[0]*n_H-rate[88]*x[1]*n_H-rate[89]*x[1]*n_H-rate[90]*x[1]*n_H-rate[137]*x[6]*n_H-rate[143]*x[8]*n_H-rate[148]*x[3]*n_H-rate[150]*x[12]*n_H-rate[175]*x[5]*n_H-rate[203]*x[7]*n_H-rate[289]-rate[299]-rate[312];
  J_temp[28*NSPEC+27] = +rate[17]*x[24]*n_H+rate[94]*x[10]*n_H+rate[145]*x[22]*n_H+rate[208]*x[14]*n_H;
  J_temp[29*NSPEC+27] = +rate[20]*x[20]*n_H+rate[21]*x[20]*n_H+rate[26]*x[16]*n_H+rate[27]*x[16]*n_H+rate[202]*x[14]*n_H+rate[260]*x[24]*n_H+rate[261]*x[24]*n_H;
  J_temp[31*NSPEC+27] = -rate[6]*x[27]*n_H+rate[167]*x[14]*n_H;
  J_temp[ 0*NSPEC+28] = -rate[73]*x[28]*n_H;
  J_temp[ 1*NSPEC+28] = -rate[85]*x[28]*n_H+rate[86]*x[17]*n_H;
  J_temp[ 3*NSPEC+28] = -rate[140]*x[28]*n_H;
  J_temp[ 4*NSPEC+28] = +rate[7]*x[31]*n_H+2*rate[14]*x[30]*n_H+rate[23]*x[20]*n_H+rate[114]*x[26]*n_H+rate[124]*x[23]*n_H+rate[209]*x[8]*n_H;
  J_temp[ 5*NSPEC+28] = -rate[61]*x[28]*n_H-rate[173]*x[28]*n_H;
  J_temp[ 6*NSPEC+28] = +rate[127]*x[29]*n_H;
  J_temp[ 7*NSPEC+28] = -rate[128]*x[28]*n_H-rate[200]*x[28]*n_H+rate[126]*x[11]*n_H;
  J_temp[ 8*NSPEC+28] = -rate[139]*x[28]*n_H+rate[188]*x[20]*n_H+rate[194]*x[16]*n_H+rate[207]*x[17]*n_H+rate[209]*x[4]*n_H;
  J_temp[10*NSPEC+28] = -rate[93]*x[28]*n_H-rate[94]*x[28]*n_H;
  J_temp[11*NSPEC+28] = -rate[39]*x[28]*n_H+rate[36]*x[29]*n_H+rate[126]*x[7]*n_H;
  J_temp[12*NSPEC+28] = -rate[142]*x[28]*n_H+rate[96]*x[24]*n_H+rate[109]*x[20]*n_H+rate[120]*x[16]*n_H+rate[149]*x[17]*n_H+rate[150]*x[27]*n_H+rate[233]*x_e*n_H;
  J_temp[13*NSPEC+28] = +rate[235]*x_e*n_H+rate[236]*x_e*n_H;
  J_temp[14*NSPEC+28] = -rate[144]*x[28]*n_H-rate[208]*x[28]*n_H+rate[151]*x[17]*n_H;
  J_temp[16*NSPEC+28] = -rate[29]*x[28]*n_H-rate[30]*x[28]*n_H+rate[25]*x[29]*n_H+rate[120]*x[12]*n_H+rate[194]*x[8]*n_H;
  J_temp[17*NSPEC+28] = +rate[5]*x[31]*n_H+rate[35]*x[21]*n_H+2*rate[38]*x[29]*n_H+rate[86]*x[1]*n_H+rate[149]*x[12]*n_H+rate[151]*x[14]*n_H+rate[207]*x[8]*n_H+rate[286]+rate[310]+rate[318]*x[31]*n_H+rate[324]*x[30]*n_H;
  J_temp[18*NSPEC+28] = -rate[164]*x[28]*n_H;
  J_temp[20*NSPEC+28] = +rate[19]*x[29]*n_H+rate[23]*x[4]*n_H+rate[109]*x[12]*n_H+rate[188]*x[8]*n_H;
  J_temp[21*NSPEC+28] = -rate[33]*x[28]*n_H-rate[34]*x[28]*n_H+rate[35]*x[17]*n_H;
  J_temp[22*NSPEC+28] = -rate[145]*x[28]*n_H;
  J_temp[23*NSPEC+28] = +rate[124]*x[4]*n_H;
  J_temp[24*NSPEC+28] = -rate[16]*x[28]*n_H-rate[17]*x[28]*n_H+rate[96]*x[12]*n_H;
  J_temp[26*NSPEC+28] = -rate[106]*x[28]*n_H+rate[114]*x[4]*n_H;
  J_temp[27*NSPEC+28] = +rate[6]*x[31]*n_H+rate[150]*x[12]*n_H;
  J_temp[28*NSPEC+28] = -rate[4]*x[31]*n_H-rate[13]*x[30]*n_H-rate[16]*x[24]*n_H-rate[17]*x[24]*n_H-rate[29]*x[16]*n_H-rate[30]*x[16]*n_H-rate[33]*x[21]*n_H-rate[34]*x[21]*n_H-rate[37]*x[29]*n_H-rate[39]*x[11]*n_H-2*2*rate[40]*x[28]*n_H-rate[61]*x[5]*n_H-rate[73]*x[0]*n_H-rate[85]*x[1]*n_H-rate[93]*x[10]*n_H-rate[94]*x[10]*n_H-rate[106]*x[26]*n_H-rate[128]*x[7]*n_H-rate[139]*x[8]*n_H-rate[140]*x[3]*n_H-rate[142]*x[12]*n_H-rate[144]*x[14]*n_H-rate[145]*x[22]*n_H-rate[164]*x[18]*n_H-rate[173]*x[5]*n_H-rate[200]*x[7]*n_H-rate[208]*x[14]*n_H-rate[254]*x[31]*n_H-rate[283]-rate[284]-rate[309]-rate[317]*x[31]*n_H-rate[323]*x[30]*n_H;
  J_temp[29*NSPEC+28] = -rate[37]*x[28]*n_H+rate[12]*x[30]*n_H+rate[19]*x[20]*n_H+rate[25]*x[16]*n_H+rate[36]*x[11]*n_H+2*rate[38]*x[17]*n_H+rate[127]*x[6]*n_H+rate[253]*x[31]*n_H;
  J_temp[30*NSPEC+28] = -rate[13]*x[28]*n_H-rate[323]*x[28]*n_H+rate[12]*x[29]*n_H+2*rate[14]*x[4]*n_H+rate[324]*x[17]*n_H;
  J_temp[31*NSPEC+28] = -rate[4]*x[28]*n_H-rate[254]*x[28]*n_H-rate[317]*x[28]*n_H+rate[5]*x[17]*n_H+rate[6]*x[27]*n_H+rate[7]*x[4]*n_H+rate[253]*x[29]*n_H+rate[318]*x[17]*n_H;
  J_temp[ 0*NSPEC+29] = -rate[71]*x[29]*n_H;
  J_temp[ 1*NSPEC+29] = +rate[88]*x[27]*n_H+rate[89]*x[27]*n_H+rate[91]*x[4]*n_H;
  J_temp[ 3*NSPEC+29] = -rate[130]*x[29]*n_H;
  J_temp[ 4*NSPEC+29] = +rate[7]*x[31]*n_H+rate[18]*x[24]*n_H+rate[91]*x[1]*n_H+rate[100]*x[10]*n_H+rate[115]*x[26]*n_H+rate[204]*x[7]*n_H+2*rate[290]+2*rate[313]+2*rate[319]*x[31]*n_H+2*rate[325]*x[30]*n_H;
  J_temp[ 5*NSPEC+29] = -rate[56]*x[29]*n_H;
  J_temp[ 6*NSPEC+29] = -rate[127]*x[29]*n_H;
  J_temp[ 7*NSPEC+29] = +rate[160]*x[31]*n_H+rate[187]*x[20]*n_H+rate[193]*x[16]*n_H+rate[199]*x[11]*n_H+rate[200]*x[28]*n_H+rate[201]*x[17]*n_H+rate[203]*x[27]*n_H+rate[204]*x[4]*n_H+rate[247]*x_e*n_H;
  J_temp[ 8*NSPEC+29] = -rate[129]*x[29]*n_H+rate[92]*x[24]*n_H+rate[105]*x[20]*n_H+rate[118]*x[16]*n_H+rate[133]*x[11]*n_H+rate[139]*x[28]*n_H+rate[141]*x[17]*n_H+rate[143]*x[27]*n_H+rate[228]*x_e*n_H+rate[285];
  J_temp[10*NSPEC+29] = -rate[262]*x[29]*n_H-rate[263]*x[29]*n_H+rate[100]*x[4]*n_H;
  J_temp[11*NSPEC+29] = -rate[36]*x[29]*n_H+rate[133]*x[8]*n_H+rate[199]*x[7]*n_H;
  J_temp[12*NSPEC+29] = -rate[131]*x[29]*n_H+rate[142]*x[28]*n_H+rate[231]*x_e*n_H+rate[232]*x_e*n_H;
  J_temp[13*NSPEC+29] = +rate[234]*x_e*n_H;
  J_temp[14*NSPEC+29] = -rate[202]*x[29]*n_H+rate[144]*x[28]*n_H+rate[238]*x_e*n_H;
  J_temp[15*NSPEC+29] = +rate[102]*x[24]*n_H+rate[116]*x[20]*n_H+2*rate[240]*x_e*n_H;
  J_temp[16*NSPEC+29] = -rate[25]*x[29]*n_H-rate[26]*x[29]*n_H-rate[27]*x[29]*n_H+rate[29]*x[28]*n_H+rate[118]*x[8]*n_H+rate[193]*x[7]*n_H;
  J_temp[17*NSPEC+29] = -rate[38]*x[29]*n_H+rate[141]*x[8]*n_H+rate[201]*x[7]*n_H;
  J_temp[18*NSPEC+29] = -rate[161]*x[29]*n_H-rate[162]*x[29]*n_H;
  J_temp[19*NSPEC+29] = -rate[125]*x[29]*n_H;
  J_temp[20*NSPEC+29] = -rate[19]*x[29]*n_H-rate[20]*x[29]*n_H-rate[21]*x[29]*n_H-rate[326]*x[29]*n_H+rate[105]*x[8]*n_H+rate[116]*x[15]*n_H+rate[187]*x[7]*n_H;
  J_temp[21*NSPEC+29] = +rate[33]*x[28]*n_H;
  J_temp[23*NSPEC+29] = -rate[117]*x[29]*n_H;
  J_temp[24*NSPEC+29] = -rate[260]*x[29]*n_H-rate[261]*x[29]*n_H+rate[16]*x[28]*n_H+rate[18]*x[4]*n_H+rate[92]*x[8]*n_H+rate[102]*x[15]*n_H;
  J_temp[26*NSPEC+29] = -rate[103]*x[29]*n_H+rate[115]*x[4]*n_H;
  J_temp[27*NSPEC+29] = +rate[88]*x[1]*n_H+rate[89]*x[1]*n_H+rate[143]*x[8]*n_H+rate[203]*x[7]*n_H+rate[289]+rate[312];
  J_temp[28*NSPEC+29] = -rate[37]*x[29]*n_H+rate[4]*x[31]*n_H+rate[16]*x[24]*n_H+rate[29]*x[16]*n_H+rate[33]*x[21]*n_H+rate[40]*x[28]*n_H+rate[139]*x[8]*n_H+rate[142]*x[12]*n_H+rate[144]*x[14]*n_H+rate[200]*x[7]*n_H+rate[284]+rate[309]+rate[317]*x[31]*n_H+rate[323]*x[30]*n_H;
  J_temp[29*NSPEC+29] = -rate[12]*x[30]*n_H-rate[19]*x[20]*n_H-rate[20]*x[20]*n_H-rate[21]*x[20]*n_H-rate[25]*x[16]*n_H-rate[26]*x[16]*n_H-rate[27]*x[16]*n_H-rate[36]*x[11]*n_H-rate[37]*x[28]*n_H-rate[38]*x[17]*n_H-rate[56]*x[5]*n_H-rate[71]*x[0]*n_H-rate[103]*x[26]*n_H-rate[117]*x[23]*n_H-rate[125]*x[19]*n_H-rate[127]*x[6]*n_H-rate[129]*x[8]*n_H-rate[130]*x[3]*n_H-rate[131]*x[12]*n_H-rate[161]*x[18]*n_H-rate[162]*x[18]*n_H-rate[202]*x[14]*n_H-rate[253]*x[31]*n_H-rate[260]*x[24]*n_H-rate[261]*x[24]*n_H-rate[262]*x[10]*n_H-rate[263]*x[10]*n_H-2*2*rate[264]*x[29]*n_H-rate[298]-rate[326]*x[20]*n_H;
  J_temp[30*NSPEC+29] = -rate[12]*x[29]*n_H+rate[323]*x[28]*n_H+2*rate[325]*x[4]*n_H;
  J_temp[31*NSPEC+29] = -rate[253]*x[29]*n_H+rate[4]*x[28]*n_H+rate[7]*x[4]*n_H+rate[160]*x[7]*n_H+rate[317]*x[28]*n_H+2*rate[319]*x[4]*n_H;
  J_temp[ 0*NSPEC+30] = +rate[67]*x[24]*n_H+rate[68]*x[20]*n_H+rate[69]*x[16]*n_H+rate[70]*x[21]*n_H+rate[71]*x[29]*n_H+rate[72]*x[11]*n_H+rate[73]*x[28]*n_H+rate[74]*x[17]*n_H+rate[75]*x[2]*n_H+rate[76]*x[27]*n_H+rate[217]*x_e*n_H+rate[267];
  J_temp[ 1*NSPEC+30] = -rate[49]*x[30]*n_H-rate[169]*x[30]*n_H+rate[78]*x[16]*n_H+rate[80]*x[21]*n_H+rate[81]*x[11]*n_H+rate[82]*x[11]*n_H;
  J_temp[ 2*NSPEC+30] = +rate[75]*x[0]*n_H;
  J_temp[ 3*NSPEC+30] = +rate[47]*x[31]*n_H+rate[229]*x_e*n_H;
  J_temp[ 4*NSPEC+30] = -rate[14]*x[30]*n_H-rate[325]*x[30]*n_H+rate[176]*x[5]*n_H+rate[325]*x[30]*n_H;
  J_temp[ 5*NSPEC+30] = -rate[48]*x[30]*n_H+rate[60]*x[11]*n_H+rate[154]*x[31]*n_H+rate[170]*x[20]*n_H+rate[171]*x[16]*n_H+rate[172]*x[11]*n_H+rate[173]*x[28]*n_H+rate[174]*x[17]*n_H+rate[175]*x[27]*n_H+rate[176]*x[4]*n_H;
  J_temp[ 6*NSPEC+30] = -rate[59]*x[30]*n_H+rate[46]*x[31]*n_H;
  J_temp[ 7*NSPEC+30] = -rate[57]*x[30]*n_H;
  J_temp[ 8*NSPEC+30] = -rate[62]*x[30]*n_H;
  J_temp[10*NSPEC+30] = -rate[51]*x[30]*n_H-rate[256]*x[30]*n_H;
  J_temp[11*NSPEC+30] = +rate[3]*x[31]*n_H+rate[45]*x[18]*n_H+rate[60]*x[5]*n_H+rate[72]*x[0]*n_H+rate[81]*x[1]*n_H+rate[82]*x[1]*n_H+rate[172]*x[5]*n_H+rate[281]+rate[282]+rate[308];
  J_temp[12*NSPEC+30] = -rate[64]*x[30]*n_H+rate[131]*x[29]*n_H+rate[232]*x_e*n_H;
  J_temp[13*NSPEC+30] = +rate[98]*x[24]*n_H+rate[234]*x_e*n_H+rate[236]*x_e*n_H;
  J_temp[14*NSPEC+30] = -rate[66]*x[30]*n_H;
  J_temp[16*NSPEC+30] = -rate[10]*x[30]*n_H+rate[1]*x[31]*n_H+rate[27]*x[29]*n_H+rate[42]*x[18]*n_H+rate[69]*x[0]*n_H+rate[78]*x[1]*n_H+rate[171]*x[5]*n_H;
  J_temp[17*NSPEC+30] = -rate[324]*x[30]*n_H+rate[5]*x[31]*n_H+rate[74]*x[0]*n_H+rate[110]*x[26]*n_H+rate[174]*x[5]*n_H+rate[324]*x[30]*n_H;
  J_temp[18*NSPEC+30] = -rate[153]*x[30]*n_H+rate[42]*x[16]*n_H+rate[45]*x[11]*n_H;
  J_temp[19*NSPEC+30] = -rate[258]*x[30]*n_H+rate[44]*x[31]*n_H+rate[125]*x[29]*n_H+rate[223]*x_e*n_H+rate[225]*x_e*n_H+rate[279];
  J_temp[20*NSPEC+30] = -rate[9]*x[30]*n_H-rate[257]*x[30]*n_H-rate[322]*x[30]*n_H+rate[0]*x[31]*n_H+rate[68]*x[0]*n_H+rate[170]*x[5]*n_H+rate[322]*x[30]*n_H;
  J_temp[21*NSPEC+30] = -rate[11]*x[30]*n_H+rate[2]*x[31]*n_H+rate[70]*x[0]*n_H+rate[80]*x[1]*n_H+rate[278]+rate[307];
  J_temp[23*NSPEC+30] = -rate[55]*x[30]*n_H+rate[43]*x[31]*n_H+rate[221]*x_e*n_H;
  J_temp[24*NSPEC+30] = -rate[8]*x[30]*n_H-rate[255]*x[30]*n_H+rate[67]*x[0]*n_H+rate[98]*x[13]*n_H;
  J_temp[26*NSPEC+30] = -rate[53]*x[30]*n_H+rate[41]*x[31]*n_H+rate[106]*x[28]*n_H+rate[110]*x[17]*n_H;
  J_temp[27*NSPEC+30] = +rate[76]*x[0]*n_H+rate[175]*x[5]*n_H;
  J_temp[28*NSPEC+30] = -rate[13]*x[30]*n_H-rate[323]*x[30]*n_H+rate[4]*x[31]*n_H+rate[73]*x[0]*n_H+rate[106]*x[26]*n_H+rate[173]*x[5]*n_H+rate[323]*x[30]*n_H;
  J_temp[29*NSPEC+30] = -rate[12]*x[30]*n_H+rate[27]*x[16]*n_H+rate[71]*x[0]*n_H+rate[125]*x[19]*n_H+rate[131]*x[12]*n_H;
  J_temp[30*NSPEC+30] = -rate[8]*x[24]*n_H-rate[9]*x[20]*n_H-rate[10]*x[16]*n_H-rate[11]*x[21]*n_H-rate[12]*x[29]*n_H-rate[13]*x[28]*n_H-rate[14]*x[4]*n_H-rate[48]*x[5]*n_H-rate[49]*x[1]*n_H-rate[51]*x[10]*n_H-rate[53]*x[26]*n_H-rate[55]*x[23]*n_H-rate[57]*x[7]*n_H-rate[59]*x[6]*n_H-rate[62]*x[8]*n_H-rate[64]*x[12]*n_H-rate[66]*x[14]*n_H-rate[153]*x[18]*n_H-rate[169]*x[1]*n_H-rate[255]*x[24]*n_H-rate[256]*x[10]*n_H-rate[257]*x[20]*n_H-rate[258]*x[19]*n_H-rate[293]-rate[294]-rate[295]-rate[315]*x[31]*n_H-rate[320]*x_e*n_H-2*2*rate[321]*x[30]*n_H-rate[322]*x[20]*n_H-rate[323]*x[28]*n_H-rate[324]*x[17]*n_H-rate[325]*x[4]*n_H-rate[328]+rate[321]*x[30]*n_H+rate[322]*x[20]*n_H+rate[323]*x[28]*n_H+rate[324]*x[17]*n_H+rate[325]*x[4]*n_H;
  J_temp[31*NSPEC+30] = -rate[315]*x[30]*n_H+rate[0]*x[20]*n_H+rate[1]*x[16]*n_H+rate[2]*x[21]*n_H+rate[3]*x[11]*n_H+rate[4]*x[28]*n_H+rate[5]*x[17]*n_H+rate[41]*x[26]*n_H+rate[43]*x[23]*n_H+rate[44]*x[19]*n_H+rate[46]*x[6]*n_H+rate[47]*x[3]*n_H+rate[154]*x[5]*n_H+rate[327]*n_H;
  J_temp[ 0*NSPEC+31] = +rate[75]*x[2]*n_H+3*rate[216]*x_e*n_H+rate[217]*x_e*n_H+rate[266];
  J_temp[ 1*NSPEC+31] = -rate[155]*x[31]*n_H-rate[156]*x[31]*n_H+rate[49]*x[30]*n_H+rate[77]*x[20]*n_H+rate[79]*x[16]*n_H+rate[81]*x[11]*n_H+rate[84]*x[11]*n_H+rate[85]*x[28]*n_H+rate[87]*x[17]*n_H;
  J_temp[ 2*NSPEC+31] = +rate[75]*x[0]*n_H+rate[147]*x[3]*n_H+rate[166]*x[18]*n_H;
  J_temp[ 3*NSPEC+31] = -rate[47]*x[31]*n_H+rate[147]*x[2]*n_H+rate[230]*x_e*n_H;
  J_temp[ 4*NSPEC+31] = -rate[7]*x[31]*n_H-rate[319]*x[31]*n_H+rate[168]*x[18]*n_H+rate[319]*x[31]*n_H;
  J_temp[ 5*NSPEC+31] = -rate[154]*x[31]*n_H+rate[48]*x[30]*n_H+rate[50]*x[24]*n_H+rate[52]*x[20]*n_H+rate[54]*x[16]*n_H+rate[56]*x[29]*n_H+rate[58]*x[11]*n_H+rate[60]*x[11]*n_H+rate[61]*x[28]*n_H+rate[63]*x[17]*n_H+rate[65]*x[27]*n_H+2*rate[215]*x_e*n_H+rate[265];
  J_temp[ 6*NSPEC+31] = -rate[46]*x[31]*n_H+rate[59]*x[30]*n_H+rate[226]*x_e*n_H+2*rate[227]*x_e*n_H;
  J_temp[ 7*NSPEC+31] = -rate[160]*x[31]*n_H+rate[57]*x[30]*n_H+rate[104]*x[20]*n_H+rate[128]*x[28]*n_H;
  J_temp[ 8*NSPEC+31] = +rate[62]*x[30]*n_H+rate[129]*x[29]*n_H+rate[228]*x_e*n_H;
  J_temp[10*NSPEC+31] = -rate[252]*x[31]*n_H+rate[51]*x[30]*n_H+rate[93]*x[28]*n_H+rate[97]*x[17]*n_H;
  J_temp[11*NSPEC+31] = -rate[3]*x[31]*n_H+rate[58]*x[5]*n_H+rate[60]*x[5]*n_H+rate[81]*x[1]*n_H+rate[84]*x[1]*n_H+rate[163]*x[18]*n_H+rate[280]+rate[282];
  J_temp[12*NSPEC+31] = +rate[64]*x[30]*n_H+2*rate[231]*x_e*n_H+rate[233]*x_e*n_H;
  J_temp[13*NSPEC+31] = +rate[234]*x_e*n_H+2*rate[235]*x_e*n_H+rate[237]*x_e*n_H;
  J_temp[14*NSPEC+31] = -rate[167]*x[31]*n_H+rate[66]*x[30]*n_H;
  J_temp[16*NSPEC+31] = -rate[1]*x[31]*n_H+rate[10]*x[30]*n_H+2*rate[26]*x[29]*n_H+rate[54]*x[5]*n_H+rate[79]*x[1]*n_H+rate[158]*x[18]*n_H+rate[273]+rate[304];
  J_temp[17*NSPEC+31] = -rate[5]*x[31]*n_H-rate[318]*x[31]*n_H+rate[63]*x[5]*n_H+rate[87]*x[1]*n_H+rate[97]*x[10]*n_H+rate[165]*x[18]*n_H+rate[286]+rate[310]+2*rate[318]*x[31]*n_H+rate[324]*x[30]*n_H;
  J_temp[18*NSPEC+31] = -rate[249]*x[31]*n_H-rate[250]*x[31]*n_H+rate[153]*x[30]*n_H+rate[157]*x[20]*n_H+rate[158]*x[16]*n_H+rate[159]*x[21]*n_H+rate[161]*x[29]*n_H+rate[162]*x[29]*n_H+rate[163]*x[11]*n_H+rate[164]*x[28]*n_H+rate[165]*x[17]*n_H+rate[166]*x[2]*n_H+rate[168]*x[4]*n_H+rate[241]*x_e*n_H;
  J_temp[19*NSPEC+31] = -rate[44]*x[31]*n_H+rate[222]*x_e*n_H+2*rate[224]*x_e*n_H+rate[225]*x_e*n_H+rate[277];
  J_temp[20*NSPEC+31] = -rate[0]*x[31]*n_H-rate[316]*x[31]*n_H+rate[9]*x[30]*n_H+rate[20]*x[29]*n_H+rate[21]*x[29]*n_H+rate[52]*x[5]*n_H+rate[77]*x[1]*n_H+rate[104]*x[7]*n_H+rate[157]*x[18]*n_H+rate[270]+rate[301]+2*rate[316]*x[31]*n_H+rate[322]*x[30]*n_H;
  J_temp[21*NSPEC+31] = -rate[2]*x[31]*n_H+rate[11]*x[30]*n_H+rate[159]*x[18]*n_H+rate[276]+rate[306];
  J_temp[22*NSPEC+31] = +rate[239]*x_e*n_H;
  J_temp[23*NSPEC+31] = -rate[43]*x[31]*n_H+rate[55]*x[30]*n_H+rate[117]*x[29]*n_H+rate[219]*x_e*n_H+2*rate[220]*x_e*n_H+rate[274];
  J_temp[24*NSPEC+31] = -rate[251]*x[31]*n_H+rate[8]*x[30]*n_H+rate[17]*x[28]*n_H+rate[50]*x[5]*n_H;
  J_temp[26*NSPEC+31] = -rate[41]*x[31]*n_H+rate[53]*x[30]*n_H+rate[103]*x[29]*n_H+rate[218]*x_e*n_H+rate[302];
  J_temp[27*NSPEC+31] = -rate[6]*x[31]*n_H+rate[65]*x[5]*n_H;
  J_temp[28*NSPEC+31] = -rate[4]*x[31]*n_H-rate[254]*x[31]*n_H-rate[317]*x[31]*n_H+rate[13]*x[30]*n_H+rate[17]*x[24]*n_H+rate[37]*x[29]*n_H+rate[61]*x[5]*n_H+rate[85]*x[1]*n_H+rate[93]*x[10]*n_H+rate[128]*x[7]*n_H+rate[164]*x[18]*n_H+rate[284]+rate[309]+2*rate[317]*x[31]*n_H+rate[323]*x[30]*n_H;
  J_temp[29*NSPEC+31] = -rate[253]*x[31]*n_H+rate[12]*x[30]*n_H+rate[20]*x[20]*n_H+rate[21]*x[20]*n_H+2*rate[26]*x[16]*n_H+rate[37]*x[28]*n_H+rate[56]*x[5]*n_H+rate[103]*x[26]*n_H+rate[117]*x[23]*n_H+rate[129]*x[8]*n_H+rate[161]*x[18]*n_H+rate[162]*x[18]*n_H;
  J_temp[30*NSPEC+31] = -rate[315]*x[31]*n_H+rate[8]*x[24]*n_H+rate[9]*x[20]*n_H+rate[10]*x[16]*n_H+rate[11]*x[21]*n_H+rate[12]*x[29]*n_H+rate[13]*x[28]*n_H+rate[48]*x[5]*n_H+rate[49]*x[1]*n_H+rate[51]*x[10]*n_H+rate[53]*x[26]*n_H+rate[55]*x[23]*n_H+rate[57]*x[7]*n_H+rate[59]*x[6]*n_H+rate[62]*x[8]*n_H+rate[64]*x[12]*n_H+rate[66]*x[14]*n_H+rate[153]*x[18]*n_H+rate[293]+2*rate[294]+3*rate[315]*x[31]*n_H+2*rate[320]*x_e*n_H+2*rate[321]*x[30]*n_H+rate[322]*x[20]*n_H+rate[323]*x[28]*n_H+rate[324]*x[17]*n_H+2*rate[328];
  J_temp[31*NSPEC+31] = -rate[0]*x[20]*n_H-rate[1]*x[16]*n_H-rate[2]*x[21]*n_H-rate[3]*x[11]*n_H-rate[4]*x[28]*n_H-rate[5]*x[17]*n_H-rate[6]*x[27]*n_H-rate[7]*x[4]*n_H-rate[41]*x[26]*n_H-rate[43]*x[23]*n_H-rate[44]*x[19]*n_H-rate[46]*x[6]*n_H-rate[47]*x[3]*n_H-rate[154]*x[5]*n_H-rate[155]*x[1]*n_H-rate[156]*x[1]*n_H-rate[160]*x[7]*n_H-rate[167]*x[14]*n_H-rate[249]*x[18]*n_H-rate[250]*x[18]*n_H-rate[251]*x[24]*n_H-rate[252]*x[10]*n_H-rate[253]*x[29]*n_H-rate[254]*x[28]*n_H-rate[292]-rate[315]*x[30]*n_H-rate[316]*x[20]*n_H-rate[317]*x[28]*n_H-rate[318]*x[17]*n_H-rate[319]*x[4]*n_H-2*rate[327]*n_H+3*rate[315]*x[30]*n_H+2*rate[316]*x[20]*n_H+2*rate[317]*x[28]*n_H+2*rate[318]*x[17]*n_H+rate[319]*x[4]*n_H;

  // IJth(J,1,1) = RCONST(-0.04);
  // IJth(J,1,2) = RCONST(1.0e4)*y3;
  // IJth(J,1,3) = RCONST(1.0e4)*y2;
  // IJth(J,2,1) = RCONST(0.04); 
  // IJth(J,2,2) = RCONST(-1.0e4)*y3-RCONST(6.0e7)*y2;
  // IJth(J,2,3) = RCONST(-1.0e4)*y2;
  // IJth(J,3,2) = RCONST(6.0e7)*y2;

  for (i=0; i<NSPEC; i++){
 
    for (j=0; j<NSPEC; j++){
 
      IJth(J,i+1,j+1) = J_temp[i*NSPEC+j];
    }
  }


  return(0);
}

/*-----------------------------------------------------------------------------------------------*/
