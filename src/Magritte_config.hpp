#define NCELLS ncells 

#define NSPEC 35 

#define NREAC 329 

#define TOT_NLEV 56 

#define TOT_NRAD 63 

#define MAX_NLEV 41 

#define MAX_NRAD 40 

#define TOT_NLEV2 1756 

#define TOT_NCOLPAR 18 

#define TOT_CUM_TOT_NCOLTRAN 1800 

#define TOT_CUM_TOT_NCOLTEMP 384 

#define TOT_CUM_TOT_NCOLTRANTEMP 44340 

#define NLEV { 5, 5, 5, 41 } 

#define NRAD { 7, 9, 7, 40 } 

#define CUM_NLEV { 0, 5, 10, 15 } 

#define CUM_NLEV2 { 0, 25, 50, 75 } 

#define CUM_NRAD { 0, 7, 16, 23 } 

#define NCOLPAR { 6, 4, 6, 2 } 

#define CUM_NCOLPAR { 0, 6, 10, 16 } 

#define NCOLTEMP { 28, 28, 27, 29, 17, 12, 17, 17, 9, 18, 26, 26, 18, 27, 19, 16, 25, 25 } 

#define NCOLTRAN { 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 820, 820 } 

#define CUM_NCOLTEMP { 0, 28, 56, 83, 112, 129, 0, 17, 34, 43, 0, 26, 52, 70, 97, 116, 0, 25 } 

#define CUM_NCOLTRAN { 0, 10, 20, 30, 40, 50, 0, 10, 20, 30, 0, 10, 20, 30, 40, 50, 0, 820 } 

#define CUM_NCOLTRANTEMP { 0, 280, 560, 830, 1120, 1290, 0, 170, 340, 430, 0, 260, 520, 700, 970, 1160, 0, 20500 } 

#define TOT_NCOLTEMP { 141, 61, 132, 50 } 

#define TOT_NCOLTRAN { 60, 40, 60, 1640 } 

#define TOT_NCOLTRANTEMP { 1410, 610, 1320, 41000 } 

#define CUM_TOT_NCOLTEMP { 0, 141, 202, 334 } 

#define CUM_TOT_NCOLTRAN { 0, 60, 100, 160 } 

#define CUM_TOT_NCOLTRANTEMP { 0, 1410, 2020, 3340 } 

#define HEALPIXVECTOR { 1.000000E+00, 0.000000E+00, 0.000000E+00, 7.071068E-01, 7.071068E-01, 0.000000E+00, 6.123234E-17, 1.000000E+00, 0.000000E+00, -7.071068E-01, 7.071068E-01, 0.000000E+00, -1.000000E+00, 1.224647E-16, 0.000000E+00, -7.071068E-01, -7.071068E-01, 0.000000E+00, -1.836970E-16, -1.000000E+00, 0.000000E+00, 7.071068E-01, -7.071068E-01, 0.000000E+00 } 

#define ANTIPOD { 4, 5, 6, 7, 0, 1, 2, 3 } 

#define N_ALIGNED { 3, 3, 3, 3, 3, 3, 3, 3 } 

#define ALIGNED {   \
{ 0, 1, 7, 0 },   \
{ 0, 1, 2, 0 },   \
{ 1, 2, 3, 0 },   \
{ 2, 3, 4, 0 },   \
{ 3, 4, 5, 0 },   \
{ 4, 5, 6, 0 },   \
{ 5, 6, 7, 0 },   \
{ 0, 6, 7, 0 },   \
 } 
