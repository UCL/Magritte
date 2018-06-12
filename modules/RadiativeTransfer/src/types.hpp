// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <vector>
using namespace std;
#include <Eigen/Core>
using namespace Eigen;


// Vectors of bool

typedef               vector<bool>   Bool1;
typedef        vector<vector<bool>>  Bool2;
typedef vector<vector<vector<bool>>> Bool3;


// Vectors of char

typedef               vector<char>   Char1;
typedef        vector<vector<char>>  Char2;
typedef vector<vector<vector<char>>> Char3;


// Vectors of int

typedef               vector<int>   Int1;
typedef        vector<vector<int>>  Int2;
typedef vector<vector<vector<int>>> Int3;


// Vectors of long

typedef                      vector<long>    Long1;
typedef               vector<vector<long>>   Long2;
typedef        vector<vector<vector<long>>>  Long3;
typedef vector<vector<vector<vector<long>>>> Long4;


// Vectors of double

typedef                             vector<double>     Double1;
typedef                      vector<vector<double>>    Double2;
typedef               vector<vector<vector<double>>>   Double3;
typedef        vector<vector<vector<vector<double>>>>  Double4;
typedef vector<vector<vector<vector<vector<double>>>>> Double5;


// Vectors of string

typedef               vector<string>   String1;
typedef        vector<vector<string>>  String2;
typedef vector<vector<vector<string>>> String3;


// Vectors of Eigen::VectorXd

typedef               vector<VectorXd>   VectorXd1;
typedef        vector<vector<VectorXd>>  VectorXd2;
typedef vector<vector<vector<VectorXd>>> VectorXd3;


// Vectors of Eigen::MatrixXd

typedef               vector<MatrixXd>   MatrixXd1;
typedef        vector<vector<MatrixXd>>  MatrixXd2;
typedef vector<vector<vector<MatrixXd>>> MatrixXd3;
