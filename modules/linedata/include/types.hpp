// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __TYPES_HPP_INCLUDED__
#define __TYPES_HPP_INCLUDED__

#include <vector>
#include <string>
using namespace std;
#include <Eigen/Core>
using namespace Eigen;


// Definitions for short notation of vector of vectors of...
// The number in the type name indicates the number of indices 


// Vectors of bool

typedef vector<bool>  Bool1;
typedef vector<Bool1> Bool2;
typedef vector<Bool2> Bool3;


// Vectors of char

typedef vector<char>  Char1;
typedef vector<Char1> Char2;
typedef vector<Char2> Char3;


// Vectors of int

typedef vector<int>  Int1;
typedef vector<Int1> Int2;
typedef vector<Int2> Int3;


// Vectors of long

typedef vector<long>  Long1;
typedef vector<Long1> Long2;
typedef vector<Long2> Long3;
typedef vector<Long3> Long4;


// Vectors of double

typedef vector<double>  Double1;
typedef vector<Double1> Double2;
typedef vector<Double2> Double3;
typedef vector<Double3> Double4;
typedef vector<Double4> Double5;


// Vectors of string

typedef vector<string>   String1;
typedef vector<String1>  String2;
typedef vector<String2>  String3;


// Vectors of Eigen::VectorXd

typedef vector<VectorXd>   VectorXd1;
typedef vector<VectorXd1>  VectorXd2;
typedef vector<VectorXd2>  VectorXd3;


// Vectors of Eigen::MatrixXd

typedef vector<MatrixXd>   MatrixXd1;
typedef vector<MatrixXd1>  MatrixXd2;
typedef vector<MatrixXd2>  MatrixXd3;


#endif // __TYPES_HPP_INCLUDED__
