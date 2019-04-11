// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __TYPES_HPP_INCLUDED__
#define __TYPES_HPP_INCLUDED__

#include <vector>
#include <string>
using std::string;
#include <Eigen/Core>
using Eigen::VectorXd;
using Eigen::MatrixXd;


// Definitions for short notation of vector of vectors of...
// The number in the type name indicates the number of indices


// Vectors of bool
typedef std::vector<bool>  Bool1;
typedef std::vector<Bool1> Bool2;
typedef std::vector<Bool2> Bool3;


// Vectors of char
typedef std::vector<char>  Char1;
typedef std::vector<Char1> Char2;
typedef std::vector<Char2> Char3;


// Vectors of int
typedef std::vector<int>  Int1;
typedef std::vector<Int1> Int2;
typedef std::vector<Int2> Int3;


// Vectors of long
typedef std::vector<long>  Long1;
typedef std::vector<Long1> Long2;
typedef std::vector<Long2> Long3;
typedef std::vector<Long3> Long4;


// Vectors of double
typedef std::vector<double>  Double1;
typedef std::vector<Double1> Double2;
typedef std::vector<Double2> Double3;
typedef std::vector<Double3> Double4;
typedef std::vector<Double4> Double5;


// Vectors of string
typedef std::vector<string>   String1;
typedef std::vector<String1>  String2;
typedef std::vector<String2>  String3;


// Vectors of Eigen::VectorXd
typedef std::vector<VectorXd>   VectorXd1;
typedef std::vector<VectorXd1>  VectorXd2;
typedef std::vector<VectorXd2>  VectorXd3;


// Vectors of Eigen::MatrixXd
typedef std::vector<MatrixXd>   MatrixXd1;
typedef std::vector<MatrixXd1>  MatrixXd2;
typedef std::vector<MatrixXd2>  MatrixXd3;


#endif // __TYPES_HPP_INCLUDED__
