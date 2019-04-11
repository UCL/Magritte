// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __LOGGER_HPP_INCLUDED__
#define __LOGGER_HPP_INCLUDED__


#include <iostream>
#include <string>
using std::string;


inline void write_to_log (
    const string text    )
{
  std::cout << text << std::endl;
}


inline void write_to_log (
    const string text,
    const double number  )
{
  std::cout << text << number << std::endl;
}


inline void write_to_log (
    const string text1,
    const double number1,
    const string text2,
    const double number2 )
{
  std::cout << text1 << number1 << text2 << number2 << std::endl;
}


inline void write_to_log (
    const string text1,
    const double number1,
    const string text2,
    const double number2,
    const string text3   )
{
  std::cout << text1 << number1 << text2 << number2 << text3 << std::endl;
}


inline void write_to_log (
    const string text1,
    const double number1,
    const string text2,
    const double number2,
    const string text3,
    const double number3 )
{
  std::cout << text1 << number1 << text2 << number2 << text3 << number3 << std::endl;
}


#endif // __LOGGER_HPP_INCLUDED__
