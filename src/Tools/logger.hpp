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
    const auto text    )
{
  std::cout << text << std::endl;
}


inline void write_to_log (
    const auto text1,
    const auto text2     )
{
  std::cout << text1 << text2 << std::endl;
}


inline void write_to_log (
    const auto text1,
    const auto text2,
    const auto text3     )
{
  std::cout << text1 << text2 << text3 << std::endl;
}


inline void write_to_log (
    const auto text1,
    const auto text2,
    const auto text3,
    const auto text4     )
{
  std::cout << text1 << text2 << text3 << text4 << std::endl;
}


inline void write_to_log (
    const auto text1,
    const auto text2,
    const auto text3,
    const auto text4,
    const auto text5     )
{
  std::cout << text1 << text2 << text3 << text4 << text5 << std::endl;
}


inline void write_to_log (
    const auto text1,
    const auto text2,
    const auto text3,
    const auto text4,
    const auto text5,
    const auto text6     )
{
  std::cout << text1 << text2 << text3 << text4 << text5 << text6 << std::endl;
}


#endif // __LOGGER_HPP_INCLUDED__
