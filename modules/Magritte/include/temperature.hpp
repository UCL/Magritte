// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __TEMPERATURE_HPP_INCLUDED__
#define __TEMPERATURE_HPP_INCLUDED__


#include <string>
using namespace std;

#include "types.hpp"
#include "io.hpp"


struct Temperature
{

  public:

      long ncells;

      Double1 gas;        ///< [K] gas temperature
      Double1 dust;       ///< [K] dust temparature
      Double1 gas_prev;   ///< [K] gas temperature in previous iteration

      Double1 vturb2;     ///< [.] microturbulence over c all squared

      // Construvtor
      Temperature (
          const Io &io);

      // Writer or output
      int write (
          const Io &io) const;


  private:

      int allocate ();

      int read (
          const Io &io);

      int setup ();


};


#endif // __TEMPERATURE_HPP_INCLUDED__
