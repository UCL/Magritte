// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __LINEDATA_HPP_INCLUDED__
#define __LINEDATA_HPP_INCLUDED__


#include <Eigen/Core>
using namespace Eigen;

#include "types.hpp"
#include "collisionPartner.hpp"
#include "io.hpp"
#include "species.hpp"


struct Linedata
{

  public:

      long   num;                        ///< number of line producing species
      string sym;                        ///< symbol of line producing species

      long nlev;                         ///< number of levels
      long nrad;                         ///< number of radiative transitions

      Long1 irad;                        ///< level index of radiative transition
      Long1 jrad;                        ///< level index of radiative transition

      Double1 energy;                    ///< energy of level
      Double1 weight;                    ///< weight of level (statistical)

      Double1 frequency;                 ///< frequency corresponding to each transition

      Double1 A;                         ///< Einstein A  (spontaneous emission)
      Double1 Ba;                        ///< Einstsin Ba (absorption)
      Double1 Bs;                        ///< Einstein Bs (stimulated emission)


      long ncolpar;                      ///< number of collision partners

      // Collision partners
      vector <CollisionPartner> colpar;   ///< Vector containing collision partner data


      // Io
      int read (
          const Io &io,
          const int l  );

      int write (
          const Io &io,
          const int l  ) const;


//  MatrixXd calc_Einstein_C (
//      const Species &species,
//      const double   temperature_gas,
//			const long     p               ) const;
//
//
//  MatrixXd calc_transition_matrix (
//      const Species &species,
////    const LINES   &lines,
//      const double   temperature_gas,
//      const Double3 &J_eff,
//      const long     p               ) const;


};


#endif // __LINEDATA_HPP_INCLUDED__
