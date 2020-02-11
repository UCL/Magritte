// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __LINEDATA_HPP_INCLUDED__
#define __LINEDATA_HPP_INCLUDED__


#include <Eigen/Core>

#include "Io/io.hpp"
#include "Tools/types.hpp"
#include "Model/parameters.hpp"
#include "CollisionPartner/collisionPartner.hpp"


struct Linedata
{

  public:

      size_t num;                              ///< number of line producing species
      string sym;                              ///< symbol of line producing species
      double inverse_mass;                     ///< 1/mass of line producing species

      size_t nlev;                               ///< number of levels
      size_t nrad;                               ///< number of radiative transitions

      Long1 irad;                              ///< level index of radiative transition
      Long1 jrad;                              ///< level index of radiative transition

      Double1 energy;                          ///< energy of level
      Double1 weight;                          ///< weight of level (statistical)

      Double1 frequency;                       ///< frequency corresponding to each transition

      Double1 A;                               ///< Einstein A  (spontaneous emission)
      Double1 Ba;                              ///< Einstsin Ba (absorption)
      Double1 Bs;                              ///< Einstein Bs (stimulated emission)


      size_t ncolpar;                            ///< number of collision partners

      // Collision partners
      std::vector <CollisionPartner> colpar;   ///< Vector containing collision partner data

      size_t ncol_tot;


      // Io
      void read  (const Io &io, const int l);
      void write (const Io &io, const int l) const;


  private:

      static const string prefix;



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
