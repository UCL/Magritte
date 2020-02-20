// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __RAYS_HPP_INCLUDED__
#define __RAYS_HPP_INCLUDED__


#include "Io/io.hpp"
#include "Tools/types.hpp"
#include "Model/parameters.hpp"


///  RAYS: data struct containing directional discretization info
/////////////////////////////////////////////////////////////////

struct Rays
{
    public:

        vector<Vector3d> rays;      ///< direction vector
        Double1          weights;   ///< weights for angular integration
        Long1            antipod;   ///< ray number of antipodal ray


      // Io
      void read  (const Io &io, Parameters &parameters);
      void write (const Io &io                        ) const;


  private:

      size_t ncells;                 ///< number of cells
      size_t nrays;                  ///< number of rays


      int setup ();

      // Helper functions
      int setup_antipodal_rays ();


      static const string prefix;

};




#endif // __RAYS_HPP_INCLUDED__
