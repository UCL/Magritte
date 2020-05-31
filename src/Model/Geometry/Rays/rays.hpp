// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __RAYS_HPP_INCLUDED__
#define __RAYS_HPP_INCLUDED__


#include "Io/io.hpp"
#include "Tools/types.hpp"
#include "Model/parameters.hpp"

#include "Tools/logger.hpp"


///  RAYS: data struct containing directional discretization info
/////////////////////////////////////////////////////////////////

struct Rays
{

    public:

        bool adaptive_ray_tracing = false;   ///< true when using adaptive ray-tracing
        size_t half_nrays;                   ///< half the number of rays
        size_t nrays;                        ///< number of rays


        /// Pre-defined rays
        ////////////////////

        Vector3d1 rays;      ///< direction vector
        Double1   weights;   ///< weights for angular integration


        /// Adaptive rays
        /////////////////

        Vector3d2 dir;
        Double1   wgt;

        vector<vector<unsigned short int>> order;
        vector<vector<unsigned short int>> pixel;

        inline int get_order (const size_t o, const size_t r) const {return order[o][r];}
        inline int get_pixel (const size_t o, const size_t r) const {return pixel[o][r];}


        inline Vector3d ray (const size_t o, const size_t r) const
        {
            if (adaptive_ray_tracing)
            {
                if (r < half_nrays) {return  dir[order[o][r           ]][pixel[o][r           ]];}
                else                {return -dir[order[o][r-half_nrays]][pixel[o][r-half_nrays]];}
            }
            else return rays[r];
        }

        inline double weight (const size_t o, const size_t r) const
        {
            if (adaptive_ray_tracing)
            {
                if (r < half_nrays) {return wgt[order[o][r           ]];}
                else                {return wgt[order[o][r-half_nrays]];}
            }
            else return weights[r];
        }


        Size1 antipod;   ///< ray number of antipodal ray


        // Io
        void read  (const Io &io, Parameters &parameters);
        void write (const Io &io                        ) const;

        void setup ();


    private:

        // Helper functions
        void setup_antipodal_rays ();

        static const string prefix;

};




#endif // __RAYS_HPP_INCLUDED__
