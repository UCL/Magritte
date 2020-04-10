// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________

#ifndef __RAYQUEUE_HPP_INCLUDED__
#define __RAYQUEUE_HPP_INCLUDED__


#include "protoRayBlock.hpp"


///  Structure to store protorayblocks while they are being completed
/////////////////////////////////////////////////////////////////////

struct RayQueue
{
    const Size                    desired_nraypairs;   ///< desired number of ray pairs in a block
    list<ProtoRayBlock>           queue;               ///< list containing the queued proto ray blocks
    list<ProtoRayBlock>::iterator complete_it;         ///< iterator to the complete proto ray block
    bool                          complete = false;    ///< indicates whether there is a complete prb


    ///  Constructor for a RayQueue
    ///  @param[in] nraypairs: desired number of ray pairs for the blocks in the queue
    //////////////////////////////////////////////////////////////////////////////////

    RayQueue (const Size nraypairs): desired_nraypairs (nraypairs) {};


    ///  Add a ray pair to the appropriate block in the ray queue
    ///  @param[in] ray_ar: antipodal ray data of the ray pair to add
    ///  @param[in] ray_ar: regular   ray data of the ray pair to add
    ///  @param[in] origin: index of origin    of the ray pair to add
    ///  @param[in] depth : depth              of the ray pair to add
    /////////////////////////////////////////////////////////////////

    inline void add (const RayData &ray_ar, const RayData &ray_rr, const Size origin, const Size depth)
    {
        for (auto it = queue.begin(); it != queue.end(); it++)
        {
            auto &prb = *it;

            if (prb.depth == depth)
            {
                prb.add (ray_ar, ray_rr, origin);

                if (prb.nraypairs() == desired_nraypairs)
                {
                    complete_it = it;
                    complete    = true;
                }

                return;
            }
        }

        queue.push_back (ProtoRayBlock (ray_ar, ray_rr, origin));

        if (queue.back().nraypairs() == desired_nraypairs)
        {
            auto last = queue.end();
            last--;

            complete_it = last;
            complete    = true;
        }

        return;
    }


    ///  Getter for a complete block in the ray queue
    ///  @returns the complete block in the ray queue
    /////////////////////////////////////////////////

    inline ProtoRayBlock get_complete_block ()
    {
        const ProtoRayBlock complete_block = *complete_it;

        queue.erase (complete_it);

        complete = false;

        return complete_block;
    }
};


#endif // __RAYQUEUE_HPP_INCLUDED__
