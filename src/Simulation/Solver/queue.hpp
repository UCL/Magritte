#ifndef MAGRITTE_QUEUE_HPP
#define MAGRITTE_QUEUE_HPP


#include <algorithm>
#include "Model/model.hpp"


struct ProtoBlock
{
    size_t depth;

    vector<RayData> rays_ar;   ///< data for the antipodal ray
    vector<RayData> rays_rr;   ///< data for the regular ray
    vector<size_t>  origins;   ///< origin of the ray

    ProtoBlock() {};

    ProtoBlock (const RayData &ray_ar, const RayData &ray_rr, const size_t origin)
            : depth (ray_ar.size() + ray_rr.size() + 1)
    {
        rays_ar.push_back (ray_ar);
        rays_rr.push_back (ray_rr);
        origins.push_back (origin);
    }


    inline void add (const RayData &ray_ar, const RayData &ray_rr, const size_t origin)
    {
        assert (depth == ray_ar.size() + ray_rr.size() + 1);

        rays_ar.push_back (ray_ar);
        rays_rr.push_back (ray_rr);
        origins.push_back (origin);
    }

    inline size_t nraypairs () const
    {
        assert (origins.size() == rays_ar.size());
        assert (origins.size() == rays_rr.size());

        return origins.size();
    }

};




struct Queue
{
    const size_t                     desired_nraypairs;   ///< desired number of ray pairs in a block
    list<ProtoBlock>                 queue;               ///< list containing the queued proto ray blocks
    list<list<ProtoBlock>::iterator> completed;           ///< iterator to the complete proto ray block

    Queue (const size_t nraypairs): desired_nraypairs (nraypairs) {};

    inline void add (
            const RayData &ray_ar,
            const RayData &ray_rr,
            const size_t origin,
            const size_t depth    )
    {
        for (auto it = queue.begin(); it != queue.end(); it++)
        {
            auto &prb = *it;

            if (prb.depth == depth)
            {
                prb.add (ray_ar, ray_rr, origin);

                if (prb.nraypairs() == desired_nraypairs)
                {
                    completed.push_back (it);
                }

                return;
            }
        }

        queue.push_back (ProtoBlock (ray_ar, ray_rr, origin));

        if (queue.back().nraypairs() == desired_nraypairs)
        {
            auto last = queue.end();
            last--;

            completed.push_back (last);
        }

        return;
    }


    inline ProtoBlock get_complete_block ()
    {
        const ProtoBlock complete_block = *completed.front();

        queue.erase (completed.front());
        completed.pop_front();

        return complete_block;
    }

    inline bool some_are_completed ()
    {
        return (completed.size() > 0);
    }


    inline bool is_not_full () const
    {
        Size area = 0;

        for (auto &prb : queue)
        {
            area += prb.nraypairs();
        }

        return true;//(area < 50);
    }


    inline ProtoBlock get_block ()
    {
        const ProtoBlock block = queue.front();

        queue.pop_front();

        return block;
    }


    inline void get_block (ProtoBlock &block, bool &avail)
    {
        avail = !queue.empty();

        if (avail)
        {
            block = queue.front(); // Copy!

            queue.pop_front();
        }
    }


    inline void ordered_add (
            const RayData &ray_ar,
            const RayData &ray_rr,
            const size_t origin,
            const size_t depth    )
    {
        // A raipair is more likely to end up in a block with more raypairs
        // (since the queue is a sample of the distribution of raypairs).
        // Store blocks with more raypairs first, so they can be inserted faster.

        for (auto it = queue.begin(); it != queue.end(); it++) if (it->depth == depth)
        {
            it->add (ray_ar, ray_rr, origin);

            while ((it != queue.begin()) && (it->nraypairs() > prev(it)->nraypairs()))
            {
                std::swap (*it, *prev(it));
                it--;
            }

            return;
        }

        queue.push_back (ProtoBlock (ray_ar, ray_rr, origin));

        return;
    }


    inline void get_block (const bool gpu_thread, ProtoBlock &block, bool &avail)
    {
        avail = !queue.empty();

        if (avail)
        {
            if (gpu_thread)
            {
                block = queue.front(); // Copy!
                queue.pop_front();
            }
            else
            {
                block = queue.back(); // Copy!
                queue.pop_back();
            }
        }
    }

};

#endif //MAGRITTE_QUEUE_HPP
