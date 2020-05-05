// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __PROTO_RAYBLOCK_HPP_INCLUDED__
#define __PROTO_RAYBLOCK_HPP_INCLUDED__


///  Structure storing a ray block before it is complete
////////////////////////////////////////////////////////

struct ProtoRayBlock
{
    const Size depth;          ///< depth of the ray  pairs in the block

    vector<RayData> rays_ar;   ///< data for the antipodal ray
    vector<RayData> rays_rr;   ///< data for the regular ray
    vector<Size>    origins;   ///< origin of the ray


    ///  Constructor for a ProtoRayBlock
    ///  @paran[in] ray_ar: ray data of the antipodal ray
    ///  @paran[in] ray_rr: ray data of the regular   ray
    ///  @paran[in] origin: index of the originating cell
    /////////////////////////////////////////////////////

    ProtoRayBlock (const RayData &ray_ar, const RayData &ray_rr, const Size origin)
            : depth (ray_ar.size() + ray_rr.size() + 1)
    {
        rays_ar.push_back (ray_ar);
        rays_rr.push_back (ray_rr);
        origins.push_back (origin);
    }


    ///  Add a ray pair to this ray block
    ///  @param[in] ray_ar: antipodal ray data of the ray pair to add
    ///  @param[in] ray_ar: regular   ray data of the ray pair to add
    ///  @param[in] origin: index of origin    of the ray pair to add
    /////////////////////////////////////////////////////////////////

    inline void add (const RayData &ray_ar, const RayData &ray_rr, const Size origin)
    {
        assert (depth == ray_ar.size() + ray_rr.size() + 1);

        rays_ar.push_back (ray_ar);
        rays_rr.push_back (ray_rr);
        origins.push_back (origin);
    }


    ///  Getter for the number of ray pairs on the protorayblock
    ///  @returns number of ray pairs in the protorayblock
    ////////////////////////////////////////////////////////////

    inline Size nraypairs () const
    {
        assert (origins.size() == rays_ar.size());
        assert (origins.size() == rays_rr.size());

        return origins.size();
    }

};


#endif // __PROTO_RAYBLOCK_HPP_INCLUDED__
