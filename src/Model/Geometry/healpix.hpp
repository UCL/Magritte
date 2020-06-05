/* -----------------------------------------------------------------------------
 *
 *  Copyright (C) 1997-2016 Krzysztof M. Gorski, Eric Hivon, Martin Reinecke,
 *                          Benjamin D. Wandelt, Anthony J. Banday,
 *                          Matthias Bartelmann,
 *                          Reza Ansari & Kenneth M. Ganga
 *
 *
 *  This file is part of HEALPix.
 *
 *  HEALPix is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  HEALPix is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with HEALPix; if not, write to the Free Software
 *  Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
 *
 *  For more information about HEALPix see http://healpix.sourceforge.net
 *
 *---------------------------------------------------------------------------*/


#ifndef MAGRITTE_HEALPIX_HPP
#define MAGRITTE_HEALPIX_HPP


#include <math.h>

#include <Eigen/Core>
using Eigen::Vector3d;


static const double twothird   = 2.0/3.0;
static const double pi         = 3.141592653589793238462643383279502884197;
static const double twopi      = 6.283185307179586476925286766559005768394;
static const double halfpi     = 1.570796326794896619231321691639751442099;
static const double inv_halfpi = 0.6366197723675813430755350534900574;


static const int jrll[] = {2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4};
static const int jpll[] = {1, 3, 5, 7, 0, 2, 4, 6, 1, 3, 5, 7};


static const short ctab[]={
#define Z(a) a,a+1,a+256,a+257
#define Y(a) Z(a),Z(a+2),Z(a+512),Z(a+514)
#define X(a) Y(a),Y(a+4),Y(a+1024),Y(a+1028)
        X(0),X(8),X(2048),X(2056)
#undef X
#undef Y
#undef Z
};

static const short utab[]={
#define Z(a) 0x##a##0, 0x##a##1, 0x##a##4, 0x##a##5
#define Y(a) Z(a##0), Z(a##1), Z(a##4), Z(a##5)
#define X(a) Y(a##0), Y(a##1), Y(a##4), Y(a##5)
        X(0),X(1),X(4),X(5)
#undef X
#undef Y
#undef Z
};


#ifndef __BMI2__

static inline int xyf2nest (const int nside, const int ix, const int iy, const int face_num)
{
    return (face_num*nside*nside)
           + (   utab[ix&0xff]
              | (utab[ix>>8  ]<<16)
              | (utab[iy&0xff]<<1 )
              | (utab[iy>>8  ]<<17) );
}


static inline void nest2xyf (const int nside, int pix, int *ix, int *iy, int *face_num)
{
    int npface_=nside*nside, raw;
    *face_num = pix/npface_;
    pix &= (npface_-1);
    raw = (pix&0x5555) | ((pix&0x55550000)>>15);
    *ix = ctab[raw&0xff] | (ctab[raw>>8]<<4);
    pix >>= 1;
    raw = (pix&0x5555) | ((pix&0x55550000)>>15);
    *iy = ctab[raw&0xff] | (ctab[raw>>8]<<4);
}

#else

#include <x86intrin.h>

static inline int xyf2nest (const int nside, const int ix, const int iy, const int face_num)
{
    return (face_num*nside*nside) + (_pdep_u32(ix,0x55555555u) | _pdep_u32(iy,0xaaaaaaaau));
}


static inline void nest2xyf (const int nside, int pix, int *ix, int *iy, int *face_num)
{
  int npface_=nside*nside, raw;
  *face_num = pix/npface_;
  pix &= (npface_-1);
  *ix=_pext_u32(pix,0x55555555u);
  *iy=_pext_u32(pix,0xaaaaaaaau);
}

#endif


static inline double fmodulo (const double v1, const double v2)
{
    if (v1>=0)
    {
        return (v1<v2) ? v1 : fmod(v1,v2);
    }

    double tmp = fmod(v1,v2)+v2;

    return (tmp==v2) ? 0. : tmp;
}


static inline int ang2pix_nest_z_phi (const long nside_, const double z, const double phi)
{
    double za = fabs(z);
    double tt = fmodulo(phi,twopi) * inv_halfpi; /* in [0,4) */
    int face_num, ix, iy;

    if (za<=twothird) /* Equatorial region */
    {
        double temp1 = nside_*(0.5+tt);
        double temp2 = nside_*(z*0.75);
        int jp = (int)(temp1-temp2); /* index of  ascending edge line */
        int jm = (int)(temp1+temp2); /* index of descending edge line */
        int ifp = jp/nside_;  /* in {0,4} */
        int ifm = jm/nside_;
        face_num = (ifp==ifm) ? (ifp|4) : ((ifp<ifm) ? ifp : (ifm+8));

        ix = jm & (nside_-1);
        iy = nside_ - (jp & (nside_-1)) - 1;
    }
    else /* polar region, za > 2/3 */
    {
        int ntt = (int)tt, jp, jm;
        double tp, tmp;
        if (ntt>=4) ntt=3;
        tp = tt-ntt;
        tmp = nside_*sqrt(3*(1-za));

        jp = (int)(tp*tmp); /* increasing edge line index */
        jm = (int)((1.0-tp)*tmp); /* decreasing edge line index */
        if (jp>=nside_) jp = nside_-1; /* for points too close to the boundary */
        if (jm>=nside_) jm = nside_-1;
        if (z >= 0)
        {
            face_num = ntt;  /* in {0,3} */
            ix = nside_ - jm - 1;
            iy = nside_ - jp - 1;
        }
        else
        {
            face_num = ntt + 8; /* in {8,11} */
            ix =  jp;
            iy =  jm;
        }
    }

    return xyf2nest(nside_,ix,iy,face_num);
}


static inline void pix2ang_nest_z_phi (const int nside_, const int pix, double *z, double *phi)
{
    int nl4 = nside_*4;
    int npix_ = 12*nside_*nside_;
    double fact2_ = 4./npix_;
    int face_num, ix, iy, nr, kshift;

    nest2xyf(nside_,pix,&ix,&iy,&face_num);
    int jr = (jrll[face_num]*nside_) - ix - iy - 1;

    if (jr < nside_)
    {
        nr = jr;
        *z = 1 - nr*nr*fact2_;
        kshift = 0;
    }
    else if (jr > 3*nside_)
    {
        nr = nl4-jr;
        *z = nr*nr*fact2_ - 1;
        kshift = 0;
    }
    else
    {
        double fact1_ = (nside_<<1)*fact2_;
        nr = nside_;
        *z = (2*nside_-jr)*fact1_;
        kshift = (jr-nside_)&1;
    }

    int jp = (jpll[face_num]*nr + ix -iy + 1 + kshift) / 2;
    if (jp>nl4) jp-=nl4;
    if (jp<1) jp+=nl4;

    *phi = (jp-(kshift+1)*0.5)*(halfpi/nr);
}


inline size_t vec2pix_nest (const size_t nside, const Vector3d vec)
{
    return ang2pix_nest_z_phi (nside, vec[2]/vec.norm(), atan2(vec[1], vec[0]));
}


inline Vector3d pix2vec_nest (const size_t nside, const size_t ipix)
{
    double z, phi; pix2ang_nest_z_phi(nside, ipix, &z, &phi);
    double stheta = sqrt((1.0-z)*(1.0+z));

    return {stheta*cos(phi), stheta*sin(phi), z};
}

#endif //MAGRITTE_HEALPIX_HPP
