#include "Simulation/Raypair/raypair.hpp"


///  Constructor for RayPair
///    @param[in] length     : number of cells on the ray pair
///    @param[in] n_off_diag : bandwidth of the Approximated Lambda operator (ALO)
//////////////////////////////////////////////////////////////////////////////////

RayPair ::
RayPair (
    const long length,
    const long n_off_diag_)
 : n_off_diag (n_off_diag_)
{

     term1.resize (length);
     term2.resize (length);
         A.resize (length);
         C.resize (length);
         F.resize (length);
         G.resize (length);
        Su.resize (length);
        Sv.resize (length);
      dtau.resize (length);
    L_diag.resize (length);
       chi.resize (length);
       nrs.resize (length);
       frs.resize (length);

       inverse_one_plus_F.resize (length);
       inverse_one_plus_G.resize (length);
        G_over_one_plus_G.resize (length);
                inverse_A.resize (length);
                inverse_C.resize (length);


    if (n_off_diag > 0)
    {
      L_upper.resize (n_off_diag);
      L_lower.resize (n_off_diag);

      for (long m = 0; m < n_off_diag; m++)
      {
        L_upper[m].resize (length);
        L_lower[m].resize (length);
      }
    }


}
