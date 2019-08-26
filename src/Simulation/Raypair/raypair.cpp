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

    //cout << "In Raypair constructor..." << endl;

     term1.resize (length);
     //cout << "1" << endl;
     term2.resize (length);
     //cout << "2" << endl;
         A.resize (length);
     //cout << "3" << endl;
         C.resize (length);
     //cout << "4" << endl;
         F.resize (length);
     //cout << "5" << endl;
         G.resize (length);
     //cout << "6" << endl;
        Su.resize (length);
     //cout << "7" << endl;
        Sv.resize (length);
     //cout << "8" << endl;
      dtau.resize (length);
     //cout << "9" << endl;
    L_diag.resize (length);
     //cout << "10" << endl;
       chi.resize (length);
     //cout << "11" << endl;
       nrs.resize (length);
     //cout << "12" << endl;
       frs.resize (length);
     //cout << "11" << endl;

       inverse_one_plus_F.resize (length);
     //cout << "12" << endl;
       inverse_one_plus_G.resize (length);
     //cout << "13" << endl;
        G_over_one_plus_G.resize (length);
     //cout << "14" << endl;
                inverse_A.resize (length);
     //cout << "15" << endl;
                inverse_C.resize (length);

     //cout << "Z" << endl;

    if (n_off_diag > 0)
    {
     //cout << "Z0" << endl;
      L_upper.resize (n_off_diag);
     //cout << "Z1" << endl;
      L_lower.resize (n_off_diag);

      for (long m = 0; m < n_off_diag; m++)
      {
     //cout << "Z2" << endl;
        L_upper[m].resize (length);
     //cout << "Z3" << endl;
        L_lower[m].resize (length);
      }
    }


}
