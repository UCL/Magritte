// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________



inline int Lambda ::
    initialize (
      const Parameters &parameters,
      const long        nrad_new   )
{

  ncells = parameters.ncells();
  nrad   = nrad_new;

  Lss.reserve (ncells * nrad);
  nrs.reserve (ncells * nrad);

  size.resize (ncells * nrad);


  Ls.resize (ncells);
  nr.resize (ncells);

  for (long p = 0; p < ncells; p++)
  {
    Ls[p].resize (nrad);
    nr[p].resize (nrad);

    for (long k = 0; k < nrad; k++)
    {
      size [index_first(p,k)] = 0;
    }
  }




  return (0);

}


//inline void Lambda ::
//    add_entry (
//        const double Ls_new,
//        const long   nr_new )
//{
//
//  for (long index = 0; index < nr.size(); index++)
//  {
//    if (nr[index] == nr_new)
//    {
//      Ls[index] += Ls_new;
//
//      return;
//    }
//  }
//
//  Ls.push_back (Ls_new);
//  nr.push_back (nr_new);
//
//  return;
//
//}




/// Index of the first element belonging to p and k
///    @param[in] p : index of the receiving cell
///    @param[in] k : index of the line transition
///////////////////////////////////////////////////

inline long Lambda ::
    index_first (
        const long p,
        const long k ) const
{
  return k + nrad * p;
}




/// Index of the last element belonging to p and k
///    @param[in] p : index of the receiving cell
///    @param[in] k : index of the line transition
//////////////////////////////////////////////////

inline long Lambda ::
    index_last (
        const long p,
        const long k ) const
{
  return index_first(p,k) + get_size(p,k) - 1;
}




///  Getter for ALO element
///    @param[in] p      : index of the receiving cell
///    @param[in] k      : index of the line transition
///    @param[in] index  : index of the ALO element
///////////////////////////////////////////////////////

inline double Lambda ::
    get_Ls (
        const long p,
        const long k,
        const long index ) const
{
  //return Ls[index_first(p,k) + index];
  return Ls[p][k][index];
}




///  Getter for cell index corresponding of ALO element
///    @param[in] p      : index of the receiving cell
///    @param[in] k      : index of the line transition
///    @param[in] index  : index of the ALO element
///////////////////////////////////////////////////////

inline long Lambda ::
    get_nr (
        const long p,
        const long k,
        const long index ) const
{
  //return nr[index_first(p,k) + index];
  return nr[p][k][index];
}




///  Getter for cell index corresponding of ALO element
///    @param[in] p      : index of the receiving cell
///    @param[in] k      : index of the line transition
///    @param[in] index  : index of the ALO element
///////////////////////////////////////////////////////

inline long Lambda ::
    get_size (
        const long p,
        const long k ) const
{
  return nr[p][k].size();
  //return size[index_first(p,k)];
}




///  Setter for an ALO element
///    @param[in] p      : index of the receiving cell
///    @param[in] k      : index of the line transition
///    @param[in] nr_new : index of the emitting cell
///    @param[in] Ls_new : new element of the ALO
///////////////////////////////////////////////////////

inline void Lambda ::
    add_element (
        const long   p,
        const long   k,
        const long   nr_new,
        const double Ls_new )
{

  for (long index = 0; index < nr[p][k].size(); index++)
  {
    if (nr[p][k][index] == nr_new)
    {
      Ls[p][k][index] += Ls_new;

      return;
    }
  }

  Ls[p][k].push_back (Ls_new);
  nr[p][k].push_back (nr_new);


  return;

}
