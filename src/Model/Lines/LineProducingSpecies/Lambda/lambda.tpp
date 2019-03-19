// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


inline void Lambda ::
    add_entry (
        const double Ls_new,
        const long   nr_new )
{

  for (long index = 0; index < nr.size(); index++)
  {
    if (nr[index] == nr_new)
    {
      Ls[index] += Ls_new;

      return;
    }
  }

  Ls.push_back (Ls_new);
  nr.push_back (nr_new);

  return;

}
