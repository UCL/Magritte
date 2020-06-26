// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


inline void Lambda :: initialize (const Parameters &parameters, const size_t nrad_new)
{
    ncells = parameters.ncells();
    nrad   = nrad_new;

    Lss.reserve (ncells * nrad);
    nrs.reserve (ncells * nrad);

    size.resize (ncells * nrad);


    Ls.resize (ncells);
    nr.resize (ncells);

    for (size_t p = 0; p < ncells; p++)
    {
        Ls[p].resize (nrad);
        nr[p].resize (nrad);

        for (size_t k = 0; k < nrad; k++)
        {
            Ls[p][k].reserve (5);
            nr[p][k].reserve (5);

            size [index_first(p,k)] = 0;
        }
    }

}



inline void Lambda :: clear ()
{
    OMP_PARALLEL_FOR (p, ncells)
    {
        for (size_t k = 0; k < nrad; k++)
        {
            Ls[p][k].clear();
            nr[p][k].clear();
        }
    }

}




/// Index of the first element belonging to p and k
///    @param[in] p : index of the receiving cell
///    @param[in] k : index of the line transition
///////////////////////////////////////////////////

inline size_t Lambda :: index_first (const size_t p, const size_t k) const
{
    return k + nrad * p;
}




/// Index of the last element belonging to p and k
///    @param[in] p : index of the receiving cell
///    @param[in] k : index of the line transition
//////////////////////////////////////////////////

inline size_t Lambda :: index_last (const size_t p, const size_t k) const
{
    return index_first(p,k) + get_size(p,k) - 1;
}




///  Getter for ALO element
///    @param[in] p      : index of the receiving cell
///    @param[in] k      : index of the line transition
///    @param[in] index  : index of the ALO element
///////////////////////////////////////////////////////

inline double Lambda :: get_Ls (const size_t p, const size_t k, const size_t index) const
{
    //return Ls[index_first(p,k) + index];
    return Ls[p][k][index];
}




///  Getter for cell index corresponding of ALO element
///    @param[in] p      : index of the receiving cell
///    @param[in] k      : index of the line transition
///    @param[in] index  : index of the ALO element
///////////////////////////////////////////////////////

inline size_t Lambda :: get_nr (const size_t p, const size_t k, const size_t index ) const
{
    //return nr[index_first(p,k) + index];
    return nr[p][k][index];
}




///  Getter for cell index corresponding of ALO element
///    @param[in] p      : index of the receiving cell
///    @param[in] k      : index of the line transition
///    @param[in] index  : index of the ALO element
///////////////////////////////////////////////////////

inline size_t Lambda :: get_size (const size_t p, const size_t k ) const
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

inline void Lambda :: add_element (const size_t p, const size_t k, const size_t nr_new, const double Ls_new)
{
    for (size_t index = 0; index < nr[p][k].size(); index++)
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




inline void Lambda :: linearize_data ()
{
    size_t size_total = 0;

#   pragma omp parallel for reduction (+: size_total)
    for (size_t p = 0; p < ncells; p++)
    {
        for (size_t k = 0; k < nrad; k++)
        {
            const size_t index = index_first (p,k);

            size[index] = nr[p][k].size();
            size_total += size[index];
        }
    }

    cout << "Size total = " << size_total << endl;

    Lss.resize (size_total);
    nrs.resize (size_total);


    OMP_PARALLEL_FOR (p, ncells)
    {
        for (size_t k = 0; k < nrad; k++)
        {
            const size_t index = index_first (p,k);

            for (size_t m = 0; m < size[index]; m++)
            {
                Lss[index+m] = Ls[p][k][m];
                nrs[index+m] = nr[p][k][m];
            }
        }
    }

}










inline int Lambda :: MPI_gather ()

#if (MPI_PARALLEL)

{
    this->linearize_data();

    int size_total = Lss.size();


    // Gather the lengths of the linearized vectors of each process
    Int1 buffer_lengths (MPI_comm_size(), 0);
    Int1 displacements  (MPI_comm_size(), 0);


    int ierr_l = MPI_Allgather (
                     &size_total,             // pointer to data to be send
                     1,                       // number of elements in the send buffer
                     MPI_INT,                 // type of the send data
                     buffer_lengths.data(),   // pointer to the data to be received
                     1,                       // number of elements in receive buffer
                     MPI_INT,                 // type of the received data
                     MPI_COMM_WORLD        );
    assert (ierr_l == 0);


    for (int w = 1; w < MPI_comm_size(); w++)
    {
        displacements[w] = buffer_lengths[w-1];

        cout << "buffer_lengths [w] = " << buffer_lengths[w-1] << endl;
    }

    cout << "buffer_lengths [f] = " << buffer_lengths[MPI_comm_size()-1] << endl;


    Double1 Lss_total;
    Long1   nrs_total;
    Long1   szs_total;


    long total_buffer_length = 0;

    for (long length : buffer_lengths) {total_buffer_length += length;}

    Lss_total.resize (total_buffer_length);
    nrs_total.resize (total_buffer_length);
    szs_total.resize (MPI_comm_size()*ncells*nrad);


    int ierr_ls =	MPI_Allgatherv (
                      Lss.data(),              // pointer to data to be send
                      size_total,              // number of elements in the send buffer
                      MPI_DOUBLE,              // type of the send data
                      Lss_total.data(),        // pointer to the data to be received
                      buffer_lengths.data(),   // list of numbers of elements in receive buffer
                      displacements.data(),    // displacements between data blocks
  	                  MPI_DOUBLE,              // type of the received data
                      MPI_COMM_WORLD);

    assert (ierr_ls == 0);


    int ierr_nr =	MPI_Allgatherv (
                      nrs.data(),              // pointer to data to be send
                      size_total,              // number of elements in the send buffer
                      MPI_LONG,                // type of the send data
                      nrs_total.data(),        // pointer to the data to be received
                      buffer_lengths.data(),   // list of numbers of elements in receive buffer
                      displacements.data(),    // displacements between data blocks
  	                  MPI_LONG,                // type of the received data
                      MPI_COMM_WORLD);

    assert (ierr_nr == 0);


    int ierr_sz =	MPI_Allgather (
                      size.data(),             // pointer to data to be send
                      ncells*nrad,             // number of elements in the send buffer
                      MPI_LONG,                // type of the send data
                      szs_total.data(),        // pointer to the data to be received
                      ncells*nrad,             // number of elements in receive buffer
                      MPI_LONG,                // type of the received data
                      MPI_COMM_WORLD);

    assert (ierr_sz == 0);


    this->clear();


    long index = 0;

    for (int w = 0; w < MPI_comm_size(); w++)
    {
        for (long p = 0; p < ncells; p++)
        {
            for (long k = 0; k < nrad; k++)
            {
                for (long m = 0; m < szs_total[index]; m++)
                {
                    add_element (p, k, nrs_total[index+m], Lss_total[index+m]);

//                    if (MPI_comm_rank () == 0)
//                    {
//                      cout << p << " " << k << " " << nrs_total[index+m] << " " << Lss_total[index+m] << endl;
//                    }
                }

                index++;
            }
        }
    }

    return (0);
}

#else

{
    return (0);
}

#endif
