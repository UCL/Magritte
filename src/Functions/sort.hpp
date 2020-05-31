#ifndef MAGRITTE_SORT_HPP
#define MAGRITTE_SORT_HPP


#include <iostream>
#include <vector>
#include <numeric>      // std::iota
#include <algorithm>    // std::sort, std::stable_sort


template <typename T>
inline std::vector<size_t> sort_indices (const std::vector<T> &v)
{
    // initialize original index locations
    std::vector<size_t> idx(v.size());
    std::iota(idx.begin(), idx.end(), 0);

    // sort indices based on comparing values in v
    // using std::stable_sort instead of std::sort
    // to avoid unnecessary index re-orderings
    // when v contains elements of equal values
    std::stable_sort(idx.begin(), idx.end(), [&v](size_t i1, size_t i2) {return v[i1] < v[i2];});

    return idx;
}


#endif //MAGRITTE_SORT_HPP
