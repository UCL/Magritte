#ifndef __SOLVER_HPP_INCLUDED__
#define __SOLVER_HPP_INCLUDED__


enum Device   { CPU,   GPU};
enum Order    {   2,     4};
enum Adaptive {True, False};



struct Solver : public RayBlock
{

    Solver();

};


#endif // __SOLVER_HPP_INCLUDED__
