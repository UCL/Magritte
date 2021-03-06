// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __SETONCE_HPP_INCLUDED__
#define __SETONCE_HPP_INCLUDED__


#include <exception>
#include <iostream>


struct DoubleSetException : public std::exception
{
	const char* what () const throw ()
    {
	    return "Tried to overwrite SetOnce object with another value.";
    }
};




struct GetBeforeSetException : public std::exception
{
	const char* what () const throw ()
    {
    	return "Tried to get SetOnce object before setting it.";
    }
};




template <typename type>
class SetOnce
{
  private:

      bool already_set = false;
      type value;


  public:

      inline void set (
          const type new_value)
      {
        if (already_set)
        {
          if (value != new_value)
          {
            std::cout << "!!! ERROR !!!"             << std::endl;
            std::cout << "    value = " <<     value << std::endl;
            std::cout << "new value = " << new_value << std::endl;
            std::cout << "!!!       !!!"             << std::endl;

            throw DoubleSetException ();
          }
        }

        else
        {
          already_set = true;
          value       = new_value;
        }
      }


      inline type get () const
      {
        if (already_set)
        {
          return value;
        }

        else
        {
          throw GetBeforeSetException ();
        }
      }

};


#endif // __SETONCE_HPP_INCLUDED__
