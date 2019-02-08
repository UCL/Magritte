// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <iostream>
#include <fstream>
using namespace std;

#include <pybind11/embed.h> // everything needed for embedding
//#include <pybind11/stl.h>
namespace py = pybind11;

#include "io_Python.hpp"



///  Constructor for IoPython
///    @param[in] implementation: python module containing the implementaion
///    @param[in] io_file: file to read from and write to
////////////////////////////////////////////////////////////////////////////

IoPython ::
    IoPython (
        const string implementation,
        const string io_file        )
  : Io             (io_file),
    implementation (implementation)
{


}   // END OF CONSTRUCTOR




///  read_length:
///  @param[in] file_name: path to file containing the data
///  @param[out] length: length to be read
///////////////////////////////////////////////////////////

int IoPython ::
    read_length (
        const string file_name,
              long  &length    ) const
{

  read_in_python <long> ("read_length", file_name, length);


  return (0);

}




///  read_number:
///  @param[in] file_name: file containing the number
///  @param[out] number: number to be read
/////////////////////////////////////////////////////

int IoPython ::
    read_number (
        const string file_name,
              long  &number    ) const
{

  read_in_python <long> ("read_number", file_name, number);


  return (0);

}




///  write_number:
///  @param[in] file_name: file containing the number
///  @param[in] number: number to be written
/////////////////////////////////////////////////////

int IoPython ::
    write_number (
        const string file_name,
        const long  &number    ) const
{

  write_in_python <long> ("write_number", file_name, number);


  return (0);

}




///  read_word:
///  @param[in] file_name: file containing the number
///  @param[out] word: word to be written
/////////////////////////////////////////////////////

int IoPython ::
    read_word  (
        const string  file_name,
              string &word      ) const
{

  read_in_python <string> ("read_word", file_name, word);


  return (0);

}




///  write_word:
///  @param[in] file_name: file containing the number
///  @param[in] word: word to be written
/////////////////////////////////////////////////////

int IoPython ::
    write_word  (
        const string  file_name,
        const string &word      ) const
{

  write_in_python <string> ("write_word", file_name, word);


  return (0);

}




///  read_list:
///  @param[in] file_name: path to file containing the data
///  @param[in] list: list to be filled
///////////////////////////////////////////////////////////

int IoPython ::
    read_list (
        const string file_name,
              Long1 &list      ) const
{

  read_in_python <Long1> ("read_array", file_name, list);


  return (0);

}




///  write_list:
///  @param[in] file_name: path to file containing the data
///  @param[in] list: list to be written
///////////////////////////////////////////////////////////

int IoPython ::
    write_list (
        const string file_name,
        const Long1 &list      ) const
{

  write_in_python <Long1> ("write_array", file_name, list);


  return (0);

}




///  read_array:
///  @param[in] file_name: path to file containing the data
///  @param[in] list: array to be filled
///////////////////////////////////////////////////////////

int IoPython ::
    read_array (
        const string   file_name,
              Long2   &array     ) const
{

  read_in_python <Long2> ("read_array", file_name, array);


  return (0);

}




///  write_array:
///  @param[in] file_name: path to file containing the data
///  @param[in] list: array to be written
///////////////////////////////////////////////////////////

int IoPython ::
    write_array (
        const string  file_name,
        const Long2  &array     ) const
{

  write_in_python <Long2> ("write_array", file_name, array);


  return (0);

}




///  read_list:
///  @param[in] file_name: path to file containing the data
///  @param[in] list: list to be filled
///////////////////////////////////////////////////////////

int IoPython ::
    read_list (
        const string   file_name,
              Double1 &list      ) const
{

  read_in_python <Double1> ("read_array", file_name, list);


  return (0);

}




///  write_list:
///  @param[in] file_name: path to file containing the data
///  @param[in] list: list to be written
///////////////////////////////////////////////////////////

int IoPython ::
    write_list (
        const string   file_name,
        const Double1 &list      ) const
{

  write_in_python <Double1> ("write_array", file_name, list);


  return (0);

}




///  read_array:
///  @param[in] file_name: path to file containing the data
///  @param[in] list: array to be filled
///////////////////////////////////////////////////////////

int IoPython ::
    read_array (
        const string   file_name,
              Double2 &array     ) const
{

  read_in_python <Double2> ("read_array", file_name, array);


  return (0);

}




///  write_array:
///  @param[in] file_name: path to file containing the data
///  @param[in] list: array to be written
///////////////////////////////////////////////////////////

int IoPython ::
    write_array (
        const string   file_name,
        const Double2 &array      ) const
{

  write_in_python <Double2> ("write_array", file_name, array);


  return (0);

}




///  read_3_vector:
///  @param[in] file_name: path to file containing the data
///  @param[in] x: x component of the vector to be read
///  @param[in] y: y component of the vector to be read
///  @param[in] z: z component of the vector to be read
///////////////////////////////////////////////////////////

int IoPython ::
    read_3_vector (
        const string   file_name,
              Double1 &x,
              Double1 &y,
              Double1 &z         ) const
{

  const long length = x.size();

  // Check if all 3 vectors are the same size

  if (   (length != y.size())
      || (length != z.size()) )
  {
    return (-1);
  }


  Double2 array (3, Double1 (length));

  read_in_python <Double2> ("read_array", file_name, array);

  x = array[0];
  y = array[1];
  z = array[2];


  return (0);

}




///  write_3_vector:
///  @param[in] file_name: path to file containing the data
///  @param[in] x: x component of the vector to be written
///  @param[in] y: y component of the vector to be written
///  @param[in] z: z component of the vector to be written
///////////////////////////////////////////////////////////

int IoPython ::
    write_3_vector (
        const string   file_name,
        const Double1 &x,
        const Double1 &y,
        const Double1 &z         ) const
{

  const long length = x.size();


  // Check if all 3 vectors are the same size

  if (   (length != y.size())
      || (length != z.size()) )
  {
    return (-1);
  }


  Double2 array (3, Double1 (length));

  array[0] = x;
  array[1] = y;
  array[2] = z;


  write_in_python <Double2> ("write_array", file_name, array);


  return (0);

}




///  execute_io_function_in_python: executes io function in python
///    @param[in] function: name of io function to execute
///    @param[in] file_name: name of the file from which to read
///    @param[out] data: the data read from the file
//////////////////////////////////////////////////////////////////

template <typename type>
void IoPython ::
    read_in_python (
        const string  function,
        const string  file_name,
              type   &data      ) const
{

  try
  {
    py::initialize_interpreter ();
  }
  catch (...)
  {
    cout << "No need to initialise..." << endl;
  }

  // Add /pyBindings folder to Python path
  py::module::import("sys").attr("path").attr("insert")(0, "../pyBindings");

  // Import function defined in
  py::object ioFunction = py::module::import(implementation.c_str()).attr(function.c_str());

  // Execute io function
  py::object result = ioFunction (io_file, file_name);

  // Cast result to appropriate type
  data = result.cast<type>();

}




///  execute_io_function_in_python: executes io function in python
///    @param[in] function: name of io function to execute
///    @param[in] file_name: name of the file from which to read
///    @param[out] data: the data read from the file
//////////////////////////////////////////////////////////////////

template <typename type>
void IoPython ::
    write_in_python (
        const string  function,
        const string  file_name,
        const type   &data      ) const
{

  try
  {
    py::initialize_interpreter ();
  }
  catch (...)
  {
    cout << "No need to initialise..." << endl;
  }

  // Add /pyBindings folder to Python path
  py::module::import("sys").attr("path").attr("insert")(0, "../pyBindings");

  // Import function defined in implementation file
  py::object ioFunction = py::module::import(implementation.c_str()).attr(function.c_str());

  // Execute io function
  ioFunction (io_file, file_name, data);


}
