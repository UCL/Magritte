// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <string>
using std::string;

#include <pybind11/pybind11.h>
#include <pybind11/embed.h> // everything needed for embedding
#include <pybind11/stl.h>

//using namespace std;
namespace py = pybind11;

#include "configure.hpp"
#include "Io/io_Python.hpp"
#include "Tools/logger.hpp"


///  Constructor for IoPython
///    @param[in] implementation: python module containing the implementaion
///    @param[in] io_file: file to read from and write to
////////////////////////////////////////////////////////////////////////////

IoPython ::
    IoPython (
        const string imp,
        const string io_file)
  : Io             (io_file),
    implementation ("io_" + imp)
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

  int err = read_in_python <long> ("read_length", file_name, length);


  return err;

}




///  read_width:
///  @param[in] file_name: path to file containing the data
///  @param[out] length: length to be read
///////////////////////////////////////////////////////////

int IoPython ::
    read_width (
        const string file_name,
              long  &length    ) const
{

  int err = read_in_python <long> ("read_width", file_name, length);


  return err;

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

  int err = read_in_python <long> ("read_number", file_name, number);


  return err;

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

  int err = write_in_python <long> ("write_number", file_name, number);


  return err;

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

  int err = read_in_python <string> ("read_attribute", file_name, word);


  return err;

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

  int err = write_in_python <string> ("write_attribute", file_name, word);


  return err;

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

  int err = read_in_python <Long1> ("read_array", file_name, list);


  return err;

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

  int err = 0;


  if (list.size() > 0)
  {
    err = write_in_python <Long1> ("write_array", file_name, list);
  }


  return err;

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

  int err = read_in_python <Double1> ("read_array", file_name, list);


  return err;

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

  int err = 0;


  if (list.size() > 0)
  {
    err = write_in_python <Double1> ("write_array", file_name, list);
  }


  return err;

}




///  read_list:
///  @param[in] file_name: path to file containing the data
///  @param[in] list: list to be filled
///////////////////////////////////////////////////////////

int IoPython ::
    read_list (
        const string   file_name,
              String1 &list      ) const
{

  int err = read_in_python <String1> ("read_array", file_name, list);


  return err;

}




///  write_list:
///  @param[in] file_name: path to file containing the data
///  @param[in] list: list to be written
///////////////////////////////////////////////////////////

int IoPython ::
    write_list (
        const string   file_name,
        const String1 &list      ) const
{

  int err = 0;


  if (list.size() > 0)
  {
    err = write_in_python <String1> ("write_array", file_name, list);
  }


  return err;

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

  int err = read_in_python <Long2> ("read_array", file_name, array);


  return err;

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

  int err = 0;


  if (array.size() > 0)
  {
    err = write_in_python <Long2> ("write_array", file_name, array);
  }


  return err;

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

  int err = read_in_python <Double2> ("read_array", file_name, array);


  return err;

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

  int err = 0;


  if (array.size() > 0)
  {
    err = write_in_python <Double2> ("write_array", file_name, array);
  }


  return err;

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


  int err = read_in_python <Double2> ("read_array", file_name, array);


  for (long p = 0; p < length; p++)
  {
    x[p] = array[p][0];
    y[p] = array[p][1];
    z[p] = array[p][2];
  }


  return err;

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


  Double2 array (length, Double1 (3));


  for (long p = 0; p < length; p++)
  {
    array[p][0] = x[p];
    array[p][1] = y[p];
    array[p][2] = z[p];
  }


  int err = 0;

  if (length > 0)
  {
    err = write_in_python <Double2> ("write_array", file_name, array);
  }


  return err;

}




///  execute_io_function_in_python: executes io function in python
///    @param[in] function: name of io function to execute
///    @param[in] file_name: name of the file from which to read
///    @param[out] data: the data read from the file
//////////////////////////////////////////////////////////////////

template <class type>
int IoPython ::
    read_in_python (
        const string  function,
        const string  file_name,
              type   &data      ) const
{

  try
  {
    py::initialize_interpreter ();
  }
  catch (...) { }


  const string io_folder = string (MAGRITTE_FOLDER) + "/src/Io";

  // Add /Io folder to Python path
  py::module::import("sys").attr("path").attr("insert")(0, io_folder);

  // Import function defined in
  py::object ioFunction = py::module::import(implementation.c_str()).attr(function.c_str());

  // Make a copy of data
  type data_copy = data;

  try
  {
    // Execute io function
    py::object result = ioFunction (io_file, file_name);

    // Cast result to appropriate type
    data = result.cast<type>();
  }
  catch (...)
  {
    // Recover previous state of data
    data = data_copy;

    return (-1);
  }


  return (0);

}




///  execute_io_function_in_python: executes io function in python
///    @param[in] function: name of io function to execute
///    @param[in] file_name: name of the file from which to read
///    @param[out] data: the data read from the file
//////////////////////////////////////////////////////////////////

template <class type>
int IoPython ::
    write_in_python (
        const string  function,
        const string  file_name,
        const type   &data      ) const
{

  try
  {
    py::initialize_interpreter ();
  }
  catch (...) { }


  const string io_folder = string (MAGRITTE_FOLDER) + "/src/Io";

  // Add /Io folder to Python path
  py::module::import("sys").attr("path").attr("insert")(0, io_folder.c_str());

  // Import function defined in implementation file
  py::object ioFunction = py::module::import(implementation.c_str()).attr(function.c_str());

  // Execute io function
  try
  {
    ioFunction (io_file, file_name, data);

    return (0);
  }
  catch (...)
  {
    return (-1);
  }


}
