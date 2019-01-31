// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <iostream>
#include <fstream>
using namespace std;

#include <pybind11/embed.h> // everything needed for embedding
#include <pybind11/stl.h>
namespace py = pybind11;

#include "io_Python.hpp"


///  Constructor for IoPython
///    @param[in] io_file: file to read from and write to
/////////////////////////////////////////////////////////

IoPython ::
    IoPython (
        const string io_file)
  : Io (io_file)
{


}   // END OF CONSTRUCTOR




///  get_number:
///  @param[in] file_name: file containing the number
/////////////////////////////////////////////////////

long IoPython ::
    get_number (
        const string file_name) const
{

  py::scoped_interpreter guard{};
  py::module sys = py::module::import("sys");

  py::module io_Python = py::module::import("io_Python");
  py::object result    = io_Python.attr("get_number")(io_file + file_name + ".txt");

  long number = result.cast<long>();


  return number;

}




///  get_length:
///  @param[in] file_name: path to file containing the data
///////////////////////////////////////////////////////////

long IoPython ::
    get_length (
        const string file_name) const
{

  py::scoped_interpreter guard{};
  py::module sys = py::module::import("sys");

  py::module io_Python = py::module::import("io_Python");
  py::object result    = io_Python.attr("get_length")(io_file + file_name + ".txt");

  long number = result.cast<long>();


  return number;

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

  py::scoped_interpreter guard{};
  py::module sys = py::module::import("sys");
  py::object os  = py::module::import("os");

  py::module io_Python = py::module::import("io_Python");
  py::object result    = io_Python.attr("read_list")(io_file + file_name + ".txt");

  list = result.cast<Long1>();


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

  ofstream file (io_file + file_name + ".txt");

  for (long n = 0; n < list.size(); n++)
  {
    file << list[n] << endl;
  }

  file.close();


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

  ifstream file (io_file + file_name + ".txt");


  for (long n1 = 0; n1 < array.size(); n1++)
  {
    for (long n2 = 0; n2 < array[n1].size(); n2++)
    {
      file >> array[n1][n2];
    }
  }

  file.close();


  return (0);

}




///  write_array:
///  @param[in] file_name: path to file containing the data
///  @param[in] list: array to be written
///////////////////////////////////////////////////////////

int IoPython ::
    write_array (
        const string   file_name,
        const  Long2   &array     ) const
{

  ofstream file (io_file + file_name + ".txt");


  for (long n1 = 0; n1 < array.size(); n1++)
  {
    for (long n2 = 0; n2 < array[n1].size(); n2++)
    {
      file << array[n1][n2];
    }

    file << endl;
  }

  file.close();


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

  ifstream file (io_file + file_name + ".txt");

  long   n = 0;

  while (file >> list[n])
  {
    n++;
  }

  file.close();


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

  ofstream file (io_file + file_name + ".txt");

  for (long n = 0; n < list.size(); n++)
  {
    file << list[n] << endl;
  }

  file.close();


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

  ifstream file (io_file + file_name + ".txt");


  for (long n1 = 0; n1 < array.size(); n1++)
  {
    for (long n2 = 0; n2 < array[n1].size(); n2++)
    {
      file >> array[n1][n2];
    }
  }

  file.close();


  return (0);

}




///  write_array:
///  @param[in] file_name: path to file containing the data
///  @param[in] list: array to be written
///////////////////////////////////////////////////////////

int IoPython ::
    write_array (
        const string   file_name,
        const Double2 &array     ) const
{

  ofstream file (io_file + file_name + ".txt");


  for (long n1 = 0; n1 < array.size(); n1++)
  {
    for (long n2 = 0; n2 < array[n1].size(); n2++)
    {
      file << array[n1][n2];
    }

    file << endl;
  }

  file.close();


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

  ifstream file (io_file + file_name + ".txt");

  long   n = 0;

  while (file >> x[n] >> y[n] >> z[n])
  {
    n++;
  }

  file.close();


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


  ofstream file (io_file + file_name + ".txt");

  for (long n = 0; n < length; n++)
  {
    file << x[n] << y[n] << z[n] << endl;
  }

  file.close();


  return (0);

}
