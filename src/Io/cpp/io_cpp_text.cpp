// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <fstream>
#include <sys/stat.h>
#include <iomanip>

#include "io_cpp_text.hpp"
#include "Tools/logger.hpp"


///  Constructor for IoText
///    @param[in] io_file: file to read from and write to
/////////////////////////////////////////////////////////

IoText ::
    IoText (
        const string io_file)
  : Io (io_file)
{


}   // END OF CONSTRUCTOR




///  pathExist: returns true if the given path exists
///  @param[in] path: path to check
/////////////////////////////////////////////////////

bool pathExist (
    const string &path)
{
  struct stat buffer;

  return (stat (path.c_str(), &buffer) == 0);
}




///  get_length:
///  @param[in] file_name: path to file containing the data
///  @param[out] length: length to be read
///////////////////////////////////////////////////////////

int IoText ::
    read_length (
        const string file_name,
              long  &length) const
{

  string fname = io_file + file_name;

  length = 0;


  if (pathExist (fname + ".txt"))
  {
    std::ifstream file (fname + ".txt");

    string line;

    while (std::getline (file, line))
    {
      length++;
    }

    file.close();
  }

  else
  {
    while (pathExist (fname + std::to_string (length)))
    {
      length++;
    }
  }


  return (0);

}




///  get_length:
///  @param[in] file_name: path to file containing the data
///  @param[out] length: length to be read
///////////////////////////////////////////////////////////

int IoText ::
    read_width (
        const string file_name,
              long  &width     ) const
{

  string fname = io_file + file_name;

  width = 0;


  if (pathExist (fname + ".txt"))
  {
    std::ifstream file (io_file + file_name + ".txt");

    string line, elem;

    std::getline (file, line);

    std::stringstream ss (line);

    while (std::getline (ss, elem, '\t'))
    {
      width++;
    }

    file.close();
  }

  else
  {
    while (pathExist (fname + std::to_string (width)))
    {
      width++;
    }
  }


  return (0);

}




///  read_number:
///  @param[in] file_name: file containing the number
///  @param[out] number: number to be read
/////////////////////////////////////////////////////

int IoText ::
    read_number (
        const string file_name,
              long  &number    ) const
{

  std::ifstream file (io_file + file_name + ".txt");

  file >> number;

  file.close();


  return (0);

}




///  write_number:
///  @param[in] file_name: file containing the number
///  @param[out] number: number to be written
/////////////////////////////////////////////////////

int IoText ::
    write_number (
        const string file_name,
        const long  &number    ) const
{

  std::ofstream file (io_file + file_name + ".txt");

  file << std::scientific << std::setprecision (16);

  file << number;

  file.close();


  return (0);

}




///  read_number:
///  @param[in] file_name: file containing the number
///  @param[out] number: number to be read
/////////////////////////////////////////////////////

int IoText ::
    read_number (
        const string  file_name,
              double &number    ) const
{

  std::ifstream file (io_file + file_name + ".txt");

  file >> number;

  file.close();


  return (0);

}




///  write_number:
///  @param[in] file_name: file containing the number
///  @param[out] number: number to be written
/////////////////////////////////////////////////////

int IoText ::
    write_number (
        const string  file_name,
        const double &number    ) const
{

  std::ofstream file (io_file + file_name + ".txt");

  file << std::scientific << std::setprecision (16);

  file << number;

  file.close();


  return (0);

}




///  read_word:
///  @param[in] file_name: file containing the number
/////////////////////////////////////////////////////

int IoText ::
    read_word  (
        const string  file_name,
              string &word      ) const
{

  std::ifstream file (io_file + file_name + ".txt");

  file >> word;

  file.close();


  return (0);

}




///  write_word:
///  @param[in] file_name: file containing the number
/////////////////////////////////////////////////////

int IoText ::
    write_word  (
        const string  file_name,
        const string &word      ) const
{

  std::ofstream file (io_file + file_name + ".txt");

  file << word;

  file.close();


  return (0);

}




///  read_list:
///  @param[in] file_name: path to file containing the data
///  @param[in] list: list to be filled
///////////////////////////////////////////////////////////

int IoText ::
    read_list (
        const string file_name,
              Long1 &list      ) const
{

  std::ifstream file (io_file + file_name + ".txt");

  long n = 0;

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

int IoText ::
    write_list (
        const string file_name,
        const Long1 &list      ) const
{

  std::ofstream file (io_file + file_name + ".txt");

  file << std::scientific << std::setprecision (16);

  for (long n = 0; n < list.size(); n++)
  {
    file << list[n] << endl;
  }

  file.close();


  return (0);

}




///  read_list:
///  @param[in] file_name: path to file containing the data
///  @param[in] list: list to be filled
///////////////////////////////////////////////////////////

int IoText ::
    read_list (
        const string   file_name,
              Double1 &list      ) const
{

  std::ifstream file (io_file + file_name + ".txt");

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

int IoText ::
    write_list (
        const string   file_name,
        const Double1 &list      ) const
{

  std::ofstream file (io_file + file_name + ".txt");

  file << std::scientific << std::setprecision (16);

  for (long n = 0; n < list.size(); n++)
  {
    file << list[n] << endl;
  }

  file.close();


  return (0);

}




///  read_list:
///  @param[in] file_name: path to file containing the data
///  @param[in] list: list to be filled
///////////////////////////////////////////////////////////

int IoText ::
    read_list (
        const string   file_name,
              String1 &list      ) const
{

  std::ifstream file (io_file + file_name + ".txt");

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

int IoText ::
    write_list (
        const string   file_name,
        const String1 &list      ) const
{

  std::ofstream file (io_file + file_name + ".txt");

  file << std::scientific << std::setprecision (16);

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

int IoText ::
    read_array (
        const string   file_name,
              Long2   &array     ) const
{

  std::ifstream file (io_file + file_name + ".txt");

  string line;


  for (long n1 = 0; n1 < array.size(); n1++)
  {
    std::getline (file, line);

    std::stringstream ss (line);

    for (long n2 = 0; n2 < array[n1].size(); n2++)
    {
      ss >> array[n1][n2];
    }
  }

  file.close();


  return (0);

}




///  write_array:
///  @param[in] file_name: path to file containing the data
///  @param[in] list: array to be written
///////////////////////////////////////////////////////////

int IoText ::
    write_array (
        const string   file_name,
        const Long2   &array     ) const
{

  std::ofstream file (io_file + file_name + ".txt");

  file << std::scientific << std::setprecision (16);

  for (long n1 = 0; n1 < array.size(); n1++)
  {
    for (long n2 = 0; n2 < array[n1].size(); n2++)
    {
      file << array[n1][n2] << "\t";
    }

    file << endl;
  }

  file.close();


  return (0);

}




///  read_array:
///  @param[in] file_name: path to file containing the data
///  @param[in] list: array to be filled
///////////////////////////////////////////////////////////

int IoText ::
    read_array (
        const string   file_name,
              Double2 &array     ) const
{

  std::ifstream file (io_file + file_name + ".txt");

  string line;

  for (long n1 = 0; n1 < array.size(); n1++)
  {
    std::getline (file, line);

    std::stringstream ss (line);

    for (long n2 = 0; n2 < array[n1].size(); n2++)
    {
      ss >> array[n1][n2];
    }
  }

  file.close();


  return (0);

}




///  write_array:
///  @param[in] file_name: path to file containing the data
///  @param[in] list: array to be written
///////////////////////////////////////////////////////////

int IoText ::
    write_array (
        const string   file_name,
        const Double2 &array     ) const
{

  std::ofstream file (io_file + file_name + ".txt");

  file << std::scientific << std::setprecision (16);

  for (long n1 = 0; n1 < array.size(); n1++)
  {
    for (long n2 = 0; n2 < array[n1].size(); n2++)
    {
      file << array[n1][n2] << "\t";
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

int IoText ::
    read_3_vector (
        const string   file_name,
              Double1 &x,
              Double1 &y,
              Double1 &z         ) const
{

  std::ifstream file (io_file + file_name + ".txt");

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

int IoText ::
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


  std::ofstream file (io_file + file_name + ".txt");

  file << std::scientific << std::setprecision (16);


  for (long n = 0; n < length; n++)
  {
    file << x[n] << "\t" << y[n] << "\t" << z[n] << endl;
  }

  file.close();


  return (0);

}
