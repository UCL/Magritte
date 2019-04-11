// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <fstream>
using namespace std;

#include "io_text.hpp"


///  Constructor for IoText
///    @param[in] io_file: file to read from and write to
/////////////////////////////////////////////////////////

IoText ::
    IoText (
        const string io_file)
  : Io (io_file)
{


}   // END OF CONSTRUCTOR




///  get_length:
///  @param[in] file_name: path to file containing the data
///  @param[out] length: length to be read
///////////////////////////////////////////////////////////

int IoText ::
    read_length (
        const string file_name,
              long  &length) const
{

  length = 0;

  ifstream file (io_file + file_name + ".txt");

  string line;

  while (getline (file, line))
  {
    length++;
  }

  file.close();


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

  //length = 0;

  //ifstream file (io_file + file_name + ".txt");

  //string line;

  //while (getline (file, line))
  //{
  //  length++;
  //}

  //file.close();


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

  ifstream file (io_file + file_name + ".txt");

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

  ofstream file (io_file + file_name + ".txt");

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

  ifstream file (io_file + file_name + ".txt");

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

  ofstream file (io_file + file_name + ".txt");

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

int IoText ::
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




///  read_list:
///  @param[in] file_name: path to file containing the data
///  @param[in] list: list to be filled
///////////////////////////////////////////////////////////

int IoText ::
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

int IoText ::
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




///  read_list:
///  @param[in] file_name: path to file containing the data
///  @param[in] list: list to be filled
///////////////////////////////////////////////////////////

int IoText ::
    read_list (
        const string   file_name,
              String1 &list      ) const
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

int IoText ::
    write_list (
        const string   file_name,
        const String1 &list      ) const
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

int IoText ::
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

int IoText ::
    write_array (
        const string   file_name,
        const Long2   &array     ) const
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




///  read_array:
///  @param[in] file_name: path to file containing the data
///  @param[in] list: array to be filled
///////////////////////////////////////////////////////////

int IoText ::
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

int IoText ::
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

int IoText ::
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


  ofstream file (io_file + file_name + ".txt");

  for (long n = 0; n < length; n++)
  {
    file << x[n] << y[n] << z[n] << endl;
  }

  file.close();


  return (0);

}