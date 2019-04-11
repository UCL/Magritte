// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include "catch.hpp"

#include "Model/Lines/LineProducingSpecies/Lambda/lambda.hpp"


TEST_CASE ("Lambda", "[add_entry]")
{

  Lambda lambda;

  lambda.add_entry (1.0, 5);
  lambda.add_entry (1.0, 4);
  lambda.add_entry (1.0, 5);

  CHECK (lambda.Ls.size() == 2);
  CHECK (lambda.Ls.size() == lambda.nr.size());
  CHECK (lambda.Ls[0] == 2.0);
  CHECK (lambda.Ls[1] == 1.0);
  CHECK (lambda.nr[0] == 5);
  CHECK (lambda.nr[1] == 4);

}
