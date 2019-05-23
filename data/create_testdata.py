# Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
#
# Developed by: Frederik De Ceuster - University College London & KU Leuven
# _________________________________________________________________________

from os  import path as ospath
from sys import path as syspath

dir_path = ospath.dirname(ospath.realpath(__file__))
syspath.insert (0, f'{dir_path}/../bin')
syspath.insert (0, f'{dir_path}/../setup')

from setup import make_file_structure

import   magritte
import ioMagritte
import quadrature


def create_testdata_for_test_lineProducingSpecies ():
    # Define the model name
    modelName = "testdata/model_test_lineProducingSpecies/"
    # Create parameter and lineProducingSpecies objects
    parameters           = magritte.Parameters ()
    lineProducingSpecies = magritte.LineProducingSpecies ()
    # Set model variables
    parameters.set_ncells (1)
    parameters.set_nquads (1)
    lineProducingSpecies.ncells = parameters.ncells()
    lineProducingSpecies.linedata.nlev      = 2
    lineProducingSpecies.linedata.nrad      = 1
    lineProducingSpecies.linedata.ncolpar   = 0
    lineProducingSpecies.linedata.A         = magritte.Double1 ([1.0])
    lineProducingSpecies.linedata.Bs        = magritte.Double1 ([1.0])
    lineProducingSpecies.linedata.Ba        = magritte.Double1 ([1.0])
    lineProducingSpecies.linedata.irad      = magritte.Long1   ([1])
    lineProducingSpecies.linedata.jrad      = magritte.Long1   ([0])
    lineProducingSpecies.population         = magritte.Double1 ([0.0, 0.0])
    lineProducingSpecies.population_tot     = magritte.Double1 ([1.0])
    lineProducingSpecies.population_prev1   = magritte.Double1 ([1.0, 2.0])
    lineProducingSpecies.population_prev2   = magritte.Double1 ([2.0, 4.0])
    lineProducingSpecies.population_prev3   = magritte.Double1 ([3.0, 6.0])
    lineProducingSpecies.Jeff               = magritte.Double2 ([magritte.Double1([1.0])])
    lineProducingSpecies.quadrature.roots   = magritte.Double1 (quadrature.H_roots   (parameters.nquads()))
    lineProducingSpecies.quadrature.weights = magritte.Double1 (quadrature.H_weights (parameters.nquads()))
    # Write data
    io = ioMagritte.IoText (modelName)
    make_file_structure(modelName)
    parameters.write (io)
    lineProducingSpecies.write (io, 0)


#def create_testdata_for_test_lineProducingSpecies ():
#    model1D = Model ()

if __name__ == '__main__':
    # Main
    create_testdata_for_test_lineProducingSpecies ()
