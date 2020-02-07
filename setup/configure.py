import os
import sys

sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}")
sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}/../bin/")

import yaml
import numpy as np

from shutil       import rmtree
from magritte     import Simulation
from setup        import Setup, linedata_from_LAMDA_file, make_file_structure
from quadrature   import H_roots, H_weights
from ioMagritte   import IoPython, IoText
from amrvac_input import process_amrvac_input


def read_config(config_file) -> dict:
    """
    Read the Magritte config file and chack for validity.
    @param[in] config_file:
    @returns a config dict
    """
    print("Reading the config file...")
    # Read the yaml configuration file
    with open(config_file, 'r') as file:
        config = yaml.load(file, Loader=yaml.Loader)
    # Check the validity of the configuration file
    if (config['line data files'] is None) or (len(config['line data files']) == 0):
        raise ValueError('Please specify at least one line data file.')
    if (len(config['line data files']) > 1):
        raise NotImplementedError('Configuration for more than one line data file is not yet implemented.')
    if (len(config['line data files']) != len(config['line producing species'])):
        raise ValueError('Please specify the line producing species for each line data file.')
    if (config['nquads' ] <= 0):
        raise ValueError('nquads should be a positive integer.')
    if (config['nrays'  ] <= 0):
        raise ValueError('nrays should be a positive integer.')
    # Set currently not implemented defaults
    config['dimension'] = 3
    config['nspecs'   ] = 5
    config['nlspecs'  ] = 1
    # Return config dict
    return config


def configure_simulation(config, data) -> Simulation():
    '''
    :param config:
    :param data:
    :return: configured simulation object
    '''
    print("Creating and configuring a Magritte simulation object...")
    # Define simulation and setup objects
    simulation = Simulation()
    setup      = Setup(dimension=config['dimension']) # TODO: Remove setup object
    # Get ncells from the mesh
    ncells = data['position'].shape[0]
    # Check the lengths of the data elements
    # for name, array in data.items():
    #     if (array.shape[0] != ncells):
    #         raise ValueError(f'Shape of {name} does not correspond to the number of points.')
    # Set parameters
    simulation.parameters.set_ncells         (ncells)
    simulation.parameters.set_nrays          (config['nrays'     ])
    simulation.parameters.set_nspecs         (config['nspecs'    ])
    simulation.parameters.set_nlspecs        (config['nlspecs'   ])
    simulation.parameters.set_nquads         (config['nquads'    ])
    print(config['scattering'], type(config['scattering']))
    simulation.parameters.set_use_scattering (config['scattering'])
    simulation.parameters.set_pop_prec       (config['pop_prec'  ])
    simulation.parameters.n_off_diag       = (config['n_off_diag'])
    # Set position and velocity
    simulation.geometry.cells.position = data['position']
    simulation.geometry.cells.velocity = data['velocity']
    # Set neighbors (connections between points)
    simulation.geometry.cells.n_neighbors = [len(nb) for nb in data['neighbors']]
    simulation.geometry.cells.neighbors   =                    data['neighbors']
    # Set boundary
    simulation.geometry.boundary.boundary2cell_nr = data['boundary']
    # Set rays
    simulation.geometry.rays = setup.rays (nrays=config['nrays'], cells=simulation.geometry.cells)
    # Set thermodynamics
    simulation.thermodynamics.temperature.gas   = data['tmp']
    simulation.thermodynamics.turbulence.vturb2 = data['trb']
    # Set Chemistry
    abundances = np.array((np.zeros(ncells), data['nl1'], data['nH2'], np.zeros(ncells), np.ones(ncells))).transpose()
    simulation.chemistry.species.abundance = abundances
    simulation.chemistry.species.sym       = ['dummy0', config['line producing species'][0], 'H2', 'e-', 'dummy1']

    simulation.lines.lineProducingSpecies.append (
        linedata_from_LAMDA_file (f"{config['data folder']}{config['line data files'][0]}", simulation.chemistry.species))

    simulation.lines.lineProducingSpecies[0].quadrature.roots   = H_roots   (config['nquads'])
    simulation.lines.lineProducingSpecies[0].quadrature.weights = H_weights (config['nquads'])

    # Define the model name
    modelName = name = f"{config['project folder']}{config['model name']}"
    # Change the model name if we do not want to overwrite
    if not config['overwrite files']:
        nr = 1
        while os.path.exists(modelName):
            modelName = f"{name}_{nr}"
            nr += 1
    # Setup for writing hdf5 output
    if   (config['model type'].lower() in ['hdf5', 'h5']):
        modelName = f"{modelName}.hdf5"
        try:
            os.remove(modelName)
        except:
            pass
        io = IoPython('hdf5', modelName)
    # Setup for writing ascii output
    elif (config['model type'].lower() in ['text', 'txt', 'ascii']):
        modelName = f"{modelName}/"
        try:
            rmtree(modelName)
        except:
            pass
        io = IoText(modelName)
        make_file_structure (model=simulation, modelName=modelName)
    # Non valid model type
    else:
        raise ValueError('No valid model type was given (hdf5, ascii).')
    # Write the simulation data using the io interface
    print("Writing out magritte model:", modelName)
    simulation.write(io)
    del simulation
    simulation_new = Simulation()
    simulation_new.read(io)
    # Return the newly read simulation object
    print("Reading in magritte model to extract simulation object.")
    # (Writing and reading again is required to guarantee a proper setup)
    return simulation_new


def process_magritte_input (config) -> Simulation():
    # Define model name
    modelName = f"{config['project folder']}{config['model name']}"
    # Choose the io interface corresponding to the model type
    if   (config['model type'].lower() in ['hdf5', 'h5']):
        io = IoPython('hdf5', f"{modelName}.hdf5")
    elif (config['model type'].lower() in ['text', 'txt', 'ascii']):
        io = IoText(f"{modelName}/")
    else:
        raise ValueError('No valid model type was given (hdf5, ascii).')
    print("Reading in magritte model to extract simulation object.")
    return Simulation().read(io)


def configure(config) -> Simulation:
    """
    :param config: a config dict extracted from a configuration file
    :return: a configured Magritte simulation object
    """
    print("Configuring Magritte...")
    # Check the input type
    if   (config['input type'].lower() == 'magritte'):
        return process_magritte_input(config)
    elif (config['input type'].lower() == 'amrvac'):
        data = process_amrvac_input(config)
    else:
        ValueError('Please specify a valid input type (magritte, amrvac)')
    # return a configured simulation object
    return configure_simulation(config=config, data=data)


if (__name__ == '__main__'):
    # Check if a configuration file was specified
    if (len(sys.argv) < 2):
        raise ValueError('Please specify a configuration file.')
    else:
        config = read_config(sys.argv[1])
    # Configure a simulation object
    simulation = configure(config)


