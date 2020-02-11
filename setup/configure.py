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
from input        import process_mesher_input, process_amrvac_input, process_phantom_input


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
    if (config['ray mode'] == 'single'):
        if (config['nrays'] != 1):
            raise ValueError('If ray mode = single, nrays should be equal to 1.')
        if (len(config['ray']) == 3):
            config['ray'] = np.array(config['ray'])
            # Normalize the ray vector
            config['ray'] = config['ray'] / np.linalg.norm(config['ray'])
        else:
            raise ValueError('If ray mode = single, the ray vector should be specified as [n_x, n_y, n_z].')
    # Set currently not implemented defaults
    config['dimension'] = 3
    config['nspecs'   ] = 5
    config['nlspecs'  ] = 1
    # Return config dict
    return config


def configure_simulation(config) -> Simulation():
    '''
    :param config:
    :param data:
    :return: configured simulation object
    '''
    print("Creating and configuring a Magritte simulation object...")
    dataName = f"{config['project folder']}{config['model name']}_"
    # Define simulation and setup objects
    simulation = Simulation()
    setup      = Setup(dimension=config['dimension']) # TODO: Remove setup object
    # Check the lengths of the data elements
    # for name, array in data.items():
    #     if (array.shape[0] != ncells):
    #         raise ValueError(f'Shape of {name} does not correspond to the number of points.')
    # Set parameters
    simulation.parameters.set_nrays          (config['nrays'     ])
    simulation.parameters.set_nspecs         (config['nspecs'    ])
    simulation.parameters.set_nlspecs        (config['nlspecs'   ])
    simulation.parameters.set_nquads         (config['nquads'    ])
    simulation.parameters.set_use_scattering (config['scattering'])
    simulation.parameters.set_pop_prec       (config['pop_prec'  ])
    simulation.parameters.n_off_diag       = (config['n_off_diag'])
    # Set position and velocity
    simulation.geometry.cells.position = np.load(dataName+'position.npy')
    simulation.geometry.cells.velocity = np.load(dataName+'velocity.npy')
    # Get ncells from the mesh
    ncells = len(simulation.geometry.cells.position)
    simulation.parameters.set_ncells (ncells)
    # Set neighbors (connections between points)
    print("Setting the neighbors...")
    # Make neighbors into a rectangular array (required for hdf5)
    # Extract the number of neighbors for each point
    neighbors = np.load(dataName+'neighbors.npy', allow_pickle=True)
    n_nbs    = [len (nb) for nb in neighbors]
    nbs_rect = np.zeros((ncells, np.max(n_nbs)), dtype=int).tolist()
    for p in range(ncells):
        for (i,nb) in enumerate(neighbors[p]):
            nbs_rect[p][i] = nb
    simulation.geometry.cells.n_neighbors = n_nbs
    simulation.geometry.cells.neighbors   = nbs_rect
    # Set boundary
    print("Setting the boundary...")
    simulation.geometry.boundary.boundary2cell_nr = np.load(dataName+'boundary.npy')
    # Set rays
    if (config['ray mode'] == 'single'):
        simulation.geometry.rays.rays    = [np.array(config['ray'])]
        simulation.geometry.rays.weights = [1.0]
    else:
        simulation.geometry.rays = setup.rays (nrays=config['nrays'], cells=simulation.geometry.cells)
    # Set thermodynamics
    print("Setting temperature and turbulence...")
    simulation.thermodynamics.temperature.gas   = np.load(dataName+'tmp.npy')
    simulation.thermodynamics.turbulence.vturb2 = np.load(dataName+'trb.npy')
    # Set Chemistry
    simulation.chemistry.species.abundance = np.array((np.zeros(ncells), np.load(dataName+'nl1.npy'), np.load(dataName+'nH2.npy'), np.zeros(ncells), np.ones(ncells))).transpose()
    simulation.chemistry.species.sym       = ['dummy0', config['line producing species'][0], 'H2', 'e-', 'dummy1']

    dataFile = f"{config['data folder']}{config['line data files'][0]}"
    simulation.lines.lineProducingSpecies.append (
        linedata_from_LAMDA_file (dataFile, simulation.chemistry.species, config))

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
    # (Writing and reading again is required to guarantee a proper setup)
    print("Writing out magritte model:", modelName)
    simulation.write(io)
    # Remove old simulation object from memory
    del simulation
    # Read the newly written simulation object
    print("Reading in magritte model to extract simulation object.")
    simulation_new = Simulation()
    simulation_new.read(io)
    # Return the newly read simulation object
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
    simulation = Simulation()
    simulation.read(io)
    return simulation


def configure(config) -> Simulation:
    """
    :param config: a config dict extracted from a configuration file
    :return: a configured Magritte simulation object
    """
    print("Configuring Magritte...")
    # Check the input type
    if   (config['input type'].lower() == 'magritte'):
        return process_magritte_input(config)
    elif (config['input type'].lower() == 'mesher'):
        data = process_mesher_input(config)
    elif (config['input type'].lower() == 'amrvac'):
        data = process_amrvac_input(config)
    elif (config['input type'].lower() == 'phantom'):
        process_phantom_input(config)
    else:
        raise ValueError('Please specify a valid input type (magritte, mesher, amrvac or phantom).')
    # return a configured simulation object
    return configure_simulation(config=config)


if (__name__ == '__main__'):
    # Check if a configuration file was specified
    if (len(sys.argv) < 2):
        raise ValueError('Please specify a configuration file.')
    else:
        config = read_config(sys.argv[1])
    # Configure a simulation object
    simulation = configure(config)


