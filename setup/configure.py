import os
import sys

sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}")
sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}/../bin/")

import yaml
import numpy as np

from shutil     import rmtree
from copy       import deepcopy
from magritte   import Simulation, IoPython, IoText
from setup      import Setup, linedata_from_LAMDA_file, make_file_structure
from quadrature import H_roots, H_weights
from input      import *
from rays       import setup_rays_spherical_symmetry




config_default = {
    'project folder'     : '',          # Use current folder as project folder
    'data folder'        : '',          # Use current folder as data folder
    'overwrite files'    : True,        # Overwrite output files or not
    'solver mode'        : 'CPU',       # Use CPU or GPU solver
    'nquads'             : 5,           # number of (Gauss-Hermite) quadrature points
    'nrays'              : 108,         # number of rays originating form each point (nrays = 12*n^2)
    'ray mode'           : 'uniform',   # ray distributions
    'scattering'         : False,       # use scattering (hence store the complete radiation field)
    'pop_prec'           : 1.0E-6,      # convergence criterion for level populations (max relative change)
    'n_off_diag'         : 0,           # width of the Approximated Lambda Operator (ALO)}
}




def write_config(config):
    """
    Write the Magritte config file.
    :param config: the config dict.
    """
    name, extension = os.path.splitext(config['model name'])
    config_file = f"{config['project folder']}{name}_config.yaml"
    with open(config_file, 'w') as file:
        file.writelines(yaml.dump(config, Dumper=yaml.Dumper))
    return




def read_config(config_file) -> dict:
    """
    Read the Magritte config file and chack for validity.
    :param config_file: configuration file.
    :returns a config dict.
    """
    # Initialize with the default config
    config = deepcopy(config_default)
    # Read the yaml configuration file
    with open(config_file, 'r') as file:
        config_new = yaml.load(file, Loader=yaml.Loader)
    # Overwrite the defaults with the new values
    for key in config_new:
        config[key] = config_new[key]
    # Return config dict
    return config




def configure_simulation(config) -> Simulation():
    """"
    :param config:
    :return: configured simulation object
    """
    print("Creating and configuring a Magritte simulation object...")
    dataName = f"{config['project folder']}{config['model name']}_"
    # Define simulation and setup objects
    simulation = Simulation()
    setup      = Setup(dimension=config['dimension']) # TODO: Remove setup object
    # Set parameters
    simulation.parameters.set_nspecs             (config['nspecs'    ])
    simulation.parameters.set_nlspecs            (config['nlspecs'   ])
    simulation.parameters.set_nquads             (config['nquads'    ])
    simulation.parameters.set_use_scattering     (config['scattering'])
    simulation.parameters.set_pop_prec           (config['pop_prec'  ])
    simulation.parameters.set_spherical_symmetry (config['input type'] == 'spherically symmetric')
    simulation.parameters.n_off_diag           = (config['n_off_diag'])
    # Set position and velocity
    simulation.geometry.cells.position = np.load(dataName+'position.npy')
    simulation.geometry.cells.velocity = np.load(dataName+'velocity.npy')
    # Get ncells from the mesh
    ncells = len(simulation.geometry.cells.position)
    simulation.parameters.set_ncells (ncells)
    # Set neighbors (connections between points)
    print("Setting the neighbors...")
    simulation.geometry.cells.neighbors = np.load(dataName+'neighbors.npy', allow_pickle=True)
    # Set boundary
    print("Setting the boundary...")
    simulation.geometry.boundary.boundary2cell_nr = np.load(dataName+'boundary.npy')
    # Set rays
    if   (config['ray mode'] == 'single'):
        simulation.parameters.set_adaptive_ray_tracing(False)
        simulation.parameters.set_nrays    (2)
        simulation.geometry.rays.rays    = [np.array(config['ray']), -np.array(config['ray'])]
        simulation.geometry.rays.weights = [0.5, 0.5]
    elif (config['ray mode'] == 'uniform'):
        simulation.parameters.set_adaptive_ray_tracing(False)
        simulation.parameters.set_nrays (config['nrays'])
        if (config['input type'] == 'spherically symmetric'):
            simulation.geometry.rays = setup_rays_spherical_symmetry(nextra=config['nrays']//2-1)
        else:
            simulation.geometry.rays = setup.rays (nrays=config['nrays'], cells=simulation.geometry.cells)
    elif (config['ray mode'] == 'adaptive'):
        simulation.parameters.set_adaptive_ray_tracing(True)
        simulation.parameters.order_min(config['order min'])
        simulation.parameters.order_max(config['order max'])
    else:
        raise ValueError('Please specify a valid ray mode (single | uniform | adaptive).')
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
    # Set io used to create model as default
    config['default io'] = io
    # Write the simulation data using the io interface
    # (Writing and reading again is required to guarantee a proper setup)
    print("Writing out magritte model:", modelName)
    simulation.write(io)
    # Remove old simulation object from memory
    del simulation
    # Read the newly written simulation object
    print("Reading in magritte model to extract simulation object.")
    try:
        simulation_new = Simulation()
        simulation_new.read(io)
    except:
        print("Failed to read simulation!")
        simulation_new = Simulation()

    # Remove the data
    os.remove(dataName+'position.npy')
    os.remove(dataName+'velocity.npy')
    os.remove(dataName+'neighbors.npy')
    os.remove(dataName+'boundary.npy')
    os.remove(dataName+'tmp.npy')
    os.remove(dataName+'trb.npy')
    os.remove(dataName+'nl1.npy')
    os.remove(dataName+'nH2.npy')

    # Return the newly read simulation object
    return simulation_new

def get_io(model_name):
    # Extract the file extension (all lowercase)
    extension = os.path.splitext(model_name)[1][1:].lower()
    # Determine the io based on the extension
    if extension in ['hdf5', 'h5']:
        return IoPython('hdf5', model_name)
    elif extension in ['text', 'txt', 'ascii']:
        return IoText(model_name)
    else:
        raise ValueError('No valid model type was given (valid types ars: hdf5 and ascii).')


def get_simulation(model_name) -> Simulation():
    simulation = Simulation()
    simulation.read(get_io(model_name))
    return simulation


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
    # Set io used to create model as default
    config['default io'] = io
    print("Reading in magritte model to extract simulation object.")
    simulation = Simulation()
    simulation.read(io)
    return simulation


def preconfig(config) -> dict:
    """
    :param config: a config dict extracted from a configuration file
    :returns a preconfigured configuration dict
    """
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
        print("Warning: In single ray mode, nrays is not take into account and set to 2 (since we consider ray pairs).")
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

    if not 'model type' in config:
        name, extension = os.path.splitext(config['model name'])
        if (extension == ''):
            raise ValueError('Please specify a model type or provide a valid file extension for the model name.')
        print('Extracted model name:', name)
        print('Extracted model type:', extension[1:])
        config['model name'] = name
        config['model type'] = extension[1:]

    if ('original model name' in config) and not ('original model type' in config):
        name, extension = os.path.splitext(config['original model name'])
        if (extension == ''):
            raise ValueError('Please specify a model type or provide a valid file extension for the model name.')
        print('Extracted original model name:', name)
        print('Extracted original model type:', extension[1:])
        config['original model name'] = name
        config['original model type'] = extension[1:]

    # Done
    return config




def configure(config) -> Simulation:
    """
    :param config: a config dict extracted from a configuration file
    :return: a configured Magritte simulation object
    """
    print("Configuring Magritte...")
    # Run preconfig
    config = preconfig(config)
    # Check the input type
    if   (config['input type'].lower() == 'magritte'):
        return process_magritte_input(config)
    elif (config['input type'].lower() == 'analytic'):
        process_analytic_input(config)
    elif (config['input type'].lower() == 'analytic geometry'):
        process_analytic_input_with_geometry(config)
    elif (config['input type'].lower() == 'mesher'):
        process_mesher_input(config)
    elif (config['input type'].lower() == 'amrvac'):
        process_amrvac_input(config)
    elif (config['input type'].lower() == 'phantom'):
        process_phantom_input(config)
    elif (config['input type'].lower() == 'spherically symmetric'):
        process_spherically_symmetric_input(config)
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


