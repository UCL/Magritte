import sys
import numpy as np

# Add package directory to path
sys.path.append(f'{sys.path[0]}/../../')

# Add magritte functions
from magritte.setup.configure import get_io, get_simulation


def max_relative_difference(a, b):
    return np.max(2.0 * np.abs((a - b) / (a + b)))


def run_iterations(model_name, n_off_diag=0, iterations=50):

    # Extract io and simulation objects
    io         = get_io        (model_name)         # Magritte io object
    simulation = get_simulation(model_name)         # Magritte simulation object

    simulation.parameters.n_off_diag = n_off_diag   # Set ALO bandwidth

    simulation.compute_spectral_discretisation()    # Initialize frequency bins
    simulation.compute_LTE_level_populations()      # Set all levels to LTE value

    for _ in range(5):
        # Compute radiation field and resulting populations
        simulation.compute_radiation_field_cpu()
        simulation.compute_Jeff()
        simulation.compute_level_populations_from_stateq()

    pop = np.copy(simulation.lines.lineProducingSpecies[0].population)

    for iteration in range(iterations):
        # Compute radiation field and resulting populations
        simulation.compute_radiation_field_cpu()
        simulation.compute_Jeff()
        simulation.compute_level_populations_from_stateq()
        # Compute the maximum relative change
        pop_prev = np.copy(pop)
        pop      = np.copy(simulation.lines.lineProducingSpecies[0].population)
        max_diff = max_relative_difference(pop, pop_prev)
        # Check for convergence
        if (max_diff < 1.0e-6):
            break
        print('After', iteration, 'iteration(s), max_diff =', max_diff)

    # Write the result
    simulation.write(io)


if (__name__ == "__main__"):

    # Run iterations for the input model
    if (len(sys.argv) < 2):
        print('Please provide a model file.')
    elif (len(sys.argv) == 3):
        # Extract model name
        model_name = str(sys.argv[1])
        # Extract the ALO bandwidth
        n_off_diag = int(sys.argv[2])
        run_iterations(model_name, n_off_diag=n_off_diag)
    elif (len(sys.argv) == 4):
        # Extract model name
        model_name = str(sys.argv[1])
        # Extract the ALO bandwidth
        n_off_diag = int(sys.argv[2])
        # Extract max number of iterations
        iterations = int(sys.argv[3])
        run_iterations(model_name, n_off_diag=n_off_diag, iterations=iterations)
    else:
        # Extract model name
        model_name = str(sys.argv[1])
        run_iterations(model_name)
