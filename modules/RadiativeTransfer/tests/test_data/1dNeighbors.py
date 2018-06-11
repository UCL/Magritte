import numpy as np
import sys

ncells = int(sys.argv[1])


# Set up neighbors and nNeighbors

neighbors = []
nNeighbors = []

neighbors.append([1])
nNeighbors.append(1)

for p in range(1, ncells-1):
    neighbors.append([p-1, p])
    nNeighbors.append(2)

neighbors.append([ncells-2])
nNeighbors.append(1)


# Set up boundary

boundary = [0, ncells-1]


# Write out neighbors and boundary files

np.savetxt('n_neighbors.txt', nNeighbors, fmt='%ld')

with open('neighbors.txt', 'w') as file:
    for p in range(ncells):
        line = ''
        for neighbor in neighbors[p]:
            line += '{}\t'.format(neighbor)
        line += '\n'
        file.write(line)

np.savetxt('boundary.txt', boundary, fmt='%ld')
