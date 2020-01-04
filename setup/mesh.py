import numpy     as np
import meshio    as mio
import itertools as itt

def relocate_indices(arr, p):
    for i in range(len(arr)):
        for j in range(len(arr[i])):
            if (arr[i][j] > p):
                arr[i][j] -= 1
    return arr


class Mesh:

    def __init__(self, meshFile):
        mesh           = mio.read(meshFile)
        self.points    = mesh.points
        self.tetras    = mesh.cells['tetra']
        self.edges     = self.get_edges()
        self.neighbors = self.get_neighbors()
        # Remove non-connected points
        non_point = self.get_non_point()
        while not (non_point is None):
            self.del_non_point(non_point)
            non_point = self.get_non_point()

    def get_edges(self):
        edges = set([])
        for tetra in self.tetras:
            for (A,B) in itt.combinations(tetra,2):
                edges.add((A,B))
        return np.array(list(edges))

    def get_non_point(self):
        for p,nl in enumerate(self.neighbors):
            if (len(nl) == 0):
                return p

    def del_non_point(self, p):
        # remove the non-connected point
        self.points    = np.delete(self.points,    p, 0)
        self.neighbors = np.delete(self.neighbors, p, 0)
        # Change all indices, since a point got removed
        self.tetras    = relocate_indices(self.tetras,    p)
        self.edges     = relocate_indices(self.edges,     p)
        self.neighbors = relocate_indices(self.neighbors, p)

    def get_neighbors(self):
        neighbors = [[] for _ in range(len(self.points))]
        for edge in self.edges:
            neighbors[edge[0]].append(edge[1])
            neighbors[edge[1]].append(edge[0])
        return np.array(neighbors)

    def get_tetra_volume(self, tetra):
        a = self.points[tetra[0]]
        b = self.points[tetra[1]]
        c = self.points[tetra[2]]
        d = self.points[tetra[3]]
        return np.abs(np.dot(a-d,np.cross(b-d,c-d)))/6

    def get_tetra_volumes(self):
        return [self.get_tetra_volume(tetra) for tetra in self.tetras]

    def get_tetra_position(self, tetra):
        a = self.points[tetra[0]]
        b = self.points[tetra[1]]
        c = self.points[tetra[2]]
        d = self.points[tetra[3]]
        return 0.25*(a+b+c+d)

    def get_tetra_positions(self):
        return [self.get_tetra_position(tetra) for tetra in self.tetras]

    def get_edge_length(self, edge):
        a = self.points[edge[0]]
        b = self.points[edge[1]]
        r = b - a
        return np.sqrt(np.dot(r,r))

    def get_edge_lengths(self):
        return [self.get_edge_length(edge) for edge in self.edges]

    def get_edge_position(self, edge):
        a = self.points[edge[0]]
        b = self.points[edge[1]]
        return 0.5*(a+b)

    def get_edge_positions(self):
        return [self.get_edge_position(edge) for edge in self.edges]
