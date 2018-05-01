import vtk

def read_grid(fileName):
    '''
    Read .vtu file and return it as a vtk type grid
    '''
    reader = vtk.vtkXMLUnstructuredGridReader()   # .vtu reader
    reader.SetFileName(fileName)
    reader.Update()
    grid = reader.GetOutput()
    return grid


def write_grid(grid):
    '''
    Write vtk type grid to magrid.vtu
    '''
    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName('magrid.vtu')
    writer.SetInputData(grid)
    writer.Write()


def add_neighbor_data(grid):
    '''
    Add neighbors lists and number of neigbors as data arrays to vtk grid
    '''
    ncells = grid.GetNumberOfCells()
    # Get neighbors and number of neighbors for all cells
    neighbors_t     = [get_neighbors(grid,p) for p in range(ncells)]
    n_neighbors     = [len(neighbors_t[p])   for p in range(ncells)]
    max_n_neighbors = max(n_neighbors)
    # Create rectangular array for neighbors
    neighbors = [[-1 for _ in range(max_n_neighbors)] for _ in range(ncells)]
    for p in range(ncells):
        for n in range(len(neighbors_t[p])):
            neighbors[p][n] = neighbors_t[p][n]
    # Get cell data
    cellData = grid.GetCellData()
    # Create vtk array for neighbors lists
    vtk_neighbors = vtk.vtkLongArray()
    vtk_neighbors.SetNumberOfComponents(max_n_neighbors)
    vtk_neighbors.SetNumberOfTuples(ncells)
    vtk_neighbors.SetName('neighbors')
    # Create vtk array for number of neighbors
    vtk_n_neighbors = vtk.vtkLongArray()
    vtk_n_neighbors.SetNumberOfComponents(1)
    vtk_n_neighbors.SetNumberOfTuples(ncells)
    vtk_n_neighbors.SetName('n_neighbors')
    # Put data in vtk arrays
    for p in range(ncells):
        vtk_neighbors.InsertTuple(p,neighbors[p])
        vtk_n_neighbors.InsertValue(p,n_neighbors[p])
    # Add vtk arrays to cell data
    cellData.AddArray(vtk_neighbors)
    cellData.AddArray(vtk_n_neighbors)


def get_neighbors(grid, cellNr):
    '''
    Get numbers of neighbouring cells of cell cellNr
    '''
    neighbors = []
    # Get the points surrounding the cell with nr cellNr
    surroundingPoints = vtk.vtkIdList()
    grid.GetCellPoints(cellNr, surroundingPoints)
    # For each of the surrounding points
    for p in range(surroundingPoints.GetNumberOfIds()):
        # Extract one of the surrounding points (and put in vtkIdList)
        oneSurroundingPoint = vtk.vtkIdList()
        oneSurroundingPoint.SetNumberOfIds(1)
        oneSurroundingPoint.SetId(0,surroundingPoints.GetId(p))
        # Find other cells surrounding that one point
        otherCellsAroundPoint = vtk.vtkIdList()
        grid.GetCellNeighbors(cellNr, oneSurroundingPoint, otherCellsAroundPoint)
        nOtherCells = otherCellsAroundPoint.GetNumberOfIds()
        # Put the Id's of these cells in the neighbors list
        for n in range(nOtherCells):
            neighbors.append(otherCellsAroundPoint.GetId(n))
    # Remove duplicates
    neighbors = list(set(neighbors))
    # Return list with cell numbers of neighbors of cell cellNr
    return neighbors
