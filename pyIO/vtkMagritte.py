import vtk


def get_neighbors(grid, cellNr):
    '''
    Get the numbers of the neighbouring cells of cell cellNr
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
