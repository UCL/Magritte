import os
import sys
import vtk
import meshio
import numpy as np

from astropy                import constants
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
from scipy.spatial          import cKDTree
from tqdm                   import tqdm


def process_amrvac_input(config):

    if (os.path.splitext(config['input file'])[1].lower() != '.vtu'):
        raise ValueError('Only .vtu AMRVAC files are currently supported.')

    if (config['line producing species'][0] != 'CO'):
        raise NotImplementedError('amrvac input currently assumes that CO is the line producing species.')

    print("Reading amrvac input...")

    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(config['input file'])
    reader.Update()

    grid = reader.GetOutput()

    print("Extracting point data...")
    points = [grid.GetPoint(p) for p in tqdm(range(grid.GetNumberOfPoints()), file=sys.stdout)]
    points = np.array(points)

    print("Extracting cell data...")
    cellData = grid.GetCellData()

    for i in range(cellData.GetNumberOfArrays()):
        array = cellData.GetArray(i)
        if (array.GetName() == 'rho'):
            rho = vtk_to_numpy(array)
        if (array.GetName() == 'CO'):
            nCO = vtk_to_numpy(array)
        if (array.GetName() == 'H2'):
            nH2 = vtk_to_numpy(array)
        if (array.GetName() == 'temperature'):
            tmp = vtk_to_numpy(array)
        if (array.GetName() == 'v1'):
            v_x = vtk_to_numpy(array)
        if (array.GetName() == 'v2'):
            v_y = vtk_to_numpy(array)
        if (array.GetName() == 'v3'):
            v_z = vtk_to_numpy(array)

    # Convert to number densities [#/m^3]
    nH2 = rho * nH2 * 1.0e+6 * constants.N_A.si.value /  2.02
    nCO = rho * nCO * 1.0e+6 * constants.N_A.si.value / 28.01

    # Convert to fractions of the speed of light
    velocity = 1.0e-2 / constants.c.si.value * np.array((v_x, v_y, v_z)).transpose()

    trb = (150.0/constants.c.si.value)**2 * np.ones(grid.GetNumberOfCells())

    tetras     = []
    tetras_rho = []
    tetras_nH2 = []
    tetras_nCO = []
    tetras_tmp = []
    centres    = []

    print("Extracting cell centres for Magritte points... (might take a while)")
    for c in tqdm(range(grid.GetNumberOfCells()), file=sys.stdout):
        # a = vtk.vtkIdList()
        # b = vtk.vtkPoints()
        cell = grid.GetCell(c)
        # cell.Triangulate(0, a, b)
        # for i in range(int(a.GetNumberOfIds()/4)):
        #     tetras    .append([a.GetId(4*i+j) for j in range(4)])
        #     tetras_rho.append(rho[c])
        #     tetras_nH2.append(nH2[c])
        #     tetras_nCO.append(nCO[c])
        #     tetras_tmp.append(tmp[c])
        #     tetras_v_x.append(v_x[c])
        #     tetras_v_y.append(v_y[c])
        #     tetras_v_z.append(v_z[c])
        centre = np.zeros(3)
        for i in range(8):
            centre = centre + np.array(cell.GetPoints().GetPoint(i))
        centre = 0.125 * centre
        centres.append(centre)

    # tetras     = np.array(tetras)
    # tetras_rho = np.array(tetras_rho)
    # tetras_abn = np.array(tetras_abn)
    # tetras_tmp = np.array(tetras_tmp)
    # tetras_v_x = np.array(tetras_v_x)
    # tetras_v_y = np.array(tetras_v_y)
    # tetras_v_z = np.array(tetras_v_z)
    centres    = np.array(centres)

    (x_max, y_max, z_max) = 0.99999*np.max(centres, axis=0)
    (x_min, y_min, z_min) = 0.99999*np.min(centres, axis=0)

    print("Extracting boundary...")
    boundary = []
    for i, (x,y,z) in tqdm(enumerate(centres), file=sys.stdout):
        if not ((x_min < x < x_max) and
                (y_min < y < y_max) and
                (z_min < z < z_max)    ):
            boundary.append(i)
    boundary = np.array(boundary)

    print("Extracting neighbours...")
    neighbors = cKDTree(centres).query(centres, 26)[1]

    data = {'position'  : centres,
            'velocity'  : velocity,
            'boundary'  : boundary,
            'neighbors' : neighbors,
            'nH2'       : nH2,
            'nl1'       : nCO,
            'tmp'       : tmp,
            'trb'       : trb      }

    # cells     = {'tetra' : tetras}
    # cell_data = {'tetra' : {'rho' : tetras_rho,
    #                         'abn' : tetras_abn,
    #                         'tmp' : tetras_tmp,
    #                         'v_x' : tetras_v_x,
    #                         'v_y' : tetras_v_y,
    #                         'v_z' : tetras_v_z }}
    #
    # meshName = f"{config['project folder']}{config['model name']}.msh"
    #
    # if not config['overwrite files']:
    #     nr = 1
    #     while os.path.exists(meshName):
    #         meshName = f"{config['project folder']}{config['model name']}_{nr}.msh"
    #         nr += 1
    #
    # meshio.write_points_cells(meshName, points=points, cells=cells, cell_data=cell_data)
    #
    # print(f"Created mesh file:\n{meshName}\n from the input amrvac file {config['input file']}.")

    return data