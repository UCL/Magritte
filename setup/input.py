import os
import sys
import vtk
import meshio
import numpy as np

from astropy                import units, constants
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
from scipy.spatial          import cKDTree, Delaunay
from tqdm                   import tqdm

sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}")
sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}/../bin/")

from mesher   import Mesh
from magritte import Model, IoPython, IoText


def process_spherically_symmetric_input(config):
    """
    Script to process mesher input based on analytic functions to create a Magritte model.
    :param config: configuration dict
    """
    # TODO: Maybe replace the save to numpy by vaex structures to reduce memory costs?

    if 'points' not in config:
        raise ValueError('"Points should be specified when assuming spherical symmetry".')
    else:
        points = config['points']

    neighbors = [[] for _ in range(len(points))]
    for i in range(len(points)-1):
        neighbors[i+1].append(i+0)
        neighbors[i+0].append(i+1)

    name = f"{config['project folder']}{config['model name']}"

    np.save(f"{name}_position",  points                      )
    np.save(f"{name}_boundary",  np.array([0, len(points)-1]))
    np.save(f"{name}_neighbors", np.array(neighbors)         )

    if 'functions' in config:
        np.save(f"{name}_velocity", config['functions']['velocity'   ](points))
        np.save(f"{name}_nl1",      config['functions']['nl1'        ](points))
        np.save(f"{name}_nH2",      config['functions']['nH2'        ](points))
        np.save(f"{name}_tmp",      config['functions']['temperature'](points))
        np.save(f"{name}_trb",      config['functions']['turbulence' ](points))
    elif 'data' in config:
        np.save(f"{name}_velocity", config['data']     ['velocity'   ])
        np.save(f"{name}_nl1",      config['data']     ['nl1'        ])
        np.save(f"{name}_nH2",      config['data']     ['nH2'        ])
        np.save(f"{name}_tmp",      config['data']     ['temperature'])
        np.save(f"{name}_trb",      config['data']     ['turbulence' ])

    return



def process_mesher_input(config):
    """
    Script to process mesher input based on another Magritte model to create a new Magritte model.
    :param config: configuration dict
    """

    # TODO: Maybe replace the save to numpy by vaex structures to reduce memory costs?

    if (os.path.splitext(config['input file'])[1].lower() != '.vtk'):
        raise ValueError('Only .vtk mesher files are currently supported.')

    print(' --- original model type:', config['original model type'])

    modelName = f"{config['project folder']}{config['original model name']}"
    # Choose the io interface corresponding to the model type
    if   (config['original model type'].lower() in ['hdf5', 'h5']):
        io = IoPython('hdf5', f"{modelName}.hdf5")
    elif (config['original model type'].lower() in ['text', 'txt', 'ascii']):
        io = IoText(f"{modelName}/")
    else:
         raise ValueError('No valid model type was given (hdf5, ascii).')

    model = Model()
    model.read(io)

    mesh = Mesh(config['input file'])

    tree = cKDTree(np.array(model.geometry.cells.position))
    corresp_points = tree.query(mesh.points)[1]

    name = f"{config['project folder']}{config['model name']}"

    np.save(f"{name}_position",  mesh.points                                                      )
    np.save(f"{name}_boundary",  mesh.boundary                                                    )
    np.save(f"{name}_neighbors", mesh.neighbors                                                   )
    np.save(f"{name}_velocity",  np.array(model.geometry.cells.velocity)          [corresp_points])
    np.save(f"{name}_nl1",       np.array(model.chemistry.species.abundance)[:,1] [corresp_points])
    np.save(f"{name}_nH2",       np.array(model.chemistry.species.abundance)[:,2] [corresp_points])
    np.save(f"{name}_tmp",       np.array(model.thermodynamics.temperature.gas)   [corresp_points])
    np.save(f"{name}_trb",       np.array(model.thermodynamics.turbulence.vturb2) [corresp_points])


    # Create a .vtk file containing the mesh
    meshio.write_points_cells(
        filename = f"{config['project folder']}{config['model name']}.vtk",
        points   = np.array(mesh.points),
        cells    = {'tetra' : mesh.tetras}                                 )

    return




def process_analytic_input(config):
    """
    Script to process mesher input based on analytic functions to create a Magritte model.
    :param config: configuration dict
    """
    # TODO: Maybe replace the save to numpy by vaex structures to reduce memory costs?
    if (os.path.splitext(config['input file'])[1].lower() != '.vtk'):
        raise ValueError('Only .vtk mesher files are currently supported.')

    mesh = Mesh(config['input file'])

    name = f"{config['project folder']}{config['model name']}"

    np.save(f"{name}_position",  mesh.points                              )
    np.save(f"{name}_boundary",  mesh.boundary                            )
    np.save(f"{name}_neighbors", mesh.neighbors                           )
    np.save(f"{name}_velocity",  config['functions']['velocity'   ](mesh.points))
    np.save(f"{name}_nl1",       config['functions']['nl1'        ](mesh.points))
    np.save(f"{name}_nH2",       config['functions']['nH2'        ](mesh.points))
    np.save(f"{name}_tmp",       config['functions']['temperature'](mesh.points))
    np.save(f"{name}_trb",       config['functions']['turbulence' ](mesh.points))

    return




def process_analytic_input_with_geometry(config):
    """
    Script to process mesher input based on analytic functions to create a Magritte model.
    :param config: configuration dict
    """
    # TODO: Maybe replace the save to numpy by vaex structures to reduce memory costs?

    name = f"{config['project folder']}{config['model name']}"

    np.save(f"{name}_position",  config['geometry' ]['position'   ])
    np.save(f"{name}_boundary",  config['geometry' ]['boundary'   ])
    np.save(f"{name}_neighbors", config['geometry' ]['neighbors'  ])
    np.save(f"{name}_velocity",  config['functions']['velocity'   ](config['geometry']['position']))
    np.save(f"{name}_nl1",       config['functions']['nl1'        ](config['geometry']['position']))
    np.save(f"{name}_nH2",       config['functions']['nH2'        ](config['geometry']['position']))
    np.save(f"{name}_tmp",       config['functions']['temperature'](config['geometry']['position']))
    np.save(f"{name}_trb",       config['functions']['turbulence' ](config['geometry']['position']))

    return


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

    print("Warning: we assume that the geometry to be a cube centred around the origin.")
    print("Warning: we assume that there is (at least) one face of the cube that has not been refined (or that is completely covered by the coarsest elements that can be found on the boundary.).")
    bound = 0.999*np.min(np.max(np.abs(centres), axis=0))
    # (x_max, y_max, z_max) = 0.99999*np.max(centres, axis=0)
    # (x_min, y_min, z_min) = 0.99999*np.min(centres, axis=0)
    x_max = y_max = z_max =  bound
    x_min = y_min = z_min = -bound

    print("Extracting boundary...")
    boundary = []
    for i, (x,y,z) in tqdm(enumerate(centres), file=sys.stdout):
        if not ((x_min < x < x_max) and
                (y_min < y < y_max) and
                (z_min < z < z_max)    ):
            boundary.append(i)
    boundary = np.array(boundary)

    print("Extracting neighbours...")
    # Assuming refinement causes a cubic cell to split in 8 subcells
    # and assuming max one level of refinement increase form one cell to the next,
    # max number of neighbours if all neighbours are refined, yields 6*4 + 12*2 + 8 = 56 neighbors.
    neighbors = cKDTree(centres).query(centres, 57)[1]
    # Closest point is point itself, execlude this
    neighbors = neighbors[:,1:]

    data = {'position'  : centres,
            'velocity'  : velocity,
            'boundary'  : boundary,
            'neighbors' : neighbors,
            'nH2'       : nH2,
            'nl1'       : nCO,
            'tmp'       : tmp,
            'trb'       : trb      }

    for (key,value) in data.items():
        name = f"{config['project folder']}{config['model name']}_{key}"
        np.save(name, value, allow_pickle=True)

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

    return


def process_phantom_input(config):

    print("Reading phantom input...")

    if (config['line producing species'][0] != 'CO'):
        raise NotImplementedError('amrvac input currently assumes that CO is the line producing species.')

    print("Warning this assumes gamma=1.2 and mu=2.381 as per Silke & Jolien's models.")
    gamma = 1.2
    mu    = 2.381
    velocity_constant = 2.9784608e+06 * 1.0e-2
    density_constant  = 5.9410314e-07
    energy_constant   = 8.8712277e+12


    print("Warning this assumes a constant H2 mass fraction of 1.6e-10 as per Frederik's guess.")
    print("Warning this assumes a constant CO mass fraction of 7.8e-12 as per Frederik's guess.")
    print("(Based on the average of a model of Ward.)")
    nH2 = 1.6e-10
    nCO = 7.8e-12


    if (os.path.splitext(config['input file'])[1].lower() != '.ascii'):
        raise ValueError('Only .ascii PHANTOM files are currently supported.')

    print("Reading phantom input...")

    (x,y,z, h, density, v_x,v_y,v_z, u) = np.loadtxt(config['input file'], skiprows=14, usecols=(0,1,2,4,5,6,7,8,9), unpack=True)

    x       = x       [h>0.0]
    y       = y       [h>0.0]
    z       = z       [h>0.0]
    density = density [h>0.0]
    v_x     = v_x     [h>0.0]
    v_y     = v_y     [h>0.0]
    v_z     = v_z     [h>0.0]
    u       = u       [h>0.0]

    density[density < 1.0e-50] = 1.0e-50

    ncells = len(x)

    position = np.array((x,  y,  z  )).transpose()
    position = position * constants.au.si.value
    velocity = np.array((v_x,v_y,v_z)).transpose()
    velocity = velocity * (1.0e-2 * velocity_constant / constants.c.si.value)

    delaunay = Delaunay(position)


    # Create a .vtk file containing the mesh
    meshio.write_points_cells(
        filename = f"{config['project folder']}{config['model name']}.vtk",
        points   = position,
        cells    = {'tetra' : delaunay.simplices} )


    # Extract Delaunay vertices (= Voronoi neighbors)
    (indptr, indices) = delaunay.vertex_neighbor_vertices
    neighbors = [indices[indptr[k]:indptr[k+1]] for k in range(ncells)]

    # Compute the indices of the boundary particles
    boundary = set([])
    for i in range(delaunay.neighbors.shape[0]):
        m1  = (delaunay.neighbors[i] == -1)
        nm1 = np.sum(m1)
        if   (nm1 == 0):
            pass
        elif (nm1 == 1):
            for b in delaunay.simplices[i][m1]:
                boundary.add(b)
        elif (nm1 >= 2):
            for b in delaunay.simplices[i]:
                boundary.add(b)
    boundary = list(boundary)

    bdy = np.array(boundary)

    b_nms = np.linalg.norm(position[bdy], axis=1)
    p_nms = np.linalg.norm(position,      axis=1)

    boundary = [i[0] for i in np.argwhere(p_nms >= np.min(b_nms))]

    # Convert to number densities [#/m^3]
    nH2 = density * density_constant * nH2 * 1.0e+6 * constants.N_A.si.value /  2.02
    nCO = density * density_constant * nCO * 1.0e+6 * constants.N_A.si.value / 28.01

    tmp = mu * (gamma-1.0) * u * energy_constant * 1.00784 * (units.erg/units.g * constants.u/constants.k_B).to(units.K).value
    tmp[tmp < 2.725] = 2.725

    trb = (150.0/constants.c.si.value)**2 * np.ones(ncells)

    data = {'position'  : position,
            'velocity'  : velocity,
            'boundary'  : boundary,
            'neighbors' : neighbors,
            'nH2'       : nH2,
            'nl1'       : nCO,
            'tmp'       : tmp,
            'trb'       : trb      }

    for (key,value) in data.items():
        name = f"{config['project folder']}{config['model name']}_{key}"
        np.save(name, value, allow_pickle=True)

    return
