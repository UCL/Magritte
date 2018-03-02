from paraview.simple import *
r = LegacyVTKReader( FileNames=['input/grid_log3.vtk'] )
w = XMLUnstructuredGridWriter()
w.FileName = 'input/grid_log3.vtu'
w.UpdatePipeline()
