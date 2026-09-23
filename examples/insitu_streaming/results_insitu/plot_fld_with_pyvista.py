import sys
# Remove VisIt's conflicting VTK paths
sys.path = [p for p in sys.path if 'visit' not in p.lower()]
import pyvista as pv
import numpy as np

# Import MPI
from mpi4py import MPI

comm = MPI.COMM_WORLD

from pysemtools.io.ppymech.neksuite import pynekread
from pysemtools.datatypes.field import FieldRegistry
from pysemtools.datatypes.msh import Mesh
from pysemtools.datatypes.msh_vtk import VTKMesh
from pysemtools_plugins.EnhancedVTKMesh import EnhancedVTKMesh

# Create mesh and field objects
m = Mesh(comm)
f = FieldRegistry(comm)

f.fields["u"].set

# Read mesh from field0.f00000 (contains coordinates)
pynekread("field0.f00000", comm, msh=m)

# Read field data from field0.f00001 (contains field values)
pynekread("field0.f00001", comm, fld=f)

# Use VTKMesh to properly handle SEM mesh structure
vtk_mesh = EnhancedVTKMesh(comm, m)

# Add field data
for field_name, field_components in f.fields.items():
    for i, component_data in enumerate(field_components):
        vtk_mesh.point_data(f'{field_name}_{i}', component_data)

# Take 2D slice at z=0
slice = vtk_mesh.grid.slice(normal=[0, 0, 1], origin=[0.0, 0.0, 0.0])

# Generate contour lines from the slice
contours = slice.contour(isosurfaces=20)

# Plot with pyvista
plotter = pv.Plotter(off_screen=True)
plotter.add_mesh(slice, cmap='viridis', scalar_bar_args={'title': 'vel_0'})
plotter.view_xy()
#plotter.add_mesh(contours, color='white', line_width=2)
if comm.Get_rank() == 0:
    plotter.screenshot('slice_z0_pyvista_contour.png', scale=2)
