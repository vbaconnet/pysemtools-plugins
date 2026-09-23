import sys
# Remove VisIt's conflicting VTK paths
sys.path = [p for p in sys.path if 'visit' not in p.lower()]

try:
    import pyvista as pv
except ModuleNotFoundError as e:
    print(e)
    raise ModuleNotFoundError("Try pip install pyvista")

from pysemtools.io.ppymech.neksuite import pynekread
from pysemtools.datatypes.msh import Mesh
from pysemtools.datatypes.msh_vtk import VTKMesh
import numpy as np
from pysemtools.monitoring.logger import Logger

class EnhancedVTKMesh:
    def __init__(self, comm, mesh: str | Mesh | VTKMesh):
        self.comm = comm
        self.mesh: VTKMesh
        self.field_names: list[str] = []
        self.log = Logger(comm=comm, module_name="EnhancedVTKMesh")

        if isinstance(mesh, str):
            self.log.write("info", "Initializing E-VTKMesh from file")

            # Read from fld file
            m = Mesh(comm)
            pynekread(mesh, comm, msh=m)

            self.mesh = VTKMesh(
                comm,
                x = m.x,
                y = m.y,
                z = m.z,
                global_connectivity=False,
                distributed_axis=0)

        elif isinstance(mesh, Mesh):
            self.log.write("info", "Initializing E-VTKMesh from Mesh() object")

            # Create VTKMesh from Mesh
            self.mesh = VTKMesh(
                comm,
                x = mesh.x,
                y = mesh.y,
                z = mesh.z,
                global_connectivity=False,
                distributed_axis=0)
            
        elif isinstance(mesh, VTKMesh):
            self.log.write("info", "Initializing E-VTKMesh from VTKMesh() object")

            # Just assign the VTKMesh
            self.mesh = mesh
        else:
            raise TypeError("Mesh can only be str, Mesh or VTKMesh")


        # Fix the mesh so it can be read into pyvista and paraview
        # (pure AI)
        # Convert VTKMesh data to VTK legacy format for pyvista
        self.log.write("info", "Fixing cells and offsets...")
        n_cells = self.mesh.Types.size
        cells_legacy = []

        for i in range(n_cells):
            n_verts = self.mesh.Sizes[i]
            start = self.mesh.Offsets[i]
            end = self.mesh.Offsets[i + 1]
            cells_legacy.append(n_verts)
            cells_legacy.extend(self.mesh.Connectivity[start:end].tolist())

        # Create pyvista UnstructuredGrid
        cell_array = np.array(cells_legacy, dtype=np.int64)
        self.log.write("info", "Creating UnstructuredGrid object")
        self.grid = pv.UnstructuredGrid(cell_array, self.mesh.Types, self.mesh.Points)

    def point_data(self, name, value, auto_ravel = True):
        if auto_ravel:
            self.grid.point_data[name] = value.ravel('C')
        else:
            self.grid.point_data[name] = value