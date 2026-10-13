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
    """
    A class to enhance VTKMesh functionality with additional features and utilities.

    The constructor takes a mesh in any of three forms and normalizes it to a
    distributed VTKMesh before converting it into a pyvista UnstructuredGrid:
    a file path is read with pynekread into a Mesh and wrapped, a Mesh object is
    wrapped directly, and an existing VTKMesh is used as is. In all cases the
    VTKMesh is created (or assumed to be) with global_connectivity=False and
    distributed_axis=0.

    Once self.mesh holds the VTKMesh, its connectivity is repacked into the VTK
    legacy format expected by pyvista: for each cell, the number of vertices
    is written first, followed by the vertex indices of that cell taken from
    the flat Connectivity array between Offsets[i] and Offsets[i+1]. The
    resulting flat index array, together with the per-cell Types (VTK cell
    types) and the Points vertex coordinates, is passed to
    pv.UnstructuredGrid to build self.grid.

    Attributes:
        comm: The communication object.
        mesh: The VTKMesh object created (or accepted) from the input, which
            can be a file path, Mesh object, or VTKMesh object.
        field_names: A list to store field names.
        log: Logger object for logging messages.
        grid: The UnstructuredGrid object created from the mesh data.
    """
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