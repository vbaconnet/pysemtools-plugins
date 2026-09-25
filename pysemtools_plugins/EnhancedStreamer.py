from venv import create

import adios2.bindings as adios2
from pysemtools.io.adios2.stream import DataStreamer
from pysemtools.io.utils import get_fld_from_ndarray
from pysemtools.datatypes.msh import Mesh
from pysemtools.datatypes.field import NoOverwriteDict
from pysemtools.monitoring.logger import Logger
from pysemtools.io.wrappers import write_data

try:
    from pysemtools_plugins.EnhancedVTKMesh import EnhancedVTKMesh
    _HAS_EVTKMESH=True
except ModuleNotFoundError:
    print("pyvista is not installed. EnhancedVTKMesh capabilities disabled.")
    _HAS_EVTKMESH=False

from pysemtools_plugins.EnhancedInterpolator import EnhancedInterpolator
import numpy as np
from pysemtools.io.catalyst import CatalystSession

class EnhancedStreamer:

    def __init__(
            self, 
            comm, 
            fields: list[str], 
            adios2_timeout: int = 300,
            create_catalyst_session: bool = False,
            catalyst_pipeline: str | None = None,
            catalyst_channel: str | None = None):
        """
        Initialize the EnhancedStreamer.

        :param comm: MPI communicator
        :param fields: List of field names to process
        :param adios2_timeout: Timeout in seconds for ADIOS2 operations (default: 300)
        """

        self.comm = comm
        self.log = Logger(comm=comm, module_name="EnhancedStreamer")

        self.adios2_timeout = adios2_timeout

        self.msh: Mesh | None = None
        self.field_names = fields
        self.fields = NoOverwriteDict()

        self.log.write("info", f"Requested fields: {fields}")

        self.log.write("info", "Initializing streamer")
        self.ds = DataStreamer(comm, timeout_seconds = adios2_timeout)

        self.interpolators = NoOverwriteDict()

        self.adios2_status = None

        self.vtk_mesh: EnhancedVTKMesh | None = None
        self.catalyst_session = None

        if create_catalyst_session:
            if not catalyst_channel and not catalyst_pipeline:
                raise ValueError("channel and pipeline must be provided if Catalyst is enabled.")

            self.catalyst_session = CatalystSession(
                self.comm, 
                catalyst_pipeline,
                catalyst_channel)

    def check_attr(self, name):
        
        if not hasattr(self, name):
            raise AttributeError(f"No {name} attribute present. Please initialize me.")
    
    def finalize(self):

        self.log.write("info", "Finalizing streamer")
        self.ds.finalize()
        self.log.write("info", "Done!")

        if isinstance(self.catalyst_session, CatalystSession):
            self.catalyst_session.finalize()

    def get_adios2_status(self):

        stream_data = False
        if self.ds.step_status == adios2.StepStatus.OK:
            stream_data = True
        elif self.ds.step_status == adios2.StepStatus.EndOfStream:
            stream_data = False

        return stream_data

    def receive_mesh(
            self, 
            dtype: str = "double", 
            create_vtkmesh: bool = False):
        """
        Receive mesh data from the streamer and initialize the mesh.

        :param dtype: Data type for the mesh coordinates ("single" for float32, "double" for float64)
        :type dtype: str
        """

        if dtype == "single":
            dt = np.float32
        else:
            dt = np.float64

        self.log.write("info", "Receiving mesh...")
        x = get_fld_from_ndarray(
            self.ds.recieve(),
            self.ds.lx, self.ds.ly, self.ds.lz, self.ds.nelv
            ).astype(dt)

        y = get_fld_from_ndarray(
            self.ds.recieve(),
            self.ds.lx, self.ds.ly, self.ds.lz, self.ds.nelv
            ).astype(dt)

        z = get_fld_from_ndarray(
            self.ds.recieve(),
            self.ds.lx, self.ds.ly, self.ds.lz, self.ds.nelv
            ).astype(dt)

        self.log.write("info", "Data received")

        self.log.write("info", "Initializing mesh...")
        self.msh = Mesh(self.comm, x = x, y = y, z = z)
        self.log.write("info", "Initializing mesh... done")

        if create_vtkmesh and _HAS_EVTKMESH:
            self.vtk_mesh = EnhancedVTKMesh(self.comm, self.msh)

        if isinstance(self.catalyst_session, CatalystSession):
            self.catalyst_session.set_mesh(
                self.msh.x,
                self.msh.y,
                self.msh.z)

    def receive_fields(self, dtype = "double"):
        if dtype == "single":
            dt = np.float32
        else:
            dt = np.float64
        self.log.write("info", "Waiting for data from Neko...")
        # Get the data, assuming that the order in which the files are 
        # specified in the init are those received from Neko
        for f in self.field_names:

            ftemp = get_fld_from_ndarray(
                self.ds.recieve(),
                self.ds.lx, self.ds.ly, self.ds.lz, self.ds.nelv
                ).astype(dt) 

            self.add_field(f, ftemp)
                
            self.log.write("info", f"Received field {f}.")
            
        self.log.write("info", "Data received")

        if isinstance(self.vtk_mesh, EnhancedVTKMesh):
            self.log.write("info", "Updating fields in VTKMesh")
            for f,v in self.fields.items():
                self.vtk_mesh.point_data(f, v)

        if isinstance(self.catalyst_session, CatalystSession):
            self.log.write("info", "Updating fields in CatalystSession")
            self.catalyst_session.set_field(self.fields)

    def add_field(self, name, value):
        """
        Add a field to the StreamProcessor or replace an existing one.

        :param name: Name of the field to add or replace.
        :param value: Value of the field to add or replace.
        """
        if name in self.fields:
            self.log.write("info", f"Replacing field {name}")
            self.fields[name][...] = value
        else:
            self.log.write("info", f"Adding new field {name}")
            self.fields[name] = value

    def dump_fields(self, fname: str):
        """
        Dump the fields to a file.

        :param fname: Name of the file to write the fields to.
        """
        if self.msh:
            write_data(self.comm, fname, self.fields, parallel_io=True,
                    msh = [self.msh.x, self.msh.y, self.msh.z], write_mesh=True)

    def add_interpolator_from_object(self, i: EnhancedInterpolator,
                                     name: str = ""):
        """
        Add an interpolator object to the StreamProcessor.

        :param i: EnhancedInterpolator object to add
        :param name: Name of the interpolator. If None, a default name is generated.
        """
        if not name:
            name_ = f"interpolator_{str(len(self.interpolators.keys()))}"
        else:
            name_ = name

        self.log.write("info", f"Adding interpolator {name_} to the StreamProcessor")
        self.interpolators[name_] = i

    def add_interpolator_from_values(self, name, **kwargs):
        """
        Add an interpolator to the StreamProcessor using provided values.

        :param name: Name of the interpolator.
        :param kwargs: Keyword arguments to initialize the EnhancedInterpolator
        """

        self.log.write("info", f"Adding interpolator {name} to the StreamProcessor")

        # Make sure the user doesn't accidentally pass msh and comm
        # as arguments
        try:
            kwargs.pop('msh')
            self.log.write("info", "Popped msh out of kwargs")
        except KeyError:
            pass
        except Exception as e:
            print(e)
            raise ValueError(f"Error: {e}")

        # Make sure the user doesn't accidentally pass msh and comm
        # as arguments
        try:
            kwargs.pop('comm')
            self.log.write("info", "Popped comm out of kwargs")
        except KeyError:
            pass
        except Exception as e:
            print(e)
            raise ValueError(f"Error: {e}")

        self.interpolators[name] = EnhancedInterpolator(
            msh = self.msh,
            comm = self.comm,
            **kwargs)

    def update_interpolator(
            self, 
            t: float, 
            interpolator_name: str, 
            field_names: list[str] = [],
            **kwargs):
        """
        Update a specific interpolator with new data.

        :param interpolator_name: Name of the interpolator to update
        :param field_names: List of field names to use for interpolation
        :param t: Time value for the interpolation
        """
        assert isinstance(field_names, list)

        self.log.write("info", f"Updating interpolator {interpolator_name}")

        if not field_names:
            self.log.write("info", "Updating all fields")

            self.interpolators[interpolator_name].interpolate_from_field_list(
                    t,
                    field_list = list(self.fields.values()),
                    field_names = list(self.fields.keys()),
                    comm = self.comm,
                    **kwargs
                    )
        else:

            if not isinstance(field_names, list):
                raise ValueError("field_names must be a list!")
            
            self.log.write("info", f"Updating fields {field_names}")

            self.interpolators[interpolator_name].interpolate_from_field_list(
                    t,
                    field_list = [self.fields[f] for f in field_names],
                    field_names = field_names,
                    comm = self.comm,
                    **kwargs
                    )

    def update_interpolators(self, t: float = 0.0, **kwargs):
        """
        Update interpolators when new data has been received
        """
        for iname in self.interpolators.keys():
            self.update_interpolator(t, iname, **kwargs)

    def get_field_from_interpolator(self, field_name, interpolator_name):
        return self.interpolators[interpolator_name].get_field(field_name)

    def add_2d_grid_interpolation(
            self,
            name: str,
            x_bounds: list[float] | float,
            y_bounds: list[float] | float,
            z_bounds: list[float] | float,
            Nx: int = 100,
            Ny: int = 100,
            Nz: int = 100,
            **kwargs
            ):
        """
        Add an 2D rectilinear grid interpolator. One of x,y,z must be a float.
        Parameters
        ----------
        name : str
            Name of the interpolator.
        x_bounds : list[float] | float
            Bounds for the x-axis. If a single float is provided, is used as the "filler" value.
        y_bounds : list[float] | float
            Bounds for the y-axis. If a single float is provided, is used as the "filler" value.
        z_bounds : list[float] | float
            Bounds for the z-axis. If a single float is provided, is used as the "filler" value.
        Nx : int, optional
            Number of points along the x-axis (default is 100).
        Ny : int, optional
            Number of points along the y-axis (default is 100).
        Nz : int, optional
            Number of points along the z-axis (default is 100).
        """

        if isinstance(x_bounds, float):
            y = np.linspace(y_bounds[0], y_bounds[1], Ny)
            z = np.linspace(z_bounds[0], z_bounds[1], Nz)
            y, z = np.meshgrid(y, z)
            x = x_bounds
        elif isinstance(y_bounds, float):
            x = np.linspace(x_bounds[0], x_bounds[1], Nx)
            z = np.linspace(z_bounds[0], z_bounds[1], Nz)
            x, z = np.meshgrid(x, z)
            y = y_bounds
        elif isinstance(z_bounds, float):
            x = np.linspace(x_bounds[0], x_bounds[1], Nx)
            y = np.linspace(y_bounds[0], y_bounds[1], Ny) 
            x, y = np.meshgrid(x, y)
            z = z_bounds
        else:
            raise ValueError("Invalid bounds provided for interpolation")

        self.add_interpolator_from_values(
            name = name,
            x = x,
            y = y,
            z = z,
            **kwargs
        )

    def execute_catalyst_session(self, tstep: int = 0, t: float = 0.0):
        """
        Execute the Catalyst session with the given timestep and time.

        Parameters:
        ----------
        tstep : int
            The current timestep.
        t : float
            The current time.

        Raises:
        ------
        ValueError
            If the Catalyst session is not initialized.
        """

        if isinstance(self.catalyst_session, CatalystSession):
            self.catalyst_session.execute(tstep, t)
        else:
            raise ValueError("Catalyst session not initialized!")