import adios2.bindings as adios2
from pysemtools.io.adios2.stream import DataStreamer
from pysemtools.io.utils import get_fld_from_ndarray
from pysemtools.datatypes.msh import Mesh
from pysemtools.datatypes.field import NoOverwriteDict
from pysemtools.monitoring.logger import Logger
from EnhancedInterpolator import EnhancedInterpolator
import numpy as np

class EnhancedStreamer:

    def __init__(self, comm, fields: list[str], adios2_timeout: int = 300):
        """
        Initialize the EnhancedStreamer.

        :param comm: MPI communicator
        :param fields: List of field names to process
        :param adios2_timeout: Timeout in seconds for ADIOS2 operations (default: 300)
        """

        self.comm = comm
        self.log = Logger(comm=comm, module_name="EnhancedStreamer")

        self.adios2_timeout = adios2_timeout

        self.msh = None
        self.field_names = fields
        self.fields = NoOverwriteDict()

        self.log.write("info", f"Requested fields: {fields}")

        self.log.write("info", "Initializing streamer")
        self.ds = DataStreamer(comm, timeout_seconds = adios2_timeout)

        self.interpolators = NoOverwriteDict()

        self.adios2_status = None

    def check_attr(self, name):
        
        if not hasattr(self, name):
            raise AttributeError(f"No {name} attribute present. Please initialize me.")
    
    def finalize(self):

        self.log.write("info", "Finalizing streamer")
        self.ds.finalize()
        self.log.write("info", "Done!")

    def get_adios2_status(self):

        if self.ds.step_status == adios2.StepStatus.OK:
            stream_data = True
        elif self.ds.step_status == adios2.StepStatus.EndOfStream:
            stream_data = False

        return stream_data

    def receive_mesh(self, dtype: str = "double"):
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
            if f in self.fields:
                self.fields[f][...] = ftemp
            else:
                self.fields[f] = ftemp
                
            self.log.write("info", f"Received field {f}.")
            
        self.log.write("info", "Data received")

    def add_interpolator_from_object(self, i: EnhancedInterpolator,
                                     name: str = None):
        """
        Add an interpolator object to the StreamProcessor.

        :param i: EnhancedInterpolator object to add
        :param name: Name of the interpolator. If None, a default name is generated.
        """
        if name is None:
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

    def update_interpolators(self, t: float = 0.0):
        """
        Update interpolators when new data has been received
        """

        for iname, i in self.interpolators.items():
            self.log.write("info", f"Updating interpolator {iname}...")
            i.interpolate_from_field_list(
                t,
                field_list = list(self.fields.values()),
                field_names = self.field_names,
                comm = self.comm,
                write_data = False
                )

    def get_field_from_interpolator(self, field_name, interpolator_name):
        return self.interpolators[interpolator_name].get_field(field_name)

