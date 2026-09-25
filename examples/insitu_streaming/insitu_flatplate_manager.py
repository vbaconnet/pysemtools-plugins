#========================================
# Import and set up general modules
#========================================
import os
import argparse

from pysemtools_plugins import FlatPlateStreamProcessor

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# Import MPI
from mpi4py import MPI #equivalent to the use of MPI_init() in C

# Split communicator for MPI - MPMD
worldcomm = MPI.COMM_WORLD
worldrank = worldcomm.Get_rank()
worldsize = worldcomm.Get_size()
col = 1
comm = worldcomm.Split(col,worldrank)
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:
    print(f"Python is running with {size} ranks")

#========================================
# Import modules
#========================================
# general functionality
import numpy as np
from os.path import join
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.tri import Triangulation

# external
# import adios2.bindings as adios2

# my classes
# from pysemtools_plugins.EnhancedStreamer import EnhancedStreamer
from pysemtools_plugins.FlatPlateStreamProcessor import FlatPlateStreamProcessor
#from pysemtools_plugins.EnhancedVTKMesh import EnhancedVTKMesh
# from pysemtools.io.catalyst import CatalystSession

# pysemtools
# from pysemtools.io.adios2.stream import DataStreamer
# from pysemtools.io.utils import get_fld_from_ndarray
# from pysemtools.datatypes.msh import Mesh
# from pysemtools.interpolation.probes import Probes
from pysemtools.monitoring.logger import Logger
log = Logger(comm=comm, module_name="cylinder_insitu_task")

import json

#=========================================
# Define some helper functions
#=========================================

def get_output_directory(fname):
    with open(fname, 'r') as f:
        data = json.load(f)

    return data["case"]["output_directory"]

def get_field_names(fname):
    with open(fname, 'r') as f:
        data = json.load(f)

    for component in data["case"]["simulation_components"]:
        if component.get("type") == "data_streamer":
            return component["fields"]
    raise ValueError("No data_streamer component found")

def init_plot(save_output_path):
    fig, axs = plt.subplots(nrows = 2, figsize = (15,12), sharex = False)

    axs[0].set_xlabel("u")
    axs[0].set_ylabel("y")
    axs[0].set_aspect('auto')
    axs[1].set_xlabel("x [m]")
    axs[1].set_ylabel("tau_x")

    fig.suptitle("Waiting to receive data from neko...")
    fname = join(save_output_path, "plate_insitu_00000.png")
    fig.savefig(fname, dpi = 200)

    return fig, axs

#=========================================
# Parse arguments
#=========================================

parser = argparse.ArgumentParser(description="In-situ visualization task for the cylinder case.")
parser.add_argument("--dry-run", action="store_true",
                    help="Import modules and exit without executing any actions.")
parser.add_argument("--timeout", type=int, default=300,
                    help="Timeout in seconds for the ADIOS2 SST stream open (default: 300).")
args = parser.parse_args()

if args.dry_run:
    log.write("info", "Dry run: all imports succeeded. Exiting.")
    sys.exit(0)

# Remove globalArray_* files
import subprocess
subprocess.run("rm -f globalArray*", shell=True)

#=========================================
# Define some variables/parameters
#=========================================

log.write("info", "Starting insitu task")

dtype_string = "double"
backend = "numpy"
if dtype_string == "single":
    dtype = np.float32
else:
    dtype = np.float64

output_path = get_output_directory("cylinder_insitu.case")
log.write("info", f"Outputting insitu snapshots to folder {output_path}")

#=========================================
# Initialize the streamer
#=========================================

streamer_field_names = get_field_names("cylinder_insitu.case")

processor = FlatPlateStreamProcessor(
    comm,
    fields = streamer_field_names,
    adios2_timeout = args.timeout,
    create_catalyst_session = True,
    catalyst_pipeline="pipeline_new.py",
    catalyst_channel="field.nek5000"
    )

processor.read_plate_coordinates("/scratch/baconnet/simulations/neko/hpc_workflows/flatplate/meshes/geometry/unique_wall_coords_2D.csv")

#=========================================
# Stream the mesh coordinates to build interpolators
#=========================================

processor.receive_mesh(dtype_string)

#=========================================
# Initialize the structured probes data
#=========================================

processor.generate_points_on_plate(
    xmin = 0.02,
    xmax = 0.5,
    Nx = 500,
    distribution = "uniform",
    interpolator_name = "plate",
    output_fname = f'{output_path}/plate.hdf5'
)

processor.generate_BL_plane(
    xmin = 0.02,
    xmax = 0.5,
    Nx = 4,
    Ly = 0.01,
    Ny = 100,
    x_distribution="uniform",
    interpolator_name = "BLs",
    write_coords = False,
    output_fname = f"{output_path}/BLs.hdf5"
)
X = processor.interpolators["BLs"].x
Y = processor.interpolators["BLs"].y

# --------------------------------------------
# Read from csv
processor.add_interpolator_from_values(
    name = "probes",
    fname = "probes.csv",
    x = 0.0,
    y = 0.0,
    z = 0.0,
    write_coords = True,
    output_fname = f"{output_path}/probes.hdf5"
)

#=========================================
# Initialize plots
#=========================================

if comm.Get_rank() == 0:
    fig, axs = init_plot(output_path)
    axs[0].plot(X, Y, "r-")
    axs[0].plot(X.T, Y.T, "k-")
    fname = join(output_path, f"plate_insitu_00000.png")
    fig.savefig(fname, dpi = 200)

    axs[0].clear()
    
    # axs[0].plot(
    #     processor.plate_coordinates["x"],
    #     processor.plate_coordinates["y"],
    #     "k-"
    #     )

#=========================================
# Start streaming data
#=========================================

j = 0
while True:

    processor.receive_fields(dtype_string)

    # Check if data was recieved or if the stream ended
    if not processor.get_adios2_status():
        break

    processor.update_interpolators(j, write_data=False)

    processor.execute_catalyst_session(j, j*0.1)

    if comm.Get_rank() == 0:

        u_BLs = processor.get_field_from_interpolator(
            streamer_field_names[0], "BLs")
        
        tau_plate = processor.get_field_from_interpolator(
            streamer_field_names[1], "plate")

        u_pbs = processor.get_field_from_interpolator(
            streamer_field_names[0], "probes"
        )

        fig.suptitle(f"time step {j}")
        log.write("info", "Plotting fields")

        axs[0].clear()
        for bl_i in range(u_BLs.shape[0]):
            axs[0].plot(
                u_BLs[bl_i,:],
                processor.interpolators["BLs"].y[bl_i,:]
            )

        # if j == 0:
        #     line0, = axs[0].plot([], [], "r-", label="probe 0")
        #     line1, = axs[0].plot([], [], "g-", label="probe 1")
        #     line2, = axs[0].plot([], [], "b-", label="probe 2")
        #     axs[0].legend()
        # else:
        #     # Append to each line's data
        #     x = np.append(line0.get_xdata(), j)
        #     line0.set_data(x, np.append(line0.get_ydata(), u_pbs[0]))
        #     line1.set_data(x, np.append(line1.get_ydata(), u_pbs[1]))
        #     line2.set_data(x, np.append(line2.get_ydata(), u_pbs[2]))

        #     # Auto-rescale the view
        #     axs[0].relim()
        #     axs[0].autoscale_view()

        axs[1].clear()
        axs[1].plot(
            processor.interpolators["plate"].x, 
            tau_plate
            )
        
        if j == 0: 
            fig.tight_layout()

        j = 0
        fname = join(output_path, f"plate_insitu_{j:05d}.png")
        subprocess.run(["ln", "-fs", fname, "current.png"])
        #fname = join(output_path, "cylinder_insitu_00000.png")
        log.write("info", f"Saving snapshot to {fname}")
        fig.savefig(fname, dpi = 200)

    j += 1

log.write("info", "Detected stream ended")

processor.finalize()
