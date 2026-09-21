#========================================
# Import and set up general modules
#========================================
import sys
import os
import argparse
from time import sleep

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

# external
import adios2.bindings as adios2

# my classes
from pysemtools_plugins.EnhancedStreamer import EnhancedStreamer

# pysemtools
from pysemtools.io.adios2.stream import DataStreamer
from pysemtools.io.utils import get_fld_from_ndarray
from pysemtools.datatypes.msh import Mesh
from pysemtools.interpolation.probes import Probes
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
    fig, axs = plt.subplots(nrows = 3, figsize = (10,12), sharex = False)

    axs[0].set_xlabel("x")
    axs[0].set_ylabel("y")
    axs[0].set_aspect('equal')
    axs[1].set_xlabel("x")
    axs[1].set_ylabel("y")
    axs[1].set_aspect('equal')
    axs[-1].set_xlabel("angle [deg]")
    axs[-1].set_ylabel("tau_x")
    
    # Set colorbar for x-velocity
    norm = colors.Normalize(vmin=-0.4, vmax=1.4)
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='viridis'),
                         ax = axs[0])
    cbar.set_label("x-velocity")
    
    # Set colorbar for z-vorticity
    norm = colors.Normalize(vmin=-4.0, vmax=4.0)
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='bwr'),
                        ax = axs[1])
    cbar.set_label("z-vorticity")

    fig.suptitle("Waiting to receive data from neko...")
    fname = join(save_output_path, "cylinder_insitu_00000.png")
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
subprocess.run(["rm", "-f", "globalArray*"])
sleep(1.0)

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

processor = EnhancedStreamer(
    comm,
    fields = streamer_field_names,
    adios2_timeout = args.timeout
    )

#=========================================
# Stream the mesh coordinates to build interpolators
#=========================================

processor.receive_mesh(dtype_string)

#=========================================
# Initialize the structured probes data
#=========================================

# Generate a 30x30 grid in the x-y plane at a given z.
N = 30
xbounds = [0.6, 4.0]
ybounds = [-1.0, 1.0]
z = 2.0
x = np.linspace(xbounds[0], xbounds[1], N)
y = np.linspace(ybounds[0], ybounds[1], N)
X, Y = np.meshgrid(x,y)
del x,y

processor.add_interpolator_from_values(
    name = "plane",
    x = X,
    y = Y,
    fill_extrude_value = 2.0,
    write_coords = False
    )

# Generate a line with 100 points along the circle of center (0,0) and radius
# R = 0.5
theta = np.linspace(0.0, np.pi, 100)
processor.add_interpolator_from_values(
    name = "circle",
    x = np.cos(theta + np.pi) * 0.5,
    y = np.sin(theta + np.pi) * 0.5,
    fill_extrude_value = 2.0,
    write_coords = False
    )

#=========================================
# Initialize plots
#=========================================

if comm.Get_rank() == 0:
    fig, axs = init_plot(output_path)

#=========================================
# Start streaming data
#=========================================

j = 0
while True:

    processor.receive_fields(dtype_string)

    # Check if data was recieved or if the stream ended
    if not processor.get_adios2_status():
        break

    # Interpolate and reshape into a structured grid for plotting
    processor.update_interpolator(
        j, 
        "plane", 
        streamer_field_names[:2]
        )
    
    processor.update_interpolator(
        j, 
        "circle", 
        streamer_field_names[-1:])

    if comm.Get_rank() == 0:

        u_probe = processor.get_field_from_interpolator(
            streamer_field_names[0], "plane")
        curl_probe = processor.get_field_from_interpolator(
            streamer_field_names[1], "plane")
        
        pressure_probe = processor.get_field_from_interpolator(
            streamer_field_names[2], "circle")

        fig.suptitle(f"time step {j}")
        log.write("info", "Plotting fields")
        #axs[0].cla()
        axs[0].contourf(processor.interpolators["plane"].x, processor.interpolators["plane"].y, u_probe, levels = 40, norm=colors.Normalize(vmin=-0.4, vmax=1.4))
        #axs[1].cla()
        axs[1].contourf(processor.interpolators["plane"].x, processor.interpolators["plane"].y, curl_probe, levels = 40, cmap = "bwr", norm=colors.Normalize(vmin=-4.0, vmax=4.0))
        #axs[2].cla()
        axs[2].plot(theta * 180 / np.pi, pressure_probe)
        if j == 0: 
            fig.tight_layout()

        fname = join(output_path, f"cylinder_insitu_{j:05d}.png")
        log.write("info", f"Saving snapshot to {fname}")
        fig.savefig(fname, dpi = 200)

    j += 1

log.write("info", "Detected stream ended")

processor.finalize()
