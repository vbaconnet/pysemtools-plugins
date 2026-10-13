import numpy as np
from mpi4py import MPI
from pysemtools.io.ppymech.neksuite import pynekread
from pysemtools.datatypes.msh import Mesh
from pysemtools.datatypes.field import Field

from pysemtools_plugins.SurfaceDescriptor import SurfaceDescriptor
from pysemtools_plugins.SurfaceInterpolator import SurfaceInterpolator

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

PLATE = __file__.replace("test_surface_interpolator.py", "unique_wall_coords_2D.csv")
FLD = __file__.replace("test_surface_interpolator.py", "field0.f00000")

# --- Read mesh and fields from the nek field file -------------------------
msh = Mesh(comm)
fld = Field(comm)
pynekread(FLD, comm, msh=msh, data_dtype=np.single)
pynekread(FLD, comm, fld=fld, data_dtype=np.single)

if rank == 0:
    print("mesh nel:", msh.nelv, "gdim:", msh.gdim)
    print("mesh x range:", msh.x.min(), msh.x.max())
    print("mesh y range:", msh.y.min(), msh.y.max())
    print("mesh z range:", msh.z.min(), msh.z.max())
    print("vel fields:", len(fld.fields["vel"]), "pres fields:", len(fld.fields["pres"]))
    print("u shape:", fld.fields["vel"][0].shape)

# --- Surface descriptor from the plate coordinates ------------------------
data = np.genfromtxt(PLATE, delimiter=",", skip_header=1)
surface = SurfaceDescriptor(data[:,0], data[:,1])
surface.generate_interpolators()

# --- User-facing input: only s (curvilinear) and z ------------------------
N, Nz = 100, 10
smin, smax = surface.interpolate_s([0.02, 0.6])
s = np.linspace(smin, smax, N)

# Exact z bounds of the mesh: the end use is z-direction averaging, so the
# extrusion must span the full span.
zmin = comm.allreduce(msh.z.min(), op=MPI.MIN)
zmax = comm.allreduce(msh.z.max(), op=MPI.MAX)
z = np.linspace(zmin, zmax, Nz)

si = SurfaceInterpolator(surface, s, z, comm=comm, msh=msh, write_coords=False)

# --- Check the internal mapping -------------------------------------------
assert si.x.shape == (N, Nz) and si.y.shape == (N, Nz) and si.z.shape == (N, Nz)
assert np.allclose(si.x[:,0], surface.interpolate_x(s)), "x mapping broken"
assert np.allclose(si.y[:,0], surface.interpolate_y(s)), "y mapping broken"
assert np.allclose(si.z[0,:], z), "z mapping broken"
assert np.allclose(si.z_line, z), "z line broken"

if rank == 0:
    print("coordinate mapping OK, probe grid:", si.x.shape)
    print("s range:", s.min(), s.max(), " z range:", z.min(), z.max())
    print("physical x range on probes:", si.x.min(), si.x.max())
    print("physical y range on probes:", si.y.min(), si.y.max())

# --- Interpolate the fields ------------------------------------------------
u, v, w = fld.fields["vel"]
p = fld.fields["pres"][0]

si.interpolate_from_field_list(
    fld.t,
    [u, v, w, p],
    ["u", "v", "w", "p"],
    comm,
    write_data=False
)

# --- Check the user-facing contract ----------------------------------------
# Full fields are (N, Nz); with average_z=True get_field returns (N,) on the
# fly. Note: the fields only exist on rank 0; other ranks hold placeholders.
if rank == 0:
    uu = si.get_field("u")
    pp = si.get_field("p")
    assert uu.shape == (N, Nz), f"expected ({N},{Nz}), got {uu.shape}"
    assert pp.shape == (N, Nz)

    uu_avg = si.get_field("u", average_z=True)
    pp_avg = si.get_field("p", average_z=True)
    assert uu_avg.shape == (N,), f"expected ({N},), got {uu_avg.shape}"
    assert pp_avg.shape == (N,)
    assert np.allclose(uu_avg, uu.mean(axis=1)), "averaging broken"
    assert np.allclose(pp_avg, pp.mean(axis=1)), "averaging broken"
    print("field shapes OK, full:", uu.shape, " averaged:", uu_avg.shape)
    print("u on plate: min %.6g max %.6g mean %.6g" % (uu.min(), uu.max(), uu.mean()))
    print("p on plate: min %.6g max %.6g mean %.6g" % (pp.min(), pp.max(), pp.mean()))
    # No-slip wall -> |u| should be small everywhere along s
    print("max |u| along s:", np.abs(uu).max())
    print("TEST PASSED")
