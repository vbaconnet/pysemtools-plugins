import numpy as np

from pysemtools_plugins.EnhancedInterpolator import EnhancedInterpolator


class SurfaceInterpolator(EnhancedInterpolator):
    """
    Facilitated interpolator on a surface extruded uniformly in z.

    The user only interacts with curvilinear coordinates: an index pair
    (i, j) refers to the point at curvilinear coordinate s[i] on the
    surface and extrusion height z[j]. The s coordinates are mapped to
    physical (x, y) coordinates through the SurfaceDescriptor
    interpolators and the resulting extruded point cloud is handed to
    the EnhancedInterpolator constructor. Interpolated fields are
    returned as arrays of shape (N, Nz), where the first axis is the
    curvilinear direction and the second axis the extrusion direction.
    Spanwise (z) averaged fields of shape (N,) can be obtained on the
    fly through get_field(key, average_z=True).

    Parameters
    ----------
    surface : SurfaceDescriptor
        Surface descriptor of the profile. Must have had
        generate_interpolators() called on it.
    s : np.ndarray
        1D array of curvilinear coordinates along the surface (length N).
        These can be generated e.g. with
        np.linspace(surface.interpolate_s(xmin), surface.interpolate_s(xmax), N).
    z : np.ndarray
        1D array of z-coordinates of the extrusion (length Nz).
    **kwargs : dict
        Additional keyword arguments for the EnhancedInterpolator.
        Must include 'comm' (MPI communicator) and 'msh' (mesh).
    """

    def __init__(
            self,
            surface,
            s: np.ndarray,
            z: np.ndarray,
            **kwargs):

        s = np.atleast_1d(np.asarray(s))
        z = np.atleast_1d(np.asarray(z))

        # Map the curvilinear coordinates to physical coordinates on the surface
        x_line = surface.interpolate_x(s)
        y_line = surface.interpolate_y(s)

        # Extrude the profile uniformly in z.
        # Note that the arrays are of shape (N,Nz) here
        x, z_pts = np.meshgrid(x_line, z, indexing = "ij")
        y, _ = np.meshgrid(y_line, z, indexing = "ij")

        super().__init__(
            x = x,
            y = y,
            z = z_pts,
            **kwargs
        )

        # Bookkeeping in curvilinear coordinates.
        # Note that self.z (and self.x, self.y) remain the (N, Nz) probe
        # arrays set by the parent; z_line is the 1D extrusion input.
        self.surface = surface
        self.s = s
        self.z_line = z

    def get_field(self, key, average_z: bool = False):
        """
        Get an interpolated field.

        If average_z is True, the field is averaged over the extrusion
        direction (z) on the fly, and an array of shape (N,) is
        returned. Otherwise the full array of shape (N, Nz) is
        returned. Raises a KeyError if the field does not exist.
        """
        field = super().get_field(key)

        if average_z and field.ndim > 1:
            field = field.mean(axis = 1)

        return field
