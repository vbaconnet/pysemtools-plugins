import numpy as np

from pysemtools_plugins.EnhancedInterpolator import EnhancedInterpolator


class SurfaceInterpolator:
    """
    Facilitated interpolator on a surface extruded uniformly in z.

    The user only interacts with curvilinear coordinates: an index pair
    (i, j) refers to the point at curvilinear coordinate s[i] on the
    surface and extrusion height z[j]. Inside the object, the s
    coordinates are mapped to physical (x, y) coordinates through the
    SurfaceDescriptor interpolators and handed to an EnhancedInterpolator.
    Interpolated fields are returned as arrays of shape (N, Nz), where
    the first axis is the curvilinear direction and the second axis the
    extrusion direction. Spanwise (z) averaged fields of shape (N,) can
    be obtained on the fly through get_field(key, average_z=True).

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

        self.surface = surface
        self.s = s
        self.z = z

        # Map the curvilinear coordinates to physical coordinates on the surface
        x_line = surface.interpolate_x(s)
        y_line = surface.interpolate_y(s)

        # Extrude the profile uniformly in z
        self.x, self.z_pts = np.meshgrid(x_line, z, indexing = "ij")
        self.y, _ = np.meshgrid(y_line, z, indexing = "ij")

        # note that arrays are of shape (N,Nz) here
        self.interpolator = EnhancedInterpolator(
            x = self.x,
            y = self.y,
            z = self.z_pts,
            **kwargs
        )

    def interpolate_from_field_list(self, t, field_list, field_names, comm, **kwargs):
        """
        Interpolate a list of fields on the extruded surface.

        The interpolated fields are stored as arrays of shape (N, Nz),
        with the first axis along the curvilinear coordinate s and the
        second axis along the extrusion z.

        Parameters
        ----------
        t : float
            Time value for the interpolation.
        field_list : list
            List of fields to interpolate.
        field_names : list[str]
            Names of the fields to use for interpolation.
        comm : MPI communicator
            Communicator to use for the interpolation.
        """

        self.interpolator.interpolate_from_field_list(
            t,
            field_list = field_list,
            field_names = field_names,
            comm = comm,
            **kwargs
        )

    def add_field(self, key, array):
        self.interpolator.add_field(key, array)

    def get_field(self, key, average_z: bool = False):
        """
        Get an interpolated field.

        If average_z is True, the field is averaged over the extrusion
        direction (z) on the fly, and an array of shape (N,) is
        returned. Otherwise the full array of shape (N, Nz) is
        returned. Raises a KeyError if the field does not exist.
        """
        field = self.interpolator.get_field(key)

        if average_z and field.ndim > 1:
            field = field.mean(axis = 1)

        return field
