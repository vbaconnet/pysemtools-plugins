from pysemtools_plugins.EnhancedInterpolator import EnhancedInterpolator
from pysemtools_plugins.EnhancedStreamer import EnhancedStreamer
import numpy as np


def compute_normal_tangent(xa,xb,ya,yb):
    """
    Compute the normal and tangent vector from two points A and B.

    Returns
    -------
    n : np.ndarray
        normal vector.
    t : np.ndarray
        tangent vector.

    """
    
    dx = xb-xa
    dy = yb-ya
    t = np.array([dx, dy])/np.sqrt(dx**2 + dy**2)
    n = np.array([-dy, dx])/np.sqrt(dx**2 + dy**2)
    return n,t

def compute_normals(x, y):
    """
    Compute normals along the profile (x,y). Use central differences and one-sided
    on the boundaries. Works also if the domain is periodic.
    
    Returns a numpy array of size (N,2) where each column is the x and y coordinate.
    """

    assert len(x) == len(y)

    N = len(x)
    n = np.zeros((N,2))
    
    # Interior points
    for i in range(1, N-1):
        n[i,:], _ = compute_normal_tangent(x[i-1], x[i+1], y[i-1], y[i+1])

    n[0 ,:], _ = compute_normal_tangent(x[ 0], x[ 1], y[ 0], y[ 1])
    n[-1,:], _ = compute_normal_tangent(x[-2], x[-1], y[-2], y[-1])
    
    return n

def compute_curvilinear_distance(x, y, normalize = False):
    """
    Generate an array of curvilinear coordinates.

    Returns
    -------
    s : numpy array
        Curvilinear coordinates.

    """
    assert len(x) == len(y)
    s = np.zeros_like(x)
    
    dx = np.diff(x)
    dy = np.diff(y)
    
    s = np.cumsum(np.sqrt( dx**2 + dy**2 ))
    s = np.concatenate([[0.0], s])
    
    if normalize:
        s = s / np.sum(s)

    return s

def generate_interpolators(x, y, s, n):
    """
    Generate interpolators based on curvilinear coordinates, for (x,y) coordinates
    and all the normals along the profile.

    """
    from scipy import interpolate

    assert len(x) == len(y) == len(s)
    assert n.shape[1] == 2

    # Generate curvilinear interpolators
    interp_x = interpolate.interp1d(s, x, kind='cubic')
    interp_y = interpolate.interp1d(s, y, kind='cubic')

    # Generate interpolators for x and y components of the normals
    interp_normals_x = interpolate.interp1d(s, n[:,0], kind='cubic')
    interp_normals_y = interpolate.interp1d(s, n[:,1], kind='cubic')

    # Generate curvilinear interpolator, maps x -> s
    # i.e. if i give you an x coordinate you give me the corresponding 
    # curvilinear coordinate s
    interp_s = interpolate.interp1d(x, s, kind='cubic')

    return interp_x, interp_y, interp_normals_x, interp_normals_y, interp_s


class FlatPlateStreamProcessor(EnhancedStreamer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def finalize(self):
        super().finalize()
    
    def read_plate_coordinates(self, path_to_unique_coordinates):
        """
        Read plate coordinates from a file and compute necessary attributes.

        Parameters
        ----------
        path_to_unique_coordinates : str
            Path to the file containing the plate coordinates.

        Returns
        -------
        None
        """
        data = np.genfromtxt(
            path_to_unique_coordinates,
            delimiter = ",",
            skip_header = 1)
        
        self.plate_coordinates = {
            'x': data[:,0],
            'y': data[:,1]
            }

        # --- Point generation
        # returns [N,2] array with coordinates of normal vectors
        self.n = compute_normals(
            self.plate_coordinates["x"],
            self.plate_coordinates["y"]
            )

        # Compute the curvilinear coordinates
        self.s = compute_curvilinear_distance(
            self.plate_coordinates["x"],
            self.plate_coordinates["y"],
            normalize=False
            )

        # Now generate the interpolators for plotting BLs
        # the interpolators are based on curvilinear distance self.s
        self.i_x, self.i_y, self.i_nx, self.i_ny, self.i_s = generate_interpolators(
            self.plate_coordinates["x"],
            self.plate_coordinates["y"],
            self.s,
            self.n
        )
        # ---

    def generate_BL_plane(self, xmin, xmax, Nx=1000, Ly = 0.1, Ny = 1000, g = 2):
        """
        Generate an interpolator object with points normal to the blade.

        Parameters
        ----------
        xmin : float
            Minimum x-coordinate for the blade.
        xmax : float
            Maximum x-coordinate for the blade.
        Nx : int, optional
            Number of points along the blade (default is 1000).
        Ly : float, optional
            Length of the line normal to the blade (default is 0.1).
        Ny : int, optional
            Number of points along the normal line (default is 1000).
        g : float, optional
            Gain for the point distribution (default is 2).

        Returns
        -------
        Probes
            An interpolator object with points normal to the blade.

        """

        # Do some checks first
        for att in ['i_x', 'i_y', 'i_nx', 'i_ny', 'i_s']:
            self.check_attr(att)

        smin = self.i_s(xmin)
        smax = self.i_s(xmax)
        s = np.linspace(smin, smax, Nx)

        x_plate = self.i_x(s)
        y_plate = self.i_y(s)

        x_pts = np.zeros((Nx,Ny))
        y_pts = np.zeros_like(x_pts)

        for i in range(Nx):

            s_tgt = self.i_s(x_plate[i])

            # Interpolate the other stuff
            nx_tgt = self.i_nx(s_tgt)
            ny_tgt = self.i_ny(s_tgt)

            def gen_line_from_normal(n, Npts, xstart, ystart, L, g):
                xline = np.zeros(Npts)
                yline = np.zeros(Npts)

                # Now generate a line with length L
                s_bbox = [L, 0.0] # yes it is reverted, because we're doing a half tanh
                s = pcs.generate_1d_arrays(s_bbox, Npts, mode="half_tanh", gain=g)
                s = np.flip(s) # flip the distribution
                xline = xstart + s*n[0]
                yline = ystart + s*n[1]
                return xline, yline

            # Generate points along the normal [nx_tgt, ny_tgt] at the 
            # starting point given by (x_tgt, y_tgt)
            x_pts[i,:], y_pts[i,:] = gen_line_from_normal([nx_tgt, ny_tgt], Ny, 
                                                x_plate[i], y_plate[i], Ly, g)

        probes = EnhancedInterpolator(
            x = x_pts, 
            y = y_pts,
            fill_extrude_value=0.0,
            comm = self.comm, 
            msh = self.mesh, 
            point_interpolator_type='multiple_point_legendre_numpy',
            max_pts=256, 
            find_points_comm_pattern='point_to_point',
            write_coords=False
            )


        return probes
