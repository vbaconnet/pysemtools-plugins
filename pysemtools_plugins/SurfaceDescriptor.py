import numpy as np
from scipy import interpolate

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
    
    dx = xb - xa
    dy = yb - ya
    t = np.array([dx, dy])/np.sqrt(dx**2 + dy**2)
    n = np.array([-dy, dx])/np.sqrt(dx**2 + dy**2)
    return n,t

class SurfaceDescriptor:

    def __init__(
            self, 
            x: np.ndarray, 
            y: np.ndarray,
            periodic: bool = False):
        
        self.x = x
        self.y = y
        self.periodic = periodic

        # Curvilinear coordinates
        self.s = np.zeros_like(x)

        # Normal and tangent vectors
        self.n = np.zeros((len(x),2))
        self.t = np.zeros_like(self.n)

        # Interpolators to resolve any points on the boundary
        self.interpolators = {}

        #
        # Now construct everything
        #
        self.compute_normals_and_tangents()
        self.compute_curvilinear_distance(normalize = False)

    def compute_normals_and_tangents(self):
        """
        Compute normals along the profile (x,y). Use central differences and one-sided
        on the boundaries. Works also if the domain is periodic.
        
        Returns a numpy array of size (N,2) where each column is the x and y coordinate.
        """

        N = len(self.x)
        assert self.n.shape == (N,2)
        assert self.t.shape == self.n.shape
        
        # Interior points
        for i in range(1, N-1):
            self.n[i,:], self.t[i,:] = compute_normal_tangent(self.x[i-1], self.x[i+1], self.y[i-1], self.y[i+1])

        # Exterior points
        if self.periodic:
            self.n[0 ,:], self.t[0,:] = compute_normal_tangent(self.x[-1], self.x[ 1], self.y[-1], self.y[ 1])
            self.n[-1,:], self.t[-1,:] = compute_normal_tangent(self.x[-2], self.x[ 0], self.y[-2], self.y[ 0])
        else:
            self.n[0 ,:], self.t[0,:] = compute_normal_tangent(self.x[ 0], self.x[ 1], self.y[ 0], self.y[ 1])
            self.n[-1,:], self.t[-1,:] = compute_normal_tangent(self.x[-2], self.x[-1], self.y[-2], self.y[-1])
        
        return self.n

    def compute_curvilinear_distance(self, normalize = False):
        """
        Generate an array of curvilinear coordinates.

        Returns
        -------
        s : numpy array
            Curvilinear coordinates.

        """
        assert len(self.x) == len(self.y)
        assert len(self.s) == len(self.x)

        dx = np.diff(self.x)
        dy = np.diff(self.y)
        
        self.s = np.cumsum(np.sqrt( dx**2 + dy**2 ))
        self.s = np.concatenate([[0.0], self.s])
        
        if normalize:
            self.s = self.s / np.sum(self.s) # I guess this should be / self.s[-1]?

        return self.s

    def generate_interpolators(self):
        """
        Generate interpolators based on curvilinear coordinates, for (x,y) coordinates
        and all the normals along the profile.

        """

        assert len(self.x) == len(self.y) == len(self.s)
        assert self.n.shape[1] == 2

        # Generate curvilinear interpolators
        interp_x = interpolate.interp1d(self.s, self.x, kind='cubic')
        interp_y = interpolate.interp1d(self.s, self.y, kind='cubic')

        # Generate interpolators for x and y components of the normals
        interp_normals_x = interpolate.interp1d(self.s, self.n[:,0], kind='cubic')
        interp_normals_y = interpolate.interp1d(self.s, self.n[:,1], kind='cubic')
        interp_tangents_x = interpolate.interp1d(self.s, self.t[:,0], kind='cubic')
        interp_tangents_y = interpolate.interp1d(self.s, self.t[:,1], kind='cubic')

        # Generate curvilinear interpolator, maps x -> s
        # i.e. if i give you an x coordinate you give me the corresponding 
        # curvilinear coordinate s
        interp_s = interpolate.interp1d(self.x, self.s, kind='cubic')

        self.interpolators["x"] = interp_x
        self.interpolators["y"] = interp_y
        self.interpolators["nx"] = interp_normals_x
        self.interpolators["ny"] = interp_normals_y
        self.interpolators["tx"] = interp_tangents_x
        self.interpolators["ty"] = interp_tangents_y
        self.interpolators["s"] = interp_s

    def interpolate_x(self, val):
        return self.interpolators["x"](val)

    def interpolate_y(self, val):
        return self.interpolators["y"](val)

    def interpolate_s(self, val):
        return self.interpolators["s"](val)

    def interpolate_n(self, sval):
        nx = self.interpolators["nx"](sval)
        ny = self.interpolators["ny"](sval)
        return nx, ny

    def interpolate_t(self, sval):
        tx = self.interpolators["tx"](sval)
        ty = self.interpolators["ty"](sval)
        return tx, ty
    