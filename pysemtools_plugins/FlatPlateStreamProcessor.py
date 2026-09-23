from multiprocessing import Value

from pysemtools_plugins.EnhancedInterpolator import EnhancedInterpolator
from pysemtools_plugins.EnhancedStreamer import EnhancedStreamer
import numpy as np
from pysemtools.interpolation.pointclouds import generate_1d_arrays
from pysemtools_plugins.SurfaceDescriptor import SurfaceDescriptor


class FlatPlateStreamProcessor(EnhancedStreamer):

    def __init__(
            self, 
            comm, 
            fields, 
            adios2_timeout=300,
            create_catalyst_session: bool = False,
            catalyst_pipeline: str | None = None,
            catalyst_channel: str | None = None, **kwargs):

        self.surface: SurfaceDescriptor

        super().__init__(
            comm, 
            fields, 
            adios2_timeout, 
            create_catalyst_session,
            catalyst_pipeline,
            catalyst_channel,
            **kwargs)

        

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

        self.surface = SurfaceDescriptor(
            data[:,0],
            data[:,1]
        )

        self.surface.generate_interpolators()

    def generate_points_on_plate(
            self, 
            xmin, 
            xmax, 
            Nx, 
            distribution = "uniform",
            gain: int = 3,
            interpolator_name: str = "",
            **kwargs):

        """
        Generate points on the plate surface.

        Parameters
        ----------
        xmin : float
            Minimum x-coordinate for the plate.
        xmax : float
            Maximum x-coordinate for the plate.
        Nx : int
            Number of points to generate along the plate.
        distribution : str, optional
            Distribution of points along the plate. Options are 'uniform' or 'refine_left' (default is 'uniform').
        gain : int, optional
            Gain for the point distribution (default is 3).

        Returns
        -------
        tuple
            A tuple containing the x and y coordinates of the generated points.
        """
        smin = self.surface.interpolate_s(xmin)
        smax = self.surface.interpolate_s(xmax)

        if distribution == "uniform":
            s = np.linspace(smin, smax, Nx)
        elif distribution == "refine_left":
            s_bbox = [smax, smin]  # yes it is reverted, because we're doing a half tanh
            s = generate_1d_arrays(s_bbox, Nx, mode="half_tanh", gain=gain)
            s = np.flip(s)  # flip the distribution
        else:
            raise ValueError(f"{distribution} distribution not supported (only 'uniform' and 'refine_left')")

        x_plate = self.surface.interpolate_x(s)
        y_plate = self.surface.interpolate_y(s)

        if interpolator_name:
            self.add_interpolator_from_values(
                name = interpolator_name,
                x = x_plate, 
                y = y_plate,
                z = 0.0,
                comm = self.comm, 
                msh = self.msh,
                **kwargs
            )

        return x_plate, y_plate

    def generate_BL_plane(
            self,
            xmin,
            xmax,
            Nx=1000,
            Ly = 0.1,
            Ny = 1000,
            g = 2.0,
            interpolator_name = "plate",
            x_distribution = "uniform",
            g_s: int = 3,
            **kwargs):
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

        x_plate, y_plate = self.generate_points_on_plate(
            xmin,
            xmax,
            Nx,
            x_distribution,
            g_s
        )

        x_pts = np.zeros((Nx,Ny))
        y_pts = np.zeros_like(x_pts)

        for i in range(Nx):

            s_tgt = self.surface.interpolate_s(x_plate[i])

            # Interpolate the other stuff
            nx_tgt, ny_tgt = self.surface.interpolate_n(s_tgt)

            def gen_line_from_normal(n, Npts, xstart, ystart, L, g):
                xline = np.zeros(Npts)
                yline = np.zeros(Npts)

                # Now generate a line with length L
                s_bbox = [L, 0.0] # yes it is reverted, because we're doing a half tanh
                s = generate_1d_arrays(s_bbox, Npts, mode="half_tanh", gain=g)
                s = np.flip(s) # flip the distribution
                xline = xstart + s*n[0]
                yline = ystart + s*n[1]
                return xline, yline

            # Generate points along the normal [nx_tgt, ny_tgt] at the 
            # starting point given by (x_tgt, y_tgt)
            x_pts[i,:], y_pts[i,:] = gen_line_from_normal([nx_tgt, ny_tgt], Ny, 
                                                x_plate[i], y_plate[i], Ly, g)

        self.add_interpolator_from_values(
            name = interpolator_name,
            x = x_pts, 
            y = y_pts,
            z = 0.0,
            comm = self.comm, 
            msh = self.msh,
            **kwargs
        )

    def compute_bl_chars(self, u, x, y, dudy = None, dudy_limit = 0.001):
        """
        Compute boundary layer characteristics:
        - delta_star, displacement thickness
        - theta, momentum thickness
        - H, shape factor

        given a series of boundary layer profiles u[Nx, Ny] along x.
        """
        
        nx = x.shape[0]
        theta = np.zeros(nx)
        delta_star = np.zeros(nx)
        H = np.zeros(nx)
        for i in range(nx):

            if dudy:
                yend = np.where(dudy[i,:]/dudy[i,0] < dudy_limit)[0][0]
            else:
                yend = -1

            yy = y[i,:yend] - y[i,0]
            uu = u[i,:yend]/u[i,yend]
            #print(yy[-1])

            delta_star[i] = np.trapezoid(1-uu, yy)
            theta[i] = np.trapezoid(uu*(1-uu), yy)
            H[i] = delta_star[i] / theta[i]

        return delta_star, theta, H

