from pysemtools.interpolation import Probes
from pysemtools.datatypes.field import NoOverwriteDict
from pysemtools.io.ppymech.neksuite import read_nekheader
from pysemtools.datatypes.msh import Mesh
from pysemtools_plugins.InterpolatorCache import InterpolatorCache
import numpy as np
import pickle

class EnhancedInterpolator:

    def __init__(self, x, y, z = None, fill_extrude_value = None, 
                 cache_dir = "", cache_key = None, force_recompute = False, **kwargs):
        """
        Initialize the EnhancedInterpolator with given coordinates and optional parameters.

        Parameters:
        ----------
        x : numpy.ndarray
            The x-coordinates of the points.
        y : numpy.ndarray
            The y-coordinates of the points.
        z : numpy.ndarray, optional
            The z-coordinates of the points. If not provided, fill_extrude_value must be provided.
        fill_extrude_value : float, optional
            The value to use for z-coordinates if z is not provided.
        cache_dir : str, optional
            Directory to store/load cache files. If None, caching is disabled.
            Currently, caching is only supported in serial mode (1 MPI rank).
        cache_key : str, optional
            Manual cache key override. If None, a key will be generated automatically.
        force_recompute : bool, optional
            If True, ignore cache and recompute even if cache exists. Default is False.
        **kwargs : dict
            Additional keyword arguments to pass to the Probes constructor.
            Must include 'comm' (MPI communicator) and 'msh' (mesh).
        """
        
        # Validate inputs
        assert x.shape == y.shape

        if z is not None: 
            assert x.shape == z.shape
        else:
            if fill_extrude_value is None:
                raise ValueError("Please provide a fill_extrude_value if z is not provided.")
            else:
                self.fill_extrude_value = fill_extrude_value

        self.x = x
        self.y = y 
        self.z = z

        self.orig_shape = self.x.shape

        self.interpolated_fields = NoOverwriteDict()

        # Caching configuration
        self.cache_dir = cache_dir
        self.cache_key = cache_key
        self.force_recompute = force_recompute
        self.cache_used = False  # Flag to track if cache was used
        
        # Must be specified, will throw a KeyError if not provided
        # comm is not added as an explicit optional argument since it is already 
        # a required argument for the Probes class, and we want to avoid redundancy.
        self.comm = kwargs.pop('comm')
        
        # Initialize cache manager if caching is enabled
        self.cache = InterpolatorCache(cache_dir) if cache_dir else None
        
        # Check if we're in serial mode (1 rank) - caching is simpler in this case
        self.is_serial = self.comm.Get_size() == 1
        
        # Try to load from cache if caching is enabled and not forcing recompute
        # For now, only support caching in serial mode
        if self.cache and not force_recompute and self.is_serial:
            # Peek at msh for cache key generation (will be popped later)
            msh_for_cache = kwargs.get('msh', None)
            if self._try_load_from_cache(kwargs, msh_for_cache):
                print("Loading from cache")
                return  # Cache was successfully loaded, skip normal initialization
        
        # Normal initialization (no cache or cache load failed)
        if self.comm.Get_rank() == 0:

            if z is None:
                xyz = np.column_stack((
                    x.ravel(),
                    y.ravel(),
                    fill_extrude_value * np.ones_like(x.ravel())
                ))
            
            else:
                xyz = np.column_stack((
                    x.ravel(),
                    y.ravel(),
                    z.ravel()
                ))

        else:
            xyz = None

        msh = kwargs.pop("msh")
        if (isinstance(msh, str)):
            header = read_nekheader(msh)
            if header.nb_dims == 2:
                raise ValueError("Mesh cannot be 2D in Enhanced Interpolator")
        elif isinstance(msh, Mesh):
            if msh.gdim == 2:
                raise ValueError("Mesh cannot be 2D in EnhancedInterpolator. Extrude your mesh before passing it here.")

        self.interpolator = Probes(
            self.comm,
            probes = xyz, 
            msh = msh,
            **kwargs
            )
        
        # Save to cache if caching is enabled (only in serial mode for now)
        if self.cache and self.is_serial and self.comm.Get_rank() == 0:
            self._save_to_cache(kwargs, msh)

    def interpolate_from_field_list(self, t, field_list, field_names, comm, write_data):

        """
        Interpolates a list of fields.
        If the field exists, overwrites.
        """

        # Do the interpolation
        # Note that the interpolated data will be stored on rank 0.
        self.interpolator.interpolate_from_field_list(
            t,
            field_list = field_list,
            field_names = field_names,
            comm = comm, 
            write_data = write_data)
        
        for i, name in enumerate(field_names, start=1):
            
            if self.comm.Get_rank() == 0:

                # Reshape the arrays to be like the originals
                tmp = np.reshape(
                    self.interpolator.interpolated_fields[:,i],
                    self.orig_shape
                    )
            
            else:
                # Add an empty field just to have a key present
                tmp = np.empty(1)

            self.add_field(name, tmp)

    def add_field(self, key, array):

        if key in self.interpolated_fields:
            self.interpolated_fields[key][...] = array
        else:
            self.interpolated_fields[key] = array

    def get_field(self, key):
        """
        Get a field from the interpolated fields.
        Raises a KeyError if the field does not exist.
        """
        try:
            return self.interpolated_fields[key]
        except KeyError as e:
            raise KeyError(f"Key {key} not found in the interpolated fields.") from e
    
    def _generate_cache_key(self, msh, kwargs):
        """
        Generate a cache key based on the current inputs.
        
        Parameters
        ----------
        msh : Mesh object or str
            Mesh information.
        kwargs : dict
            Keyword arguments passed to Probes constructor.
            
        Returns
        -------
        str
            A unique cache key.
        """
        if self.cache_key:
            return self.cache_key
        
        # Create a copy of kwargs without parameters we handle separately
        other_kwargs = {k: v for k, v in kwargs.items() 
                       if k not in ['msh', 'comm', 'probes']}
        
        return self.cache.generate_cache_key(
            self.x, self.y, self.z, 
            getattr(self, 'fill_extrude_value', None),
            msh, 
            **other_kwargs
        )
    
    def _try_load_from_cache(self, kwargs, msh):
        """
        Try to load interpolation data from cache.
        
        This method attempts to load a pickled Probes object from cache.
        If successful, it sets self.interpolator to the loaded object.
        
        Parameters
        ----------
        kwargs : dict
            Keyword arguments passed to Probes constructor.
        msh : Mesh object or str
            Mesh information.
            
        Returns
        -------
        bool
            True if cache was successfully loaded, False otherwise.
        """
        # Generate cache key
        cache_key = self._generate_cache_key(msh, kwargs)
        
        # Load from cache
        cached_data = self.cache.load(cache_key)
        
        if cached_data is None:
            return False
        
        # Extract cached data
        data_dict = cached_data['data']
        
        # Try to unpickle the Probes object
        if 'probes_object' in data_dict:
            try:
                # Convert bytes back to file-like object for unpickling
                import io
                probes_bytes = data_dict['probes_object']
                
                # Handle both bytes and numpy array of bytes
                if isinstance(probes_bytes, np.ndarray):
                    if probes_bytes.dtype == np.uint8:
                        probes_bytes = bytes(probes_bytes)
                
                self.interpolator = pickle.loads(probes_bytes)
                
                # Mark that cache was used
                self.cache_used = True
                
                return True
            except Exception as e:
                print(f"Warning: Failed to unpickle cached Probes object: {e}")
                return False
        
        return False
    
    def _save_to_cache(self, kwargs, msh):
        """
        Save interpolation point-finding data to cache.
        
        This method pickles the entire Probes object and saves it to cache.
        
        Parameters
        ----------
        kwargs : dict
            Keyword arguments passed to Probes constructor.
        msh : Mesh object or str
            Mesh information.
        """
        # Generate cache key
        cache_key = self._generate_cache_key(msh, kwargs)
        
        # Pickle the Probes object
        try:
            probes_pickle = pickle.dumps(self.interpolator)
        except Exception as e:
            print(f"Warning: Failed to pickle Probes object: {e}")
            return
        
        # Store as numpy array of bytes for HDF5 compatibility
        probes_bytes = np.frombuffer(probes_pickle, dtype=np.uint8)
        
        # Create data dict
        data_dict = {
            'probes_object': probes_bytes,
        }
        
        # Create metadata
        metadata_dict = {
            'n_probes': self.interpolator.n_probes if hasattr(self.interpolator, 'n_probes') else len(self.x.ravel()),
            'orig_shape': list(self.orig_shape),
            'xyz_hash': self.cache._hash_coordinates(
                self.x, self.y, self.z, getattr(self, 'fill_extrude_value', None)
            ),
            'pickle_size_mb': len(probes_bytes) / (1024 * 1024),
        }
        
        # Save to cache
        self.cache.save(cache_key, data_dict, metadata_dict)
        
        # Set the cache key for this instance
        self.cache_key = cache_key
    
    def get_cache_info(self):
        """
        Get information about the cache used for this instance.
        
        Returns
        -------
        dict or None
            Cache metadata if cache was used, None otherwise.
        """
        if not self.cache_used or not self.cache_key:
            return None
        
        return self.cache.get_cache_info(self.cache_key)
    
    def clear_cache(self):
        """Clear the cache for this instance if a cache key was generated."""
        if self.cache and self.cache_key:
            self.cache.clear_cache(self.cache_key)
            self.cache_key = None
