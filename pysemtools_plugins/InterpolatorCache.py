"""
Cache manager for EnhancedInterpolator.

This module provides caching functionality for the point-finding step
in the interpolation process, which is the most expensive operation.
"""

import hashlib
import json
import os
from datetime import datetime
import numpy as np

try:
    import h5py
except ImportError:
    h5py = None


def get_pysemtools_version():
    """Get the version of pysemtools."""
    try:
        import pysemtools
        return getattr(pysemtools, '__version__', 'unknown')
    except ImportError:
        return 'unknown'


class InterpolatorCache:
    """
    Cache manager for interpolation point-finding data.
    
    This class handles the storage and retrieval of cached point-finding results
    to avoid recomputing expensive operations when the same interpolation
    is performed multiple times.
    
    The cache is stored in HDF5 files with accompanying JSON metadata files.
    
    Parameters
    ----------
    cache_dir : str
        Directory where cache files will be stored.
    """
    
    # Key parameters that affect point finding and should be included in cache key
    POINT_FINDING_PARAMS = [
        'find_points_tol',
        'find_points_max_iter',
        'elem_percent_expansion',
        'global_tree_type',
        'global_tree_nbins',
        'local_data_structure',
        'use_oriented_bbox',
        'find_points_comm_pattern',
        'find_points_iterative',
        'point_interpolator_type',
        'max_pts',
        'clean_search_traces',
    ]
    
    def __init__(self, cache_dir):
        """Initialize the cache manager."""
        self.cache_dir = cache_dir
        
        # Create cache directory if it doesn't exist
        if self.cache_dir and not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)
    
    def generate_cache_key(self, x, y, z, fill_extrude_value, msh, **kwargs):
        """
        Generate a unique cache key based on interpolation inputs.
        
        Parameters
        ----------
        x, y, z : numpy.ndarray or None
            Coordinate arrays. z can be None if fill_extrude_value is provided.
        fill_extrude_value : float or None
            Value to use for z-coordinates if z is None.
        msh : Mesh object, str, or None
            Mesh information (file path or Mesh object).
        **kwargs : dict
            Additional keyword arguments passed to Probes.
            
        Returns
        -------
        str
            A unique cache key string.
        """
        # Hash the coordinates
        coord_hash = self._hash_coordinates(x, y, z, fill_extrude_value)
        
        # Hash mesh information
        mesh_hash = self._hash_mesh(msh)
        
        # Hash relevant kwargs that affect point finding
        kwargs_hash = self._hash_kwargs(kwargs)
        
        # Combine all hashes
        cache_key = f"{coord_hash}_{mesh_hash}_{kwargs_hash}"
        
        # Add fill_extrude_value to the key (it affects the result)
        if fill_extrude_value is not None:
            fev_hash = hashlib.sha256(str(fill_extrude_value).encode()).hexdigest()[:16]
            cache_key = f"{cache_key}_{fev_hash}"
        
        return cache_key
    
    def _hash_coordinates(self, x, y, z, fill_extrude_value):
        """Hash the coordinate arrays."""
        hasher = hashlib.sha256()
        
        # Hash x and y
        hasher.update(x.tobytes())
        hasher.update(y.tobytes())
        
        # Hash z or fill_extrude_value
        if z is not None:
            hasher.update(z.tobytes())
        elif fill_extrude_value is not None:
            hasher.update(str(fill_extrude_value).encode())
        
        return hasher.hexdigest()
    
    def _hash_mesh(self, msh):
        """Hash the mesh information."""
        if msh is None:
            return "none"
        
        if isinstance(msh, str):
            # Hash the file path and its modification time
            if os.path.exists(msh):
                mtime = os.path.getmtime(msh)
                return hashlib.sha256(f"{msh}_{mtime}".encode()).hexdigest()
            else:
                return hashlib.sha256(msh.encode()).hexdigest()
        
        # For Mesh objects, hash the coordinates and dimensions
        try:
            mesh_hash = hashlib.sha256()
            # Include gdim to differentiate 2D and 3D meshes
            if hasattr(msh, 'gdim'):
                mesh_hash.update(str(msh.gdim).encode())
            mesh_hash.update(msh.x.tobytes())
            mesh_hash.update(msh.y.tobytes())
            mesh_hash.update(msh.z.tobytes())
            return mesh_hash.hexdigest()
        except AttributeError:
            # Fallback: use object id (not ideal but works)
            return hashlib.sha256(str(id(msh)).encode()).hexdigest()
    
    def _hash_kwargs(self, kwargs):
        """Hash the relevant kwargs that affect point finding."""
        # Filter to only point-finding parameters
        relevant_kwargs = {
            k: v for k, v in kwargs.items() 
            if k in self.POINT_FINDING_PARAMS
        }
        
        # Convert to JSON string for consistent hashing
        kwargs_str = json.dumps(relevant_kwargs, sort_keys=True, default=str)
        return hashlib.sha256(kwargs_str.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key):
        """Get the file paths for a cache entry."""
        if not self.cache_dir:
            return None, None
        
        data_path = os.path.join(self.cache_dir, f"{cache_key}_data.hdf5")
        meta_path = os.path.join(self.cache_dir, f"{cache_key}_meta.json")
        
        return data_path, meta_path
    
    def exists(self, cache_key):
        """Check if a cache entry exists and is valid."""
        if not self.cache_dir:
            return False
        
        data_path, meta_path = self._get_cache_path(cache_key)
        
        # Check if both files exist
        if not os.path.exists(data_path) or not os.path.exists(meta_path):
            return False
        
        # Validate cache (check version, etc.)
        return self._validate_cache(meta_path)
    
    def _validate_cache(self, meta_path):
        """Validate that a cache entry is still valid."""
        try:
            with open(meta_path, 'r') as f:
                metadata = json.load(f)
            
            # Check pysemtools version
            current_version = get_pysemtools_version()
            cached_version = metadata.get('pysemtools_version', 'unknown')
            
            if cached_version != current_version:
                # Version mismatch - cache may be invalid
                # For now, we'll still use it but this could be configurable
                pass
            
            return True
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            return False
    
    def save(self, cache_key, data_dict, metadata_dict, rank=None):
        """
        Save cache data to disk.
        
        Parameters
        ----------
        cache_key : str
            The cache key.
        data_dict : dict
            Dictionary containing numpy arrays to save.
        metadata_dict : dict
            Dictionary containing metadata to save.
        rank : int, optional
            MPI rank. If provided, appends rank to cache key for per-rank caching.
            
        Returns
        -------
        bool
            True if save was successful, False otherwise.
        """
        if not self.cache_dir or h5py is None:
            return False
        
        # For per-rank data, append rank to cache key
        if rank is not None:
            full_cache_key = f"{cache_key}_rank{rank}"
        else:
            full_cache_key = cache_key
        
        data_path, meta_path = self._get_cache_path(full_cache_key)
        
        try:
            # Save HDF5 data
            with h5py.File(data_path, 'w') as f:
                for key, value in data_dict.items():
                    if isinstance(value, np.ndarray):
                        # Use compression for large arrays
                        if value.nbytes > 1024 * 1024:  # > 1MB
                            f.create_dataset(key, data=value, compression='gzip')
                        else:
                            f.create_dataset(key, data=value)
                    else:
                        # Save scalar values as attributes or datasets
                        f.attrs[key] = value
            
            # Save metadata
            metadata = {
                'cache_key': full_cache_key,
                'created': datetime.now().isoformat(),
                'pysemtools_version': get_pysemtools_version(),
            }
            metadata.update(metadata_dict)
            
            with open(meta_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            return True
            
        except Exception as e:
            print(f"Warning: Failed to save cache: {e}")
            return False
    
    def load(self, cache_key, rank=None):
        """
        Load cache data from disk.
        
        Parameters
        ----------
        cache_key : str
            The cache key.
        rank : int, optional
            MPI rank. If provided, appends rank to cache key for per-rank loading.
            
        Returns
        -------
        dict or None
            Dictionary containing loaded data, or None if cache doesn't exist or is invalid.
        """
        if not self.cache_dir or h5py is None:
            return None
        
        # For per-rank data, append rank to cache key
        if rank is not None:
            full_cache_key = f"{cache_key}_rank{rank}"
        else:
            full_cache_key = cache_key
        
        data_path, meta_path = self._get_cache_path(full_cache_key)
        
        # Check if cache exists and is valid
        if not self.exists(full_cache_key):
            return None
        
        try:
            # Load HDF5 data
            data_dict = {}
            with h5py.File(data_path, 'r') as f:
                for key in f.keys():
                    data_dict[key] = f[key][:]
                
                # Load attributes
                for key in f.attrs.keys():
                    data_dict[key] = f.attrs[key]
            
            # Load metadata
            with open(meta_path, 'r') as f:
                metadata = json.load(f)
            
            return {
                'data': data_dict,
                'metadata': metadata
            }
            
        except Exception as e:
            print(f"Warning: Failed to load cache: {e}")
            return None
    
    def get_cache_info(self, cache_key):
        """
        Get information about a cache entry without loading the data.
        
        Parameters
        ----------
        cache_key : str
            The cache key.
            
        Returns
        -------
        dict or None
            Dictionary containing cache metadata, or None if cache doesn't exist.
        """
        if not self.cache_dir:
            return None
        
        _, meta_path = self._get_cache_path(cache_key)
        
        if not os.path.exists(meta_path):
            return None
        
        try:
            with open(meta_path, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return None
    
    def list_caches(self):
        """List all cache entries in the cache directory."""
        if not self.cache_dir or not os.path.exists(self.cache_dir):
            return []
        
        cache_keys = []
        for filename in os.listdir(self.cache_dir):
            if filename.endswith('_meta.json'):
                cache_key = filename.replace('_meta.json', '')
                cache_keys.append(cache_key)
        
        return cache_keys
    
    def clear_cache(self, cache_key=None):
        """
        Clear cache entries.
        
        Parameters
        ----------
        cache_key : str or None
            If provided, clear only this cache entry. If None, clear all caches.
        """
        if not self.cache_dir:
            return
        
        if cache_key is not None:
            data_path, meta_path = self._get_cache_path(cache_key)
            for path in [data_path, meta_path]:
                if os.path.exists(path):
                    os.remove(path)
        else:
            # Clear all caches
            for cache_key in self.list_caches():
                self.clear_cache(cache_key)
