# pySEMTools-plugins

Plugin modules for [pySEMTools](https://github.com/ICCS-FDS/pySEMTools), extending its functionality for stream processing and interpolation.

## Modules

| File | Description |
|------|-------------|
| `EnhancedStreamer.py` | Extended stream processor with ADIOS2 support for receiving mesh and field data |
| `EnhancedInterpolator.py` | Enhanced interpolator with caching support for point-finding operations |
| `InterpolatorCache.py` | Cache manager for storing/loading interpolation point-finding data (HDF5 + JSON) |
| `FlatPlateStreamProcessor.py` | Specialized processor for flat plate geometries with boundary layer plane generation |

## Installation

Install in editable mode (recommended for development):

```bash
cd /path/to/pysemtools-plugins
git clone git@github.com:vbaconnet/pysemtools-plugins.git
cd pysemtools-plugins
pip install --editable .
```

This will install the package with dependencies:
- `numpy`
- `scipy`
- `h5py`

## Requirements

The following must be available in your Python environment:
- `pySEMTools` - The main SEM tools package
- `adios2` - ADIOS2 Python bindings (provided by pySEMTools)

Set up your environment:

```bash
# Example: Add pySEMTools to PYTHONPATH
export PYTHONPATH=/path/to/pySEMTools:$PYTHONPATH

# Then install this package
pip install --editable .
```

## Usage

```python
from pysemtools_plugins import EnhancedStreamer, EnhancedInterpolator
from pysemtools_plugins.FlatPlateStreamProcessor import FlatPlateStreamProcessor

# Use the modules with your MPI communicator
streamer = EnhancedStreamer(comm=comm, fields=["velocity", "pressure"])
# ...
```

## License

MIT
