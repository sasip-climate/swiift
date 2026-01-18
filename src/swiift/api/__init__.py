"""TODO"""

from . import spectra, utils
from .api import Experiment, load_pickle, load_pickles

# Explicitly expose the members of the module `api.py`.
# Exclude the module itself to avoid shadowing the subpackage `api/`;
# the namespace will however exists due to the `from .api` import.
# The namespaces associated with the two other submodules are added explicitly,
# so they can be added to a higher-level namespace.
__all__ = [s for s in dir() if not s.startswith("_") and s != "api"]
