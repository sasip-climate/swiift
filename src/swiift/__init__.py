"""SWIIFT: Surface Waves Impact on sea Ice---Fracture Toolkit.

Subpackages
===========

There are three subpackages that need to be explicitly imported:

::

    api
    model
    lib

"""

from swiift.__about__ import __version__

from .api import *
from .lib import att
from .model import *
from .model import frac_handlers as fh
