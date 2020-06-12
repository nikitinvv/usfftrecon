from pkg_resources import get_distribution, DistributionNotFound

from usfftrecon.radonusfft import *
from usfftrecon.solver_tomo import *
try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    # package is not installed
    pass