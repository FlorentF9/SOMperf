"""
The `somperf.utils` module contains utility functions.
"""

from .topology import rectangular_topology_dist
from .topology import square_topology_dist

from .neighborhood import gaussian_neighborhood
from .neighborhood import window_neighborhood

__all__ = [
    'rectangular_topology_dist',
    'square_topology_dist',
    'gaussian_neighborhood',
    'window_neighborhood'
]
