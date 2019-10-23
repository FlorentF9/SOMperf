"""
Topology functions
"""


def rectangular_topology_dist(map_size):
    """Rectangular topology distance function.

    Returns the distance function between two units on a rectangular map (Manhattan distance).

    Parameters
    ----------
    map_size : tuple (height, width)
        SOM height and width.

    Returns
    -------
    dist_fun : (k : int, l : int) => int
        distance function between units k and l on the map.
    """
    def dist_fun(k, l):
        return abs(k // map_size[1] - l // map_size[1]) + abs(k % map_size[1] - l % map_size[1])
    return dist_fun


def square_topology_dist(map_size):
    """Square topology distance function.

    Returns the distance function between two units on a square map (Manhattan distance).

    Parameters
    ----------
    map_size : int
        SOM height or width.

    Returns
    -------
    dist_fun : function (k : int, l : int) => int
        distance function between units k and l on the map.
    """
    def dist_fun(k, l):
        return abs(k // map_size - l // map_size) + abs(k % map_size - l % map_size)
    return dist_fun
