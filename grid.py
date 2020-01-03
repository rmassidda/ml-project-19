from itertools import product

class Grid:
    """
    A list of dictionaries is used to represent a sequence of grids.
    This data structure can be iterated to return all the possibile
    combinations of values per dictionary.
    """

    def __init__(self, grid):
        self.grid = grid

    def __iter__(self):
        for p in self.grid:
            # Sorted elements of the grid
            items = sorted(p.items())
            # Keys and values to combine
            keys, values = zip(*items)
            # Cartesian product of the values
            for v in product(*values):
                # Assign each value to a key
                params = dict(zip(keys, v))
                yield params
