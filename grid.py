from itertools import product

class Grid:
    """
    A list of dictionaries is used to represent a sequence of grids.
    This data structure can be iterated to return all the possibile
    combinations of values per dictionary.
    """

    def __init__(self, grid):
        self.grid   = grid
        self.length = 0
        for p in self.grid:
            items = sorted(p.items())
            keys, values = zip(*items)
            for v in product(*values):
                self.length += 1

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

    def __len__(self):
        return self.length
