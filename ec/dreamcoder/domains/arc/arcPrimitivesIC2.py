from dreamcoder.program import *
from dreamcoder.type import arrow, baseType, tint, tlist, t0, t1, t2, tboolean
from dreamcoder.task import Task
from dreamcoder.grammar import Grammar
from dreamcoder.program import Primitive

from scipy.ndimage import binary_dilation, label
from scipy import ndimage
from scipy.ndimage import measurements
from math import sin,cos,radians,copysign
import numpy as np
from collections import deque

from typing import Tuple, NewType, List, Callable, Dict, Type

tcolour = baseType("colour") # Any colour. We could use 1x1 grids for this, but by typing it we reduce the search space
Colour = NewType("Colour", int)

tpos = baseType("pos") # Position-only type
Position = NewType("Position", Tuple[int, int])

tsize = baseType("tsize")
Size = NewType("Size", Tuple[int, int])

tcount = baseType("count")
Count = NewType("Count", int)

typemap: Dict[Type, TypeConstructor] = {
    Colour: tcolour,
    Position: tpos,
    Size: tsize,
    Count: tcount
}
    
tgrid = baseType("grid") # Any grid. Position is always included
class Grid():
    """
    Represents an ARC grid, along with a position.
    For an unshifted grid (e.g. the input), the position is (0,0).
    Position holds the canonical index of the top-left grid pixel.
    This encompasses the Input, Grid and Object types from Alford.

    Instantiation with cutout=True will remove any background.
    """

    def __init__(self, grid: np.ndarray, position: Tuple[int, int]=(0, 0), cutout=False):
        self.position = position

        self.cutout = cutout
        if cutout:
            self.grid, (xpos, ypos) = Grid.cutout(grid)
            self.position[0] += xpos
            self.position[1] += ypos
        else:
            self.grid = grid

    @property
    def size(self) -> Size:
        return self.grid.shape

    @classmethod
    def cutout(grid: np.ndarray) -> np.ndarray:
        xr, yr = np.nonzero(grid)
        xpos, ypos = min(xr), min(yr)
        return grid[xpos:max(xr)+1, ypos:max(yr)+1], (xpos, ypos)
    
    def newgrid(self, grid: np.ndarray, offset=None, cutout=False) -> "Grid":
        """
        Return a new Grid object containing a new userprovided grid.
        The position is cloned from the parent.
        """
        position = self.position
        if offset:
            position[0] += offset[0]
            position[1] += offset[1]

        return Grid(grid, position, cutout)
    
    def count(self) -> int:
        """
        Returns the number of non-zero elements in the grid
        """
        return np.count_nonzero(self.grid)

    def __repr__(self):
        return f"Grid({self.grid.shape[0]}x{self.grid.shape[1]} at {self.position})"

def primitive_assert(boolean, message=None):
    """
    Raise a PrimitiveException if the condition is false.
    This stops execution on the current program and does not raise an error.
    """
    if not boolean:
        raise PrimitiveException(message)

# Define primitives now
def ic_invert(g: Grid) -> Grid:
    """
    In icecuber, this was filtercol with ID 0, we make it explicit
    Replaces all colours with zeros, and replaces the zeros with the first colour
    In our case, we replace it with the most common colour (arbitrary choice)
    """
    mode = np.argmax(np.bincount(g.grid)[1:])+1 # skip 0

    grid = np.zeros_like(g.grid)
    grid[g.grid == 0] = mode
    return g.newgrid(grid)

def ic_filtercol(g: Grid, c: Colour) -> Grid:
    "Remove all colours except the selected colour"
    primitive_assert(c != 0, "filtercol with 0 has no effect")

    grid = np.copy(g.grid) # Do we really need to copy? old one thrown away anyway
    grid[grid != c] = 0
    return g.newgrid(grid)

def ic_erasecol(g: Grid, c: Colour) -> Grid:
    "Remove a specified colour from the grid, keeping others intact"
    primitive_assert(c != 0, "erasecol with 0 has no effect")
    grid = np.copy(g.grid)
    grid[grid == c] = 0
    return g.newgrid(grid)

def setcol(g: Grid, c: Colour) -> Grid:
    """
    Set all pixels in the grid to the specified colour.
    This was named colShape in icecuber. 
    """
    primitive_assert(c != 0, "setcol with 0 has no effect")

    grid = np.zeros_like(g.grid)
    grid[np.nonzero(grid)] == c
    return g.newgrid(grid)

def ic_compress(g: Grid) -> Grid:
    raise NotImplementedError()

def getpos(g: Grid) -> Position:
    return g.position

def getsize(g: Grid) -> Size:
    return g.size

# TODO: Have a think about position/size/hull and how they fit in
# For now I skip getSize0, getHull, getHull0

def ic_toorigin(g: Grid) -> Grid:
    "Reset a grid's position to zero"
    return Grid(g.grid)

def ic_fill(g: Grid) -> Grid:
    raise NotImplementedError

def ic_interior(g: Grid) -> Grid:
    raise NotImplementedError

def ic_interior2(g: Grid) -> Grid:
    raise NotImplementedError

def ic_border(g: Grid) -> Grid:
    raise NotImplementedError

def ic_center(g: Grid) -> Grid:
    # TODO: Figure out why this is useful
    w,h = g.size
    
    newsize = ((w + 1) % 2 + 1, (h + 1) % 2 + 1)
    newgrid = np.ones(newsize)
    newpos = (
        g.position[0] + (newsize[0] - w) / 2,
        g.position[1] + (newsize[1] - h) / 2
    )

    return Grid(newgrid, newpos)

def topcol(g: Grid) -> Colour:
    """
    Returns the most common colour, excluding black.
    majCol in icecuber.
    """
    return np.argmax(np.bincount(g.grid)[1:])+1

## Rigid transformations

def rot90(g: Grid) -> Grid:
    return g.newgrid(np.rot90(g.grid))

def rot180(g: Grid) -> Grid:
    return g.newgrid(np.rot90(g.grid, k=2))

def rot270(g: Grid) -> Grid:
    return g.newgrid(np.rot90(g.grid, k=3))

def flipx(g: Grid) -> Grid:
    return g.newgrid(np.flip(g.grid, axis=0))

def flipy(g: Grid) -> Grid:
    return g.newgrid(np.flip(g.grid, axis=1))

def swapxy(g: Grid) -> Grid:
    return g.newgrid(g.grid.T)

# TODO: Transpose on other dimension

def mirrorHeuristic(g: Grid) -> Grid:
    # Performs a transformation based on centre of gravity
    # TODO
    raise NotImplementedError

####################################
# To/from Counts
####################################

def countPixels(g: Grid) -> Count:
    return np.count_nonzero(g.grid)

def countColours(g: Grid) -> Count:
    """Return the number of unique colours in the grid, excluding zero"""
    return np.count_nonzero(np.bincount(g.grid)[1:])

def countComponents(g: Grid) -> Count:
    """
    Returns the number of objects in the grid
    Matching the behaviour in icecuber:
    - colours are IGNORED (object can have multiple)
    - diagonals count as the same object (8-structural)
    """
    raise NotImplementedError

# TODO: Figure out how I want to do colours here - is this the best way?
def countToXY(c: Count, col: Colour) -> Grid:
    return Grid(np.zeros((c, c))+col)

def countToX(c: Count, col: Colour) -> Grid:
    return Grid(np.zeros((c, c))+col)

def countToY(c: Count, col: Colour) -> Grid:
    return Grid(np.zeros((c, c)+col))

####################################
# Smearing
####################################

def ic_smear_kernel(g: Grid, kernels: List[Tuple[int, int]]):
    """
    Smear a grid using a kernel.
    The kernel is a tuple of (x, y) offsets.
    """
    raise NotImplementedError

sk = { # smear kernels
    "R": {1, 0},
    "L": {-1, 0},
    "D": {0, 1},
    "U": {0, -1},
    "X": {1, 1},
    "Y": {-1, -1},
    "Z": {1, -1},
    "W": {-1, 1}
}

smear_functions = {
    "R": lambda g: ic_smear_kernel(g, [sk["R"]]),
    "L": lambda g: ic_smear_kernel(g, [sk["L"]]),
    "D": lambda g: ic_smear_kernel(g, [sk["D"]]),
    "U": lambda g: ic_smear_kernel(g, [sk["U"]]),
    "RL": lambda g: ic_smear_kernel(g, [sk["R"], sk["L"]]),
    "DU": lambda g: ic_smear_kernel(g, [sk["D"], sk["U"]]),
    "RLDU": lambda g: ic_smear_kernel(g, [sk["R"], sk["L"], sk["D"], sk["U"]]),
    "X": lambda g: ic_smear_kernel(g, [sk["X"]]),
    "Y": lambda g: ic_smear_kernel(g, [sk["Y"]]),
    "Z": lambda g: ic_smear_kernel(g, [sk["Z"]]),
    "W": lambda g: ic_smear_kernel(g, [sk["W"]]),
    "XY": lambda g: ic_smear_kernel(g, [sk["X"], sk["Y"]]),
    "ZW": lambda g: ic_smear_kernel(g, [sk["Z"], sk["W"]]),
    "XYZW": lambda g: ic_smear_kernel(g, [sk["X"], sk["Y"], sk["Z"], sk["W"]]),
    "RLDUXYZW": lambda g: ic_smear_kernel(g, [sk["R"], sk["L"], sk["D"], sk["U"], sk["X"], sk["Y"], sk["Z"], sk["W"]]),
}

####################################
# border/compression
####################################

def ic_makeborder(g: Grid) -> Grid:
    """
    Return a new grid which is the same as the input, but with a border of 1s around it.
    Only elements which are 0 in the original grid are set to 1 in the new grid.
    8-structuring element used to determine border positions.
    """

    binary_grid = g.grid > 0
    output_grid = np.zeros_like(g.grid)

    grown_binary_grid = binary_dilation(binary_grid, structure=np.ones((3, 3)))
    output_grid[grown_binary_grid & ~binary_grid] = 1

    return g.newgrid(output_grid)

def ic_makeborder2(g: Grid) -> Grid:
    pass

def ic_makeborder2_maj(g: Grid) -> Grid:
    pass

def ic_compress2(g: Grid) -> Grid:
    """Deletes any black rows/columns in the grid"""
    keep_rows = np.any(g.grid, axis=1)
    keep_cols = np.any(g.grid, axis=0)

    return g.newgrid(g.grid[keep_rows][:, keep_cols])

def ic_compress3(g: Grid) -> Grid:
    """
    Keep any rows/columns which differ in any way from the previous row/column
    The first row/column is always kept.
    """
    keep_rows = np.ones(g.grid.shape[0], dtype=bool)
    keep_cols = np.ones(g.grid.shape[1], dtype=bool)

    for row in range(1, g.grid.shape[0]):
        if np.all(g.grid[row] == g.grid[row-1]):
            keep_rows[row] = False

    for col in range(1, g.grid.shape[1]):
        if np.all(g.grid[:, col] == g.grid[:, col-1]):
            keep_cols[col] = False

    return g.newgrid(g.grid[keep_rows][:, keep_cols])

####################################
# Connect
####################################

def ic_connect_kernel(g: Grid, x: bool, y: bool) -> Grid:
    """
    Implements a generic connect (not a primitive)
    x and y control whether it is enabled in the horizontal and vertical directions
    
    Connect works as follows:
    Two cells in the same row/column are connected iff:
    - they are both non-zero
    - they have the same value c
    - there are only zeros in between them
    Connect fills in the zero values in-between with c.
    Note that vertical connection happens after horizontal connection and "overrides" it.

    This function is SLOW
    TODO: Verify this actually works because I'm not confident
    """
    ret = np.zeros_like(g.grid)

    if x:
        for row in range(g.grid.shape[0]):
            last = last_value = -1
            for col in range(g.grid.shape[1]):
                if g.grid[row, col]:
                    if g.grid[row, col] == last_value:
                        ret[row, last+1:col] = last_value
                    last_value = g.grid[row, col]
                    last = col
    
    if y:
        for col in range(g.grid.shape[1]):
            last = last_value = -1
            for row in range(g.grid.shape[0]):
                if g.grid[row, col]:
                    if g.grid[row, col] == last_value:
                        ret[last+1:row, col] = last_value
                    last_value = g.grid[row, col]
                    last = row

    return g.newgrid(ret)


ic_connectX = lambda g: ic_connect_kernel(g, True, False)
ic_connectY = lambda g: ic_connect_kernel(g, False, True)
ic_connectXY = lambda g: ic_connect_kernel(g, True, True)

####################################
# Spread colours
####################################

def ic_spread(g: Grid) -> Grid:
    """
    Loop through each cell in the grid.
    Place all indices and colours in a queue.

    Later, take indices out of the queue, and for each one:
    - Take the 4 structuring element, if any of the neighbours are not yet coloured, colour them with this colour.
   
    SLOW
    """

    # Create a queue of (x, y, colour) tuples
    queue = deque()

    # Create a new grid to store the output
    output_grid = np.copy(g.grid)

    # Reproduce the done array
    done = np.bool(g.grid)
    queue = deque([(row, col, g.grid[row, col]) for row, col in zip(*np.where(done))])

    # Loop through the queue
    while queue:
        # Get the next item from the queue
        row, col, colour = queue.popleft()

        # Add the neighbours to the queue
        for x, y in [(row-1, col), (row+1, col), (row, col-1), (row, col+1)]:
            if 0 <= x < g.grid.shape[0] and 0 <= y < g.grid.shape[1] and not done[x, y]:
                output_grid[x, y] = colour
                done[x, y] = 1
                queue.append((x, y, colour))

    return g.newgrid(output_grid)

def ic_spread_minor(g: Grid) -> Grid:
    """
    Same as spread, but ignoring the most common colour.
    SLOW
    """
    # Create a queue of (x, y, colour) tuples
    queue = deque()

    # Create a new grid to store the output
    output_grid = np.copy(g.grid)

    # Reproduce the done array
    done = np.bool(g.grid & (g.grid != np.bincount(g.grid.flatten()).argmax()))
    queue = deque([(row, col, g.grid[row, col]) for row, col in zip(*np.where(done))])

    # Loop through the queue
    while queue:
        # Get the next item from the queue
        row, col, colour = queue.popleft()

        # Add the neighbours to the queue
        for x, y in [(row-1, col), (row+1, col), (row, col-1), (row, col+1)]:
            if 0 <= x < g.grid.shape[0] and 0 <= y < g.grid.shape[1] and not done[x, y]:
                output_grid[x, y] = colour
                done[x, y] = 1
                queue.append((x, y, colour))

    return g.newgrid(output_grid)

####################################
# Cropping
####################################

def left_half(g: Grid) -> Grid:
    primitive_assert(g.size[1] > 1, "Grid is too small to crop")
    return g.newgrid(g.grid[:, :g.grid.shape[1]//2])

def right_half(g: Grid) -> Grid:
    """Note that left_half + right_half != identity, middle column may be lost"""
    primitive_assert(g.size[1] > 1, "Grid is too small to crop")
    return g.newgrid(g.grid[:, -g.grid.shape[1]//2:], 
                     offset=(0, g.grid.shape[1]//2 + g.grid.shape[1]%2))

def top_half(g: Grid) -> Grid:
    primitive_assert(g.size[0] > 1, "Grid is too small to crop")
    return g.newgrid(g.grid[:g.grid.shape[0]//2])

def bottom_half(g: Grid) -> Grid:
    """Note that top_half + bottom_half != identity, middle row may be lost"""
    primitive_assert(g.size[0] > 1, "Grid is too small to crop")
    return g.newgrid(g.grid[-g.grid.shape[0]//2:], 
                     offset=(g.grid.shape[0]//2 + g.grid.shape[0]%2, 0))

# TODO: 25 move primitives, seems kind of excessive...

####################################
# Binary operations
####################################
# In the original code, the 2nd argument is generated by a static shape detector
# We can implment the shape detector as a separate thing that yields Shape
# TODO: binary ops

def ic_embed(img: Grid, shape: Grid) -> Grid:
    """
    Embeds a grid into a larger shape defined by a 2nd argument (zero-padded).
    If the image is larger than the shape, it is cropped.
    """
    ret = np.zeros_like(shape.grid)
    
    xoffset = shape.position[0] - img.position[0]
    yoffset = shape.position[1] - img.position[1]

    xsize = min(img.grid.shape[0], shape.grid.shape[0] - xoffset)
    ysize = min(img.grid.shape[1], shape.grid.shape[1] - yoffset)

    ret[xoffset:xoffset+xsize, yoffset:yoffset+ysize] = img.grid[:xsize, :ysize]
    return shape.newgrid(ret)

def ic_wrap(line: Grid, area: Grid) -> Grid:
    primitive_assert(line.size > (0, 0))
    primitive_assert(area.size > (0, 0))

    zeros_like = np.zeros_like(area.grid)

####################################
# Split operations
####################################

def ic_cut(g: Grid) -> List[Grid]:
    """
    Cut up a grid.
    Returns a single colour, cut into at least 2 disjoint pieces. 
    Must touch at least 2 opposite sides, and smallest piece should be as big as possible
    """

    return NotImplementedError
    top_colour = topcol(g)
    ret_score = -1

    colour_mask = g.grid == top_colour
    done = np.zeros_like(colour_mask, dtype=np.bool)

def ic_splitcols(g: Grid) -> List[Grid]:
    """
    Split a grid into multiple grids, each with a single colour.
    """
    ret = []
    for colour in np.unique(g.grid):
        if colour:
            ret.append(g.newgrid(g.grid == colour))
    return ret

def ic_splitall(g: Grid) -> List[Grid]:
    """
    Find all objects using 4-structuring element
    Each colour is separated
    """
    colours = np.unique(g.grid)
    ret = []
    for colour in colours:
        if colour:
            objects = ndimage.find_objects(ndimage.label(g.grid == colour)[0])
            ret += [g.newgrid(g.grid[obj], offset=(obj[0].start, obj[1].start)) for obj in objects]
    return ret

def ic_splitcolumns(g: Grid) -> List[Grid]:
    """
    Return all the columns 
    """
    return [g.newgrid(g.grid[:, i:i+1], offset=(0, i)) for i in range(g.grid.shape[1])]

def ic_splitrows(g: Grid) -> List[Grid]:
    """
    Return all the rows 
    """
    return [g.newgrid(g.grid[i:i+1], offset=(i, 0)) for i in range(g.grid.shape[0])]

def ic_insidemarked(g: Grid) -> List[Grid]:
    raise NotImplementedError

def ic_gravity(g: Grid) -> Grid:
    """
    Gravity: drop all objects to the bottom of the grid
    """
    raise NotImplementedError

####################################
# Join
####################################

def _compress_count_key(o):
    """
    Returns the number of zeros in the grid once compressed.
    """
    comp = ic_compress(o)
    count = np.sum(comp.grid != 0)
    return o.grid.shape[0] * o.grid.shape[1] - count

def majority_col(o):
    """Returns the color that is most common in the grid."""
    return np.argmax(np.bincount(o.grid.ravel()))

def icecuber_fill(o):
    """Returns the grid with all holes filled."""

    a = o.grid
    majority_colour = majority_col(o)
    ret = np.full(o.grid.shape, majority_colour, dtype=np.int8)

    # Determine the majority color of the input grid

    # Create a binary mask of the input grid with True where pixels have a value of 0
    mask = a == 0

    # Create an empty binary mask for storing connected border pixels
    connected_border = np.zeros_like(mask, dtype=bool)

    # Set the border pixels in the connected_border mask
    connected_border[0, :] = mask[0, :]
    connected_border[-1, :] = mask[-1, :]
    connected_border[:, 0] = mask[:, 0]
    connected_border[:, -1] = mask[:, -1]

    # Iteratively perform binary dilation until there are no changes in the connected_border mask
    while True:
        new_connected_border = binary_dilation(connected_border, structure=np.ones((3, 3)), mask=mask)
        if np.array_equal(new_connected_border, connected_border):
            break
        connected_border = new_connected_border

    # Set the connected border pixels to 0 in the `ret` grid
    ret[connected_border] = 0

    # Return the filled grid
    return ret


def _icecuber_interior(o):
    filled = icecuber_fill(o)
    filled[o.grid != 0] = 0
    return filled
    
def _interior_count_key(o):
    interior = _icecuber_interior(o)
    count = np.sum(interior.grid != 0)
    return o.grid.shape[0] * o.grid.shape[1] - count

pickmax_functions = {
    "count":     lambda l: max(l, key=lambda o: np.sum(o != 0)),
    "neg_count": lambda l: min(l, key=lambda o: -np.sum(o != 0)),
    "size":      lambda l: max(l, key=lambda o: o.grid.shape[0] * o.grid.shape[1]),
    "neg_size":  lambda l: min(l, key=lambda o: -o.grid.shape[0] * o.grid.shape[1]),
    "cols":      lambda l: max(l, key=lambda o: len(np.unique(o.grid))),

    "components": lambda l: max(l, key=lambda o: len(countComponents(o))), # SLOW!

    "compress_count": lambda l: max(l, key=lambda o: _compress_count_key(o)),
    "neg_compress_count": lambda l: min(l, key=lambda o: -_compress_count_key(o)),
    "interior_count": lambda l: max(l, key=lambda o: _interior_count_key(o)),
    "neg_interior_count": lambda l: min(l, key=lambda o: -_interior_count_key(o)),

    # TODO: p.x/p.y pos/neg
    "x_pos": lambda l: max(l, key=lambda o: o.position[0]),
    "x_neg": lambda l: min(l, key=lambda o: o.position[0]),
    "y_pos": lambda l: max(l, key=lambda o: o.position[1]),
    "y_neg": lambda l: min(l, key=lambda o: o.position[1]),
}

def ic_pickunique(l: List[Grid]) -> Grid:
    """
    Given a list of grids, return the one which has a unique colour unused by any other grid
    If there are no such grids or more than one, terminate.
    """
    # TODO: think about this implementation a bit more, it might be wrong
    counts = np.zeros(10)
    uniques = [np.unique(g.grid) for g in l]
    for u in uniques:
        counts[u] += 1
    
    colour_mask = counts == 1
    ccount = np.sum(colour_mask)
    if not ccount:
        raise PrimitiveException("pickunique: no unique grids")
    
    for g, u in zip(l, uniques):
        if np.sum(colour_mask[u]) == ccount:
            return g
        
    raise PrimitiveException("pickunique: no unique grids (2)")

def ic_composegrowing(l: List[Grid]) -> Grid:
    """
    Compose a list of grids, to the minimum sized grid that holds all images
    This is position-aware, so two grids which do not overlap will generate one larger grid
    """

    xpos = min([g.position[0] for g in l])
    ypos = min([g.position[1] for g in l])
    xsize = max([g.position[0]+g.size[0] for g in l]) - xpos
    ysize = max([g.position[1]+g.size[1] for g in l]) - ypos

    newgrid = Grid(np.zeros((xsize, ysize)), position=(xpos, ypos))

    # Order grids in DESCENDING order of number of filled pixels
    # TODO: the original code seems like it is ascending, although this doesnt make much sense?
    sorted_list = sorted(l, key=lambda g: g.count())

    for g in sorted_list:
        xstart = g.position[0] - xpos
        ystart = g.position[1] - ypos

        # slice returns a view, assignment to this modifies the original array
        slice = newgrid[xstart:xstart+g.size[0], ystart:ystart+g.size[1]]

        mask = np.nonzero(g.grid)
        slice[mask] = g.grid[mask]

    return newgrid

# TODO: stackLine, myStack, pickMaxes, pickNotMaxes

#############################
# PRIMITIVE GENERATION
#############################

import inspect, typing

class PrimitiveBank:
    def __init__(self, typemap: Dict[Type, TypeConstructor], verbose=False):
        self.typemap = typemap
        self.primitives = {}
        self.verbose = verbose

    def cvt_type(self, anno):
        # Handle list types natively
        # These annotations have type typing._GenericAlias
        # __origin__ attr is list, __args__ attr is a tuple of constituent types
        if hasattr(anno, '__origin__'):
            if anno.__origin__ == list:
                # We recursively convert the constituent
                return tlist(self.cvt_type(anno.__args__[0]))
        
        if anno in self.typemap:
            return self.typemap[anno]
        
        raise TypeError(f"Annotation {anno} has no corresponding DreamCoder type")

    def register(self, f: Callable, name: str=None, type: TypeConstructor=None, autocurry: bool=True):
        if name is None:
            name = f.__name__
            if name == '<lambda>':
                raise ValueError('<lambda> passed to Primitive constructor, name must be specified')
            
        if type is None:
            # Generate a DreamCoder type signature for this function by inspection
            arrow_args = []
            fn_sig = inspect.signature(f)
            params = list(fn_sig.parameters.items())
            for arg, argtype in params:
                anno = argtype.annotation
                arrow_args.append(self.cvt_type(anno))

            dc_type = arrow(*arrow_args, self.cvt_type(fn_sig.return_annotation))
        else:
            dc_type = type

        # This function has more than 1 input and needs to be curried
        # We have special cases for 2/3 params because these are significantly faster
        if autocurry and len(params) > 1:
            if len(params) == 2:
                f = lambda x: lambda y: f(x, y)
            elif len(params) == 3:
                f = lambda x: lambda y: lambda z: f(x, y, z)
            else:
                def curry(f, n, args):
                    if n:
                        return lambda x: curry(f, n-1, args + [x])
                    return f(*args)
                f = curry(f, len(params), [])

        if self.verbose:
            print(f"Registered {name} with inferred type {dc_type}.")

        primitive = Primitive(name, dc_type, f)
        self.primitives[name] = primitive
    
    def registerMany(self, funcs: List[Callable]):
        for func in funcs:
            try:
                self.register(func)
            except KeyboardInterrupt:
                raise
            except:
                print(f"Error occured on {f}")
                raise

p = PrimitiveBank(typemap, verbose=True)

p.register(ic_filtercol)
p.register(ic_erasecol)
p.register(setcol)

# p.register(ic_compress)
p.register(getpos)
p.register(getsize)

#hull?
p.register(ic_toorigin)
p.register(ic_fill)
p.register(ic_interior)
p.register(ic_interior2)
p.register(ic_border)
p.register(ic_center)
p.register(topcol)

# rigid?
# count?
for dir, f in smear_functions.items():
    p.register(f, f"smear{dir}", arrow(tgrid, tgrid))

p.register(ic_makeborder)
p.register(ic_makeborder2)
p.register(ic_makeborder2_maj)

p.register(ic_compress2)
p.register(ic_compress3)

p.register(ic_connectX, "ic_connectX", arrow(tgrid, tgrid))
p.register(ic_connectY, "ic_connectY", arrow(tgrid, tgrid))
p.register(ic_connectXY, "ic_connectY", arrow(tgrid, tgrid))

# spreadcols?
p.registerMany([left_half, right_half, top_half, bottom_half])

# move?
p.registerMany([
    ic_embed, 
    ic_wrap
    # broadcast, repeat, mirror?
    ])

p.registerMany([
    ic_cut,
    ic_splitcols,
    ic_splitall,
    ic_splitcolumns,
    ic_splitrows,
    ic_insidemarked,
    # gravity?
])

for name, func in pickmax_functions:
    p.register(func, name, arrow(list(tgrid, tgrid)))

p.register(ic_pickunique)
p.register(ic_composegrowing)
# stackline, mystrack, pickmaxes, picknotmaxes?