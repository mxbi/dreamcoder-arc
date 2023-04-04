from dreamcoder.program import *
from dreamcoder.type import arrow, baseType, tint, tlist, t0, t1, t2, tboolean
from dreamcoder.task import Task
from dreamcoder.grammar import Grammar
from dreamcoder.program import Primitive

from scipy.ndimage import binary_dilation
from scipy.ndimage import measurements
from math import sin,cos,radians,copysign
import numpy as np
from collections import deque

MAX_GRID_LENGTH = 30
MAX_COLOR = 9
MAX_INT = 9

toriginal = baseType("original") # the original grid from input
tgrid = baseType("grid") # any modified grid
tobject = baseType("object")
tpixel = baseType("pixel")
tcolor = baseType("color")
tdir = baseType("dir")
tinput = baseType("input")
tposition = baseType("position")
tinvariant = baseType("invariant")
toutput = baseType("output")
tbase_bool = baseType('base_bool')

# raising an error if the program isn't good makes enumeration continue. This is
# a useful way of checking for correct inputs and such to speed up enumeration /
# catch mistakes.
def arc_assert(boolean, message=None):
    if not boolean: 
        # print('ValueError')
        raise ValueError(message)

class Grid():
    """
       Represents a grid.
    """
    def __init__(self, grid):
        assert type(grid) in (type(np.array([1])), type([1])), 'bad grid type: {}'.format(type(grid))
        self.grid = np.array(grid)
        self.position = (0, 0)
        self.input_grid = self.grid

    def __str__(self):
        return str(self.grid)

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        if hasattr(other, "grid"):
            return np.array_equal(self.grid, other.grid)
        else:
            return False

    def absolute_grid(self):
        g = np.zeros((30, 30))
        x, y = self.position
        w, h = self.grid.shape
        g[x : x + w, y : y + h] = self.grid
        return g


class Object(Grid):
    def __init__(self, grid, position=(0,0), input_grid=None, cutout=True):
        # input the grid with zeros. This turns it into a grid with the
        # background "cut out" and with the position evaluated accordingly

        if input_grid is not None:
            assert type(input_grid) == type(np.array([1])), 'bad grid type'

        if not cutout:
            super().__init__(grid)
            self.position = position
            self.input_grid = input_grid
            return

        def cutout(grid):
            x_range, y_range = np.nonzero(grid)

            position = min(x_range), min(y_range)
            cut = grid[min(x_range):max(x_range) + 1, min(y_range):max(y_range) + 1]
            return position, cut

        position2, cut = cutout(grid)
        # TODO add together?
        if position is None: position = position2
        super().__init__(cut)
        self.position = position
        self.input_grid = input_grid

    def __str__(self):
        return super().__str__() + ', ' + str(self.position)


class Pixel(Object):
    def __init__(self, grid, position, input_grid):
        assert grid.shape == (1,1), 'invalid pixel'
        super().__init__(grid, position, input_grid, cutout=False)


class Input():
    """
        Combines i/o examples into one input, so that we can synthesize a solution
        which looks at different examples at once

    """
    def __init__(self, input_grid, training_examples):
        assert type(input_grid) in (type(np.array([1])), type([1])), 'bad grid type'
        self.input_grid = Grid(input_grid)
        # all the examples
        self.grids = [(Grid(ex["input"]), Grid(ex["output"])) for ex in
                training_examples]

    def __str__(self):
        return "i: {}, grids={}".format(self.input_grid, self.grids)

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.input_grid == other.input_grid and self.grids == other.grids
        else:
            return False

# Grid primitives

# list primitives
def _get(l):
    def get(l, i):
        arc_assert(i >= 0 and i < len(l))
        return l[i]

    return lambda i: get(l, i)

def _get_first(l):
    return l[1]

def _get_last(l):
    return l[-1]

def _length(l):
    return len(l)

def _remove_head(l):
    return l[1:]

def _sortby(l):
    return lambda f: sorted(l, key=f)

def _map(f):
    return lambda l: [f(x) for x in l]

def _apply(l):
    return lambda arg: [f(arg) for f in l]

def _zip(list1): 
    return lambda list2: lambda f: list(map(lambda x,y: f(x)(y), list1, list2))

def _compare(f):
    return lambda a: lambda b: f(a)==f(b)

def _contains_color(o):
    return lambda c: c in o.grid

def _filter_list(l):
    return lambda f: [x for x in l if f(x)]

def _reverse(l):
    return l[::-1]

def _apply_colors(l_objects):
    return lambda l_colors: [_color_in(o)(c) for (o, c) in zip(l_objects, l_colors)]

def _find_in_list(obj_list):
    def find(obj_list, obj):
        for i, obj2 in enumerate(obj_list):
            if np.array_equal(obj.grid, obj2.grid):
                return i

        return None

    return lambda o: find(obj_list, o)

# def _y_split(g):
    # find row with all one color, return left and right
    # np.
    np.all(g.grid == g.grid[:, 0], axis = 0)


def _map_i_to_j(g):
    def map_i_to_j(g, i, j):
        m = np.copy(g.grid)
        m[m==i] = j
        return Grid(m)

    return lambda i: lambda j: map_i_to_j(g, i, j)

def _set_shape(g):
    def set_shape(g, w, h):
        g2 = np.zeros((w, h))
        g2[:len(g.grid), :len(g.grid[0])] = g.grid
        return Grid(g2)

    return lambda s: set_shape(g, s[0], s[1])

def _shape(g):
    return g.grid.shape

def _find_in_grid(g):
    def find(grid, obj):
        for i in range(len(grid) - len(obj) + 1):
            for j in range(len(grid[0]) - len(obj[0]) + 1):
                sub_grid = grid[i: i + len(obj), j : j + len(obj[0])]
                if np.array_equal(obj, sub_grid):
                    return (i, j)
        return None

    return lambda o: find(g.grid, o.grid)

def _filter_color(g):
    return lambda color: Grid(g.grid * (g.grid == color))

def _colors(g):
    # https://stackoverflow.com/questions/15637336/numpy-unique-with-order-preserved
    _, idx = np.unique(g.grid, return_index=True)
    colors = g.grid.flatten()[np.sort(idx)]
    colors = colors.tolist()
    if 0 in colors: colors.remove(0) # don't want black!
    return colors

def _object(g):
    return Object(g.grid, (0,0), g.input_grid, cutout=True)

def _pixel2(c):
    return Pixel(np.array([[c]]), position=(0, 0))

def _pixel(g):
    return lambda i: lambda j: Pixel(g.grid[i:i+1,j:j+1], (i, j))

def _overlay(g):
    return lambda g2: _stack_no_crop([g, g2])

def _list_of(g):
    return lambda g2: [g, g2]

def _color(g):
    # from https://stackoverflow.com/a/28736715/4383594
    # returns most common color besides black

    a = np.unique(g.grid, return_counts=True)
    a = zip(*a)
    a = sorted(a, key=lambda t: -t[1])
    a = [x[0] for x in a]
    if a[0] != 0 or len(a) == 1:
        return a[0]
    return a[1]

def _objects_by_color(g):
    l = [_filter_color(g)(color) for color in range(MAX_COLOR+1)]
    l = [_object(a) for a in l if np.sum(a.grid) != 0]
    return l
        
def _objects(g):
    connect_diagonals = False
    separate_colors = True
    out = _objects2(g)(connect_diagonals)(separate_colors)
    return out

def _rows(g):
    return [Object(g.grid[i:i+1], (i, 0), g.grid, cutout=False) for i in range(len(g.grid))]

def _columns(g):
    return [Object(g.grid[:, i:i+1], (0,i), g.grid, cutout=False) for i in range(len(g.grid))]
        
def _objects2(g):
    """
    This one has options for connecting diagonals and grouping colors together
    """
    def mask(grid1, grid2):
        grid3 = np.copy(grid1)
        grid3[grid2 == 0] = 0
        return grid3

    def objects_ignoring_colors(grid, connect_diagonals=False):
        objects = []

        # if included, this makes diagonally connected components one object.
        # https://stackoverflow.com/questions/46737409/finding-connected-components-in-a-pixel-array
        structure = np.ones((3,3)) if connect_diagonals else None

        #if items of the same color are separated...then different objects
        labelled_grid, num_features = measurements.label(grid, structure=structure)
        for object_i in range(1, num_features + 1):
            # array with 1 where that object is, 0 elsewhere
            object_mask = np.where(labelled_grid == object_i, 1, 0) 
            x_range, y_range = np.nonzero(object_mask)
            # position is top left corner of obj
            position = min(x_range), min(y_range)
            # get the original colors back
            original_object = mask(grid, object_mask)
            obj = Object(original_object, position, grid)
            objects.append(obj)


        return objects

    def objects(g, connect_diagonals=False, separate_colors=True):
        if separate_colors:
            separate_color_grids = [_filter_color(g)(color) 
                for color in np.unique(g.grid)]
            objects_per_color = [objects_ignoring_colors(
                g.grid, connect_diagonals)
                for g in separate_color_grids]
            objects = [obj for sublist in objects_per_color for obj in sublist]
        else:
            objects = objects_ignoring_colors(g.grid, connect_diagonals)

        objects = sorted(objects, key=lambda o: o.position)
        return objects
    
    return lambda connect_diagonals: lambda separate_colors: objects(g,
            connect_diagonals, separate_colors)



def _pixels(g):
    # TODO: always have relative positions?
    pixel_grid = [[Pixel(g.grid[i:i+1, j:j+1],
        position=(i + g.position[0], j + g.position[1]),
        input_grid = g.input_grid)
            for i in range(len(g.grid))]
            for j in range(len(g.grid[0]))]
    # flattens nested list into single list
    return [item for sublist in pixel_grid for item in sublist]

def _nonzero_pixels(g):
    return [p for p in _pixels(g) if p.grid[0][0] != 0]

def _hollow_objects(g):
    def hollow_objects(g):
        m = np.copy(g.grid)
        entriesToChange = []
        for i in range(1, len(m)-1):
            for j in range(1, len(m[i])-1):
                if(m[i][j]==m[i-1][j] and m[i][j]==m[i+1][j] and m[i][j]==m[i][j-1] and m[i][j]==m[i][j+1]):
                    entriesToChange.append([i, j])
        for entry in entriesToChange:
            m[entry[0]][entry[1]] = 0
        return Grid(m)
    return hollow_objects(g)

def _fill_line(g):
    def fill_line(g, background_color, line_color, color_to_add):
         m = np.copy(g.grid)
         for i in range(0, len(m)):
            for j in range(1, len(m[i])-1):
                if(m[i][j-1] == line_color and m[i][j] == background_color and m[i][j+1] == line_color):
                    m[i][j] = color_to_add
         return Grid(m)
    return lambda background_color: fill_line(g, background_color, 1, 2)
    #return lambda background_color: lambda line_color: lambda color_to_add: fill_line(g, background_color, line_color, color_to_add)

def _y_mirror(g):
    return Grid(np.flip(g.grid, axis=1))

def _x_mirror(g):
    return Grid(np.flip(g.grid, axis=0))

def _rotate_ccw(g):
    return Grid(np.rot90(g.grid))

def _rotate_cw(g):
    return Grid(np.rot90(g.grid, k=3))

def _combine_grids_horizontally(g1):
    def combine_grids_horizontally(g1, g2):
        m1 = np.copy(g1.grid)
        m2 = np.copy(g2.grid)
        m = np.column_stack([m1, m2])
        return Grid(m)
    return lambda g2: combine_grids_horizontally(g1, g2)
    
def _combine_grids_vertically(g1):
    def combine_grids_vertically(g1, g2):
        if g1.grid.shape[1] != g2.grid.shape[1]:
            raise PrimitiveException("combine_grids_vertically: grids must have same width")
        m1 = np.copy(g1.grid)
        m2 = np.copy(g2.grid)
        m = np.concatenate([m1, m2])
        return Grid(m)
    return lambda g2: combine_grids_vertically(g1, g2)

# color primitives

# input primitives
def _input(i): return i

def _input_grids(i): return [a for (a, b) in i.grids]

def _output_grids(i): return [b for (a, b) in i.grids]

def _find_corresponding(i):
    # object corresponding to object - working with lists of objects
    def location_in_input(inp, o):
        for i, input_example in enumerate(_input_grids(inp)):
            objects = _objects(input_example)
            location = _find_in_list(objects)(o)
            if location is not None:
                return i, location
        return None

    def find(inp, o):
        location = location_in_input(inp, o)
        arc_assert(location is not None)
        out = _get(_objects(_get(_output_grids(inp))(location[0])))(location[1])
        # make the position of the newly mapped equal to the input positions
        out.pos = o.pos
        return out

    return lambda o: find(i, o)

# list consolidation
def _vstack(l):
    # stacks list of grids atop each other based on dimensions
    # TODO won't work if they have different dimensions
    arc_assert(np.all([len(l[0].grid[0]) == len(x.grid[0]) for x in l]))
    l = [x.grid for x in l]
    return Grid(np.concatenate(l, axis=0))

def _hstack(l):
    # stacks list of grids horizontally based on dimensions
    # TODO won't work if they have different dimensions
    arc_assert(np.all([len(l[0].grid) == len(x.grid) for x in l]))
    return Grid(np.concatenate(l, axis=1))

def _positionless_stack(l):
    # doesn't use positions, just absolute object shape + overlay
    grid = np.zeros((30, 30))
    for g in l:
        # mask later additions
        grid += g.grid * (grid == 0)

    # get rid of extra shape -- needed?
    grid = Object(grid, (0, 0), grid)

    return Grid(grid.grid)

def _stack(l):
    # TODO absolute grid needed?
    # stacks based on positions atop each other, masking first to last
    grid = np.zeros((30, 30))
    for g in l:
        # mask later additions
        grid += g.absolute_grid() * (grid == 0)

    # get rid of extra shape -- needed?
    grid = Object(grid, (0, 0), grid)

    return Grid(grid.grid.astype("int"))

def _stack_no_crop(l):
    # stacks based on positions atop each other, masking first to last
    # assumes the grids are all the same size
    stackedgrid = np.zeros(shape=l[0].grid.shape)
    for g in l:
        # mask later additions
        commonsize = np.minimum(stackedgrid.shape, g.grid.shape)
        # stackedgrid += g.grid * (stackedgrid == 0)
        stackedgrid[:commonsize[0], :commonsize[1]] += g.grid[:commonsize[0], :commonsize[1]] * (stackedgrid[:commonsize[0], :commonsize[1]] == 0)

    return Grid(stackedgrid.astype("int"))



# boolean primitives
def _and(a): return lambda b: a and b
def _or(a): return lambda b: a or b
def _not(a): return not a
def _ite(a): return lambda b: lambda c: b if a else c 
def _eq(a): return lambda b: a == b

# object primitives
def _position(o): return o.position
def _x(o): return o.pos[0]
def _y(o): return o.pos[1]
def _size(o): return o.grid.size
def _area(o): return np.sum(o.grid != 0)

def _color_in(o):
    def color_in(o, c):
        grid = np.copy(o.grid)
        if np.sum(grid[grid != 0]) > 0:
            grid[grid != 0] = c
        else:
            grid[:] = c
        return Object(grid, o.position, o.input_grid)

    return lambda c: color_in(o, c)

def _color_in_grid(g):
    def color_in_grid(g, c):
        grid = g.grid
        grid[grid != 0] = c
        return Grid(grid)

        # return g

    return lambda c: color_in_grid(g, c)

def _flood_fill(g):
    def flood_fill(g, c):
        # grid = np.ones(shape=g.grid.shape).astype("int")*c
        # return Grid(grid)
        return g

    return lambda c: flood_fill(g, c)

# pixel primitives


# misc primitives
def _inflate(o):
    # currently does pixel-wise inflation. may want to generalize later
    def inflate(o, scale):
        # scale is 1, 2, 3, maybe 4
        x, y = o.grid.shape
        shape = (x*scale, y*scale)
        grid = np.zeros(shape)
        for i in range(len(o.grid)):
            for j in range(len(o.grid[0])):
                grid[scale * i : scale * (i + 1),
                     scale * j : scale * (j + 1)] = o.grid[i,j]

        return Grid(grid)

    return lambda inflate_factor: inflate(o, inflate_factor)

def _deflate(o):
    def deflate(o, scale):
        w, h = o.grid.shape
        arc_assert(w % scale == 0 and h % scale == 0)
        grid = np.zeros(w/scale, h/scale)
        i2 = 0
        for i in range(0, len(o), scale):
            j2 = 0
            for j in range(0, len(o[0]), scale):
                grid[i2][j2] = o.grid[i][j]
                # need to have contiguous squares to use this method
                arc_assert(np.all(o.grid[i:i+scale,j:j+scale] == o.grid[i][j]))
                j2 += 1
            i2 += 1

        return Object(grid, position=(0,0), innput_grid=o.input_grid)

    return lambda scale: deflate(o, scale)


def _top_half(g):
    return Grid(g.grid[0:int(len(g.grid)/2), :])

def _bottom_half(g):
    return Grid(g.grid[int(len(g.grid)/2):, :])

def _left_half(g):
    # Raise PrimitiveException if result of this will be empty
    if g.grid.shape[1] < 2:
        raise PrimitiveException("left_half: too small")
    return Grid(g.grid[:, 0:int(len(g.grid[0])/2)])

def _right_half(g):
    return Grid(g.grid[:, int(len(g.grid[0])/2):])

def _has_y_symmetry(g):
    return np.array_equal(np.flip(g.grid, axis=1), g.grid)

def _has_x_symmetry(g):
    return np.array_equal(np.flip(g.grid, axis=0), g.grid)

def _has_color(o):
    return lambda c: o.color == c

def _has_rotational_symmetry(g):
    return np.array_equal(_rotate_ccw(g).grid, g.grid)

## making the actual primitives

def _icecuber_filterCol(g, color):
    grid = np.copy(g.grid)
    grid[grid != color] = 0
    return Grid(grid)

_icecuber_filterCol_curry = lambda g: lambda color: _icecuber_filterCol(g, color)

def _icecuber_colShape(g, color):
    grid = np.copy(g.grid)
    grid[grid != 0] = color
    return Grid(grid)

_icecuber_colShape_curry = lambda g: lambda color: _icecuber_colShape(g, color)

# composeGrowing (list of images) → image
# Stack the list of images on top of each other, treating 0 as transparent.
# The image with the fewest non-zero pixels is at the top.
def _icecuber_composeGrowing(l):
    # Sort by number of non-zero pixels
    l = sorted(l, key=lambda g: np.sum(g.grid != 0), reverse=True)

    # assumes the grids are all the same size
    stackedgrid = l[0]
    for g in l[1:]:
        # mask later additions
        mask = g.grid != 0
        # Filter out-of-bounds
        mask = np.logical_and(mask, np.indices(mask.shape)[0] < stackedgrid.grid.shape[0])
        mask = np.logical_and(mask, np.indices(mask.shape)[1] < stackedgrid.grid.shape[1])
        stackedgrid.grid[mask] = g.grid[mask]

    return Grid(stackedgrid)

# compress (image) → image
# Extract minimal sub-image containing all non-zero pixels.
def _icecuber_compress(g):
    # Thanks to https://stackoverflow.com/a/39466129/5128131
    if np.max(g.grid) == 0:
        raise PrimitiveException("compress: image is empty")
    true_points = np.argwhere(g.grid)
    top_left = true_points.min(axis=0)
    bottom_right = true_points.max(axis=0)
    return Grid(g.grid[top_left[0]:bottom_right[0]+1, top_left[1]:bottom_right[1]+1])

# Cut (image) → list of images
# Tries to figure out a background color and splits the remaining pixels into
# corner connected groups.
def _objects(g):
    connect_diagonals = False
    separate_colors = True
    out = _objects2(g)(connect_diagonals)(separate_colors)
    return out

def _objects2(g):
    """
    This one has options for connecting diagonals and grouping colors together
    """
    def mask(grid1, grid2):
        grid3 = np.copy(grid1)
        grid3[grid2 == 0] = 0
        return grid3

    def objects_ignoring_colors(grid, connect_diagonals=False):
        objects = []

        # if included, this makes diagonally connected components one object.
        # https://stackoverflow.com/questions/46737409/finding-connected-components-in-a-pixel-array
        structure = np.ones((3,3)) if connect_diagonals else None

        #if items of the same color are separated...then different objects
        labelled_grid, num_features = measurements.label(grid, structure=structure)
        for object_i in range(1, num_features + 1):
            # array with 1 where that object is, 0 elsewhere
            object_mask = np.where(labelled_grid == object_i, 1, 0) 
            x_range, y_range = np.nonzero(object_mask)
            # position is top left corner of obj
            position = min(x_range), min(y_range)
            # get the original colors back
            original_object = mask(grid, object_mask)
            obj = Object(original_object, position, grid)
            objects.append(obj)


        return objects

    def objects(g, connect_diagonals=False, separate_colors=True):
        if separate_colors:
            separate_color_grids = [_filter_color(g)(color) 
                for color in np.unique(g.grid)]
            objects_per_color = [objects_ignoring_colors(
                g.grid, connect_diagonals)
                for g in separate_color_grids]
            objects = [obj for sublist in objects_per_color for obj in sublist]
        else:
            objects = objects_ignoring_colors(g.grid, connect_diagonals)

        objects = sorted(objects, key=lambda o: o.position)
        return objects
    
    return lambda connect_diagonals: lambda separate_colors: objects(g,
            connect_diagonals, separate_colors)

colors = {
    'color'+str(i): Primitive("color"+str(i), tcolor, i) for i in range(0, MAX_COLOR + 1)
    }

grid_primitives = {
    # "map_i_to_j": Primitive("map_i_to_j", arrow(tgrid, tcolor, tcolor, tgrid), _map_i_to_j),
    # "find_in_list": Primitive("find_in_list", arrow(tlist(tgrid), tint), _find_in_list),
    # "find_in_grid": Primitive("find_in_grid", arrow(tgrid, tgrid, tposition), _find_in_grid),
    # "filter_color": Primitive("filter_color", arrow(tgrid, tcolor, tgrid), _filter_color),
    # "colors": Primitive("colors", arrow(tgrid, tlist(tcolor)), _colors),
    # # "color": Primitive("color", arrow(tobject, tcolor), _color),
    # "color": Primitive("color", arrow(tgrid, tcolor), _color),
    # "objects": Primitive("objects", arrow(toriginal, tlist(tobject)), _objects),
    # "objects_by_color": Primitive("objects_by_color", arrow(tgrid, tlist(tgrid)), _objects_by_color),
    # # "object": Primitive("object", arrow(toriginal, tgrid), _object),
    # "object": Primitive("object", arrow(tgrid, tgrid), _object),
    # "objects2": Primitive("objects2", arrow(tgrid, tbase_bool, tbase_bool, tlist(tgrid)), _objects2),
    # "objects3": Primitive("objects3", arrow(tgrid, tlist(tgrid)), lambda g: _objects2(g)(True)(True)),
    # "pixel2": Primitive("pixel2", arrow(tcolor, tgrid), _pixel2),
    # "pixel": Primitive("pixel", arrow(tint, tint, tgrid), _pixel),
    # "list_of": Primitive("list_of", arrow(tgrid, tgrid, tlist(tgrid)), _list_of),
    # "pixels": Primitive("pixels", arrow(tgrid, tlist(tgrid)), _pixels),
    # "set_shape": Primitive("set_shape", arrow(tgrid, tposition, tgrid), _set_shape),
    # "shape": Primitive("shape", arrow(tgrid, tposition), _shape),
    # "y_mirror": Primitive("y_mirror", arrow(tgrid, tgrid), _y_mirror),
    "x_mirror": Primitive("x_mirror", arrow(tgrid, tgrid), _x_mirror),
    # "rotate_ccw": Primitive("rotate_ccw", arrow(tgrid, tgrid), _rotate_ccw),
    "rotate_cw": Primitive("rotate_cw", arrow(tgrid, tgrid), _rotate_cw),
    "left_half": Primitive("left_half", arrow(tgrid, tgrid), _left_half),
    "overlay": Primitive("overlay", arrow(tgrid, tgrid, tgrid), _overlay),
    "combine_grids_vertically": Primitive("combine_grids_vertically", arrow(tgrid, tgrid, tgrid), _combine_grids_vertically),

    "cut": Primitive("cut", arrow(tgrid, tlist(tgrid)), _objects),
    "filterCol": Primitive("filterCol", arrow(tgrid, tcolor, tgrid), _icecuber_filterCol_curry),
    "colShape": Primitive("colShape", arrow(tgrid, tcolor, tgrid), _icecuber_colShape_curry),
    # "composeGrowing": Primitive("composeGrowing", arrow(tlist(tgrid), tgrid), _icecuber_composeGrowing),
    "compress": Primitive("compress", arrow(tgrid, tgrid), _icecuber_compress),
    "mklist": Primitive("mklist", arrow(tgrid, tgrid, tlist(tgrid)), lambda x: lambda y: [x, y]),
    "mkcons": Primitive("mkcons", arrow(tgrid, tlist(tgrid), tlist(tgrid)), lambda x: lambda y: [x] + y),
    # Rigid replaced by rotate & mirror
    
    # "has_x_symmetry": Primitive("has_x_symmetry", arrow(tgrid, tboolean), _has_x_symmetry),
    # "has_y_symmetry": Primitive("has_y_symmetry", arrow(tgrid, tboolean), _has_y_symmetry),
    # "has_rotational_symmetry": Primitive("has_rotational_symmetry", arrow(tgrid, tboolean), _has_rotational_symmetry),
    }

# Icecuber analysis:
## An image has position p(x, y) and size sz(w, h).
## Image_ is an Image&


def compress_count_key(o):
    """
    Returns the number of zeros in the grid once compressed.
    """
    comp = _icecuber_compress(o)
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
    return Grid(filled)
    
def interior_count_key(o):
    interior = _icecuber_interior(o)
    count = np.sum(interior.grid != 0)
    return o.grid.shape[0] * o.grid.shape[1] - count

pickmax_functions = {
    "count":     lambda l: max(l, key=lambda o: np.sum(o != 0)),
    "neg_count": lambda l: min(l, key=lambda o: -np.sum(o != 0)),
    "size":      lambda l: max(l, key=lambda o: o.grid.shape[0] * o.grid.shape[1]),
    "neg_size":  lambda l: min(l, key=lambda o: -o.grid.shape[0] * o.grid.shape[1]),
    "cols":      lambda l: max(l, key=lambda o: len(np.unique(o.grid))),

    "components": lambda l: max(l, key=lambda o: len(_objects(o))), # SLOW!

    "compress_count": lambda l: max(l, key=lambda o: compress_count_key(o)),
    "neg_compress_count": lambda l: min(l, key=lambda o: -compress_count_key(o)),
    "interior_count": lambda l: max(l, key=lambda o: interior_count_key(o)),
    "neg_interior_count": lambda l: min(l, key=lambda o: -interior_count_key(o)),

    # TODO: p.x/p.y pos/neg
    "x_pos": lambda l: max(l, key=lambda o: o.position[0]),
    "x_neg": lambda l: min(l, key=lambda o: o.position[0]),
    "y_pos": lambda l: max(l, key=lambda o: o.position[1]),
    "y_neg": lambda l: min(l, key=lambda o: o.position[1]),
}

def wrap_pickmax(f):
    """Check that the list is not empty before applying pickmax"""
    def f_(l):
        if not len(l):
            raise PrimitiveException("pickmax: empty list")
        return f(l)
    return f_

# TODO implement these
pickmax_primitives = {
    f"pickmax_{key}": Primitive(f"pickmax_{key}", arrow(tlist(tgrid), tgrid), wrap_pickmax(func))
    for key, func in pickmax_functions.items()
}

def _input(i): return i

primitive_dict = {
        # "input": Primitive("input", arrow(tinput, tgrid), _input),
        **grid_primitives,
        **pickmax_primitives,
        **colors,
        }

primitives = list(primitive_dict.values())

def generate_ocaml_primitives():
    lines = [p.ocaml_string() + '\n' for p in primitive_dict.values()]

    with open("solvers/program.ml", "r") as f:
        contents = f.readlines()

    start_ix = min([i for i in range(len(contents)) if contents[i][0:7] == '(* AUTO'])
    end_ix = min([i for i in range(len(contents)) if contents[i][0:11] == '(* END AUTO'])
    contents = contents[0:start_ix+1] + lines + contents[end_ix:]

    with open("solvers/program.ml", "w") as f:
        f.write(''.join(contents))

