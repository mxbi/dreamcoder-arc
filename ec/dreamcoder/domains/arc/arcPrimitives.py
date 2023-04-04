from dreamcoder.program import *
from dreamcoder.type import arrow, baseType, tint, tlist, t0, t1, t2, tboolean
from dreamcoder.task import Task
from dreamcoder.grammar import Grammar
from dreamcoder.program import Primitive
#from dreamcoder.domains.arc.arcPixelwisePrimitives import _stack_xor, _stack_and, _apply_function, _complement, _return_subgrids, _grid_split

from scipy.ndimage import measurements
from math import sin,cos,radians,copysign
from operator import itemgetter
import numpy as np

MAX_GRID_LENGTH = 30
MAX_COLOR = 9
MAX_INT = 9

toriginal = baseType("original") # the original grid from input
tgrid = baseType("grid") # any modified grid
tobject = baseType("object")
tcolor = baseType("color")
tdir = baseType("dir")
tinput = baseType("input")
tposition = baseType("position")
tinvariant = baseType("invariant")
toutput = baseType("output")
tbase_bool = baseType('base_bool')

def arc_assert(boolean, message=None):
    """
    For sanity checking. The assertion fails silently and
    enumeration will continue, but whatever program caused the assertion is
    immediately discarded as nonviable. This is useful for checking correct
    inputs, not creating a massive grid that uses up memory by kronecker super
    large grids, and so on.
    """
    if not boolean:
        # print('ValueError')
        raise PrimitiveException(message)


class Grid():
    """
       Represents a grid.

       Position is (y, x) where y axis increases downward from 0 at the top.
    """
    def __init__(self, grid, position=(0, 0), cutout=False):
        assert type(grid) in (type(np.array([1])), type([1])), 'bad grid type: {}'.format(type(grid))

        if cutout:
            # crop the background, so that the grid is focused on nonzero
            # pixels. Example of this is task 30.
            y_range, x_range = np.nonzero(grid)
            dy, dx = min(y_range), min(x_range)
            grid = grid[min(y_range):max(y_range) + 1,
                        min(x_range):max(x_range) + 1]
            position = position[0] + dy, position[1] + dx

        max_dim = 60
        min_dim = -60 # maybe you want to start off the grid? idk
        arc_assert(min(grid.shape) > 0)
        arc_assert(max(grid.shape) < max_dim)
        arc_assert(min(position) >= min_dim)
        arc_assert(max(position) <= max_dim)

        self.grid = np.array(grid)
        assert len(grid.shape) == 2, 'bad grid shape: {}'.format(grid)
        self.position = position


    def __str__(self):
        if self.position != (0, 0):
            return str(self.grid) + ', ' + str(self.position)
        else:
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
        y, x = self.position
        h, w = self.grid.shape
        g[y : y + h, x : x + w] = self.grid
        return g

#HAD TO PUT THIS HERE BC ARCPIXELWISEPRIMITIVES USES GRID CLASS
from dreamcoder.domains.arc.arcPixelwisePrimitives import _stack_xor, _stack_and, _complement, _return_subgrids, _grid_split

class Input():
    """
        Combines i/o examples into one input, so that we can synthesize a solution
        which looks at different examples at once

    """
    def __init__(self, input_grid, training_examples):
        assert type(input_grid) in (type(np.array([1])), type([1])), 'bad grid type: {}'.format(type(input_grid))
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

    def __hash__(self):
        # https://stackoverflow.com/questions/16589791/most-efficient-property-to-hash-for-numpy-array
        # don't really need to be efficient, but this was easiest.
        # I was getting an error in dreamcoder/dreaming.py line 46 for
        # unhashable type when trying to use biasOptimal param.
        return hash(self.input_grid.grid.tostring())


# list primitives
def _get(l):
    def get(l, i):
        arc_assert(i >= 0 and i < len(l))
        return l[i]

    return lambda i: get(l, i)

def _get_first(l):
    arc_assert(len(l),'list length has to be at least 1')
    return l[0]

def _get_last(l):
    return l[-1]

def _length(l):
    return len(l)

def _sort_incr(l):
    return lambda f: sorted(l, key=f)

def _sort_decr(l):
    return lambda f: sorted(l, key=lambda o: -f(o))

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
    def filter_list(l, f):
        out = [x for x in l if f(x)]
        # might have to change this eventually.
        arc_assert(len(out) >= 1)
        return out

    return lambda f: filter_list(l, f)

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
    # np.all(g.grid == g.grid[:, 0], axis = 0)


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

def _num_colors(g):
    return len(_colors(g))

def _object(g):
    return Grid(g.grid, (0,0), cutout=True)


# moves by changing position parameter, not by changing array.
def _move_down2(obj):
    y, x = obj.position
    return Grid(obj.grid, position=(y+1, x))


def _move_down(g):
    # o.grid = np.roll(o.grid, 1, axis=0)
    # return Grid(o.grid)

    o = _get(_objects(g))(0)
    newg = Grid(g.grid)
    newg.grid[o.grid==1]=0 # remove object from old grid
    o.grid = np.roll(o.grid, 1, axis=0) # move down object
    return _overlay(newg)(o) # add object back to grid

def _pixel2(c):
    return Grid(np.array([[c]]), position=(0, 0))

def _pixel(g):
    return lambda i: lambda j: Grid(g.grid[i:i+1,j:j+1], (i, j))

def _overlay(g):
    return lambda g2: _stack_overlay([g, g2])

def _list_of(g):
    return lambda g2: [g, g2]

def _list_of_one(g):
    return [g]

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

def _object_frequency(grid):
    def object_frequency(obj, grid):
        return np.sum(obj == elem in _objects(grid))
    return lambda obj: object_frequency(obj, grid)

def _max_object_frequency(g):
    def max_object_frequency(g):
        dictionary = _object_frequency_list(g)
        max_obj = max(dictionary, key = dictionary.get)
        max_object = max_obj.replace('[','').replace(']','')
        formatted = max_object.splitlines()
        outArr = []
        for item in formatted:
            arrentry = np.fromstring(item, dtype=np.int, sep = ' ')
            outArr.append(arrentry)
        return(Grid(outArr))

    return max_object_frequency(g)

def _object_frequency_list(g):
    def object_frequency_list(g):
        connect_diagonals = True
        separate_colors = True
        out = _objects2(g)(connect_diagonals)(separate_colors)
        frequency = dict()
        for obj in out:
            if str(obj.grid) not in frequency:
                frequency[str(obj.grid)] = 0
            frequency[str(obj.grid)] +=1
        return frequency
    return object_frequency_list(g)

def _min_object_frequency(grid):
    def min_object_frequency(g):
        dictionary = _object_frequency_list(g)
        min_obj = min(dictionary, key = dictionary.get)
        min_object = min_obj.replace('[','').replace(']','')
        formatted = min_object.splitlines()
        outArr = []
        for item in formatted:
            arrentry = np.fromstring(item, dtype=np.int, sep = ' ')
            outArr.append(arrentry)
        return(Grid(outArr))

    return min_object_frequency(g)


def _objects(g):
    connect_diagonals = False
    separate_colors = True
    # don't crop. I think this is what Anshula was using
    out = _objects_no_crop(g)(connect_diagonals)(separate_colors)
    return out

def _rows(g):
    return [Grid(g.grid[i:i+1], (i, 0), cutout=False) for i in range(len(g.grid))]

def _columns(g):
    return [Grid(g.grid[:, i:i+1], (0,i), cutout=False) for i in range(len(g.grid))]

def _objects_no_crop(g):
    """
    This one doesn't crop objects.
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
            y_range, x_range = np.nonzero(object_mask)
            # position is top left corner of obj
            position = min(y_range), min(x_range)
            # get the original colors back
            original_object = mask(grid, object_mask)
            obj = Grid(original_object, position, cutout=False)
            objects.append(obj)


        # print('objects: {}'.format(objects))
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
        arc_assert(len(objects) >= 1)
        return objects

    return lambda connect_diagonals: lambda separate_colors: objects(g,
            connect_diagonals, separate_colors)


def _objects2(g):
    """
    This one has options for connecting diagonals and grouping colors together.
    Does crop objects
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
            # get the original colors back
            original_object = mask(grid, object_mask)
            # when cutting out, we automatically set the position, so only need to add original position
            obj = Grid(original_object, position=g.position, cutout=True)
            objects.append(obj)


        # print('objects: {}'.format(objects))
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
        arc_assert(len(objects) >= 1)
        return objects

    return lambda connect_diagonals: lambda separate_colors: objects(g,
            connect_diagonals, separate_colors)




def _pixels(g):
    # TODO: always have relative positions?
    pixel_grid = [[Grid(g.grid[i:i+1, j:j+1],
            position=(i + g.position[0], j + g.position[1]))
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

def _reflect_down(g):
    return _combine_grids_vertically(g)(_x_mirror(g))

def _crop_down(g):
    """ crop out all the zero rows at the bottom of the grid """
    newg = np.copy(g.grid)
    while np.all( newg[-1,:]==0 ):
        newg = newg[:-1,:]
    return Grid(newg)

def _rotate_ccw(g):
    return Grid(np.rot90(g.grid))

def _rotate_cw(g):
    return Grid(np.rot90(g.grid,k=3))

def _combine_grids_horizontally(g1):
    def combine_grids_horizontally(g1, g2):
        m1 = np.copy(g1.grid)
        m2 = np.copy(g2.grid)
        m = np.column_stack([m1, m2])
        return Grid(m)
    return lambda g2: combine_grids_horizontally(g1, g2)

def _combine_grids_vertically(g1):
    def combine_grids_vertically(g1, g2):
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

def _stack_overlay(l):
    # if there are positions, uses those.
    min_y = min([o.position[0] for o in l])
    max_y = max([o.position[0] + o.grid.shape[0] for o in l])
    min_x = min([o.position[1] for o in l])
    max_x = max([o.position[1] + o.grid.shape[1] for o in l])

    grid = np.zeros((max_y - min_y, max_x - min_x)).astype('int')

    for g in l:
        y, x = g.grid.shape
        py, px = g.position
        py, px = py - min_y, px - min_x
        grid[py:py+y, px:px+x] += (g.grid * (grid[py:py+y, px:px+x] == 0)).astype(grid.dtype)

    grid = Grid(grid, (min_y, min_x), cutout=False)
    return grid


def _positioned_stack(l):
    grid = np.zeros((60, 60))
    place_into_grid(grid, l)
    return Grid(grid, (0, 0), cutout=True)


def _stack(l):
    # doesn't use positions
    # reverse list so that first item shows up on top
    l = l[::-1]
    max_y = max([o.grid.shape[0] for o in l])
    max_x = max([o.grid.shape[1] for o in l])
    grid = np.zeros((max_y, max_x))

    for g in l:
        # mask later additions
        grid[:g.shape[0], :g.shape[1]] = g.grid

    # get rid of extra shape
    grid = Grid(grid, (0, 0), cutout=True)
    return grid


# boolean primitives
def _and(a): return lambda b: a and b
def _or(a): return lambda b: a or b
def _not(a): return not a
def _ite(a): return lambda b: lambda c: b if a else c
def _eq(a): return lambda b: a == b

# object primitives
def _position(o): return o.position
def _y(o): return o.pos[0]
def _x(o): return o.pos[1]
def _size(o): return o.grid.size
def _area(o): return np.sum(o.grid != 0)

def _color_in(o):
    def color_in(o, c):
        grid = np.copy(o.grid)
        # if grid is blank, fill it all in. otherwise, just fill nonblank cells.
        if np.sum(grid[grid != 0]) > 0:
            grid[grid != 0] = c
        else:
            grid[:] = c
        return Grid(grid, o.position)

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
def _kronecker(o1):
    def kron(o1, o2):
        max_dim = 100
        arc_assert(max(new_shape) < max_dim)
    return lambda o2: Grid(np.kron(o1.grid != 0, o2.grid))

def _inflate(o):
    # currently does pixel-wise inflation. may want to generalize later
    def inflate(o, scale):
        arc_assert(scale <= 10)
        # scale is 1, 2, 3, maybe 4
        y, x = o.grid.shape
        shape = (y*scale, x*scale)
        grid = np.zeros(shape)
        for i in range(len(o.grid)):
            for j in range(len(o.grid[0])):
                grid[scale * i : scale * (i + 1),
                     scale * j : scale * (j + 1)] = o.grid[i,j]

        return Grid(grid)

    return lambda inflate_factor: inflate(o, inflate_factor)

def _deflate_detect_scale(o):
    # scale must be a factor of length and width
    # want largest one for which all of the new pixels are a single color.

    # if scale works, will return deflated obj. else returns false
    def try_scale(scale, grid):
        w, h = grid.shape
        if w % scale != 0 or h % scale != 0:
            return False
        grid2 = np.zeros((int(w/scale), int(h/scale)))
        i2 = 0
        for i in range(0, len(grid), scale):
            j2 = 0
            for j in range(0, len(grid[0]), scale):
                grid2[i2][j2] = grid[i][j]
                # need to have contiguous squares to use this method
                if not np.all(grid[i:i+scale, j:j+scale] == grid[i][j]):
                    return False
                j2 += 1
            i2 += 1

        return Grid(grid2, position=(0,0))

    for scale in range(min(o.grid.shape), 1, -1):
        out = try_scale(scale, o.grid)
        if out is not False:
            # worked. return it
            return out

    # couldn't find anything
    arc_assert(False)

def _deflate(o):
    def deflate(o, scale):
        h, w = o.grid.shape
        arc_assert(h % scale == 0 and w % scale == 0)
        grid = np.zeros((int(h/scale), int(w/scale)))
        i2 = 0
        for i in range(0, len(o.grid), scale):
            j2 = 0
            for j in range(0, len(o.grid[0]), scale):
                grid[i2][j2] = o.grid[i][j]
                # need to have contiguous squares to use this method
                arc_assert(np.all(o.grid[i:i+scale,j:j+scale] == o.grid[i][j]))
                j2 += 1
            i2 += 1

        return Grid(grid, position=(0,0))

    return lambda scale: deflate(o, scale)


def _top_half(g):
    return Grid(g.grid[0:int(len(g.grid)/2), :])

def _bottom_half(g):
    return Grid(g.grid[int(len(g.grid)/2):, :])

def _left_half(g):
    return Grid(g.grid[:, 0:int(len(g.grid[0])/2)])

def _right_half(g):
    return Grid(g.grid[:, int(len(g.grid[0])/2):])

def _has_y_symmetry(g):
    return np.array_equal(np.flip(g.grid, axis=1), g.grid)

def _has_x_symmetry(g):
    return np.array_equal(np.flip(g.grid, axis=0), g.grid)

def _has_color(o):
    return lambda c: o.color == c

def _group_objects_by_color(g):
    """
    Returns list with objects of same colors
    e.g. [obs with color 1, obs with color 2, obs with color 3...]
    but not necessarily in that order
    """
    obs = _objects(g)

    # [func comparing color of o1...func comparing color of o6]
    funcs = _map(_compare(_color))(obs)

    # [[obj 1 and ob 5], [obj 2 and ob 3]...]
    return _map  ( _filter_list(obs) ) (funcs)

def _has_rotational_symmetry(g):
    return np.array_equal(_rotate_ccw(g).grid, g.grid)

def _draw_connecting_line(g):
    # takes in the grid, the starting object, and list of 2 objects to connect
    # draws each line on a separate grid, then returns the grid with the stack
    def draw_connecting_line(g, l):
        grids = []
        o1 = l[0]
        o2 = l[1]

        gridx,gridy = g.grid.shape
        line = np.zeros(shape=(gridx,gridy)).astype("int")

        # draw line between two positions
        startx, starty = o1.position
        endx, endy = o2.position

        sign = lambda a: 1 if a>0 else -1 if a<0 else 0

        x_step = sign(endx-startx) # normalize it, just figure out if its 1,0,-1
        y_step = sign(endy-starty) # normalize it, just figure out if its 1,0,-1

        # as a default, make the line the same color as the first object
        c = _color(o1)

        x,y=startx, starty
        try: # you might end up off the grid if the steps don't line up neatly
            line[startx][starty]=c
            while not (x==endx and y==endy):
                x += x_step
                y += y_step
                line[x][y]=c
        except:
            raise Exception("There's no straight line that can cnonect the two points")

        grids.append(Grid(line))

        return _stack_overlay(grids)

    return lambda l: draw_connecting_line(g,l)

def _draw_line(g):
    """
    Draws line on current grid
    from position of object o
    in direction d
    """

    def draw_line(g, d):

        o = _get(_objects(g))(0)

        gridx,gridy = g.grid.shape
        # line = np.zeros(shape=(gridx,gridy)).astype("int")
        grid = np.copy(g.grid)

        # dir can be 0 45 90 135 180 ... 315 (degrees)
        # but we convert to radians
        # and then add 90 so it will be on the np array what we expect
        direction=radians(d)+90

        x,y=o.position
        while x < gridx and x >= 0 and y < gridy and y >= 0:
            grid[x][y]=1
            x,y=int(round(x+cos(direction))), int(round(y-sin(direction)))

        # go in both directions
        bothways=False
        if bothways:
            direction=radians(d+180)

            x,y=o.position
            while x < gridx and x >= 0 and y < gridy and y >= 0:
                grid[x][y]=1
                x,y=int(round(x+cos(direction))), int(round(y-sin(direction)))


        return Grid(grid)

    return lambda d: draw_line(g,d)

def _equals_exact(obj1):
    def equals_exact(obj1, obj2):
        return np.array_equal(obj1.grid, obj2.grid)

    return lambda obj2: equals_exact(obj1, obj2)


def map_multiple(grid, old_colors, new_colors):
    new_grid = np.copy(grid)
    for old_color, new_color in zip(old_colors, new_colors):
        new_grid[grid == old_color] = new_color
    return new_grid


def _color_transform(obj):
    # sort colors by frequency, position and map accordingly
    counts = np.unique(obj.grid, return_counts=True)
    # list of (element, frequency) tuples
    counts = list(zip(*counts))
    # given color, returns smallest position where color is found
    f = lambda c: (lambda a: (a[0][0], a[1][0]))(np.where(obj.grid == c))
    # add the smallest position for that color, to break frequency ties
    counts = [(color, count) + f(color) for (color, count) in counts]
    # sort with most common first, then by position
    counts = sorted(counts, key=lambda t: (-t[1], t[2], t[3]))
    # now it's just the colors, sorted by frequency
    colors = [color for (color, _, _, _) in counts]

    # map colors based on frequency
    return Grid(map_multiple(obj.grid, colors, range(len(colors))))


def test_and_fix_invariance(input_obj, output_obj, source_obj, invariant):
    # returns tuple (is_equivalent, fixed_output_obj)
    # where is_equivalent is a boolean for whether input_obj == source_obj under
    # the invariance, and fixed_output_obj is the output_obj with the invariance
    # fixed
    if _equals_exact(input_obj)(source_obj):
        return True, Grid(output_obj.grid, (0, 0), cutout=False)

    if invariant == 'rotation':
        if _equals_exact(source_obj)(_rotate_ccw(input_obj)):
            return True, _rotate_ccw(output_obj)
        elif _equals_exact(source_obj)(_rotate_ccw(_rotate_ccw(input_obj))):
            return True, _rotate_ccw(_rotate_ccw(output_obj))
        elif _equals_exact(source_obj)(_rotate_ccw(_rotate_ccw(_rotate_ccw(input_obj)))):
            return True, _rotate_ccw(_rotate_ccw(_rotate_ccw(output_obj)))
        return False, None
    elif invariant == 'color':
        matches = _equals_exact(_color_transform(source_obj))(_color_transform(input_obj))
        if matches:
            # make map between colors
            colors, locations = np.unique(source_obj.grid, return_index=True)
            corresponding_colors = [input_obj.grid.flatten()[i] for i in locations]
            # reversing the mapping
            return True, Grid(map_multiple(output_obj.grid, corresponding_colors, colors))
        else:
            return False, None
    elif invariant == 'size':
        if len(input_obj.grid) > len(source_obj.grid):
            # need to deflate output_obj
            scale = len(input_obj.grid) / len(source_obj.grid)
            if scale != int(scale):
                return False, None

            scale = int(scale)
            if _equals_exact(_deflate(input_obj)(scale))(source_obj):
                return True, _deflate(output_obj)(scale)
            else:
                return False, None
        else:
            # need to inflate output_obj
            scale = len(input_obj.grid) / len(source_obj.grid)
            if scale != int(scale):
                return False, None

            scale = int(scale)

            if _equals_exact(_inflate(input_obj)(scale))(source_obj):
                return True, _inflate(output_obj)(scale)
            else:
                return False, None
    else:
        arc_assert(invariant == 'none')
        return False, None


def _equals_invariant(obj1):
    def equals(obj1, obj2, invariant):
        if _equals_exact(obj1)(obj2): return True

        if invariant == 'rotation':
            return np.any([_equals_exact(obj1)(_rotate_ccw(obj2)),
                    _equals_exact(obj1)(_rotate_ccw(_rotate_ccw(obj2))),
                    _equals_exact(obj1)(_rotate_ccw(_rotate_ccw(
                        _rotate_ccw(obj2))))])
        elif invariant == 'color':
            return _equals_exact(_color_transform(obj1))(_color_transform(obj2))
        elif invariant == 'size':
            if len(obj1.grid) > len(obj2.grid):
                obj1, obj2 = obj2, obj1

            scale = len(obj2.grid) / len(obj1.grid)
            if scale != int(scale):
                return False

            scale = int(scale)
            return _equals_exact(_inflate(obj1)(scale))(obj2)
        else:
            arc_assert(invariant == 'none')
            return _equals_exact(obj1)(obj2)

    return lambda obj2: lambda invariant: equals(obj1, obj2, invariant)

def _construct_mapping2(invariant):
    def construct(invariant, input):
        obj_fn = lambda g: Grid(g.grid, position=(0,0), cutout=False)
        return _construct_mapping(lambda g: [obj_fn(g)])(lambda g: [g])(invariant)(input)[0]

    return lambda i: construct(invariant, i)

def _construct_mapping3(f):
    def construct(f, input):
        list1 = [f(grid) for grid in _input_grids(input)]
        list2 = [grid for grid in _output_grids(input)]
        list_pairs = zip(list1, list2)

        to_map = f(_input(input))

        candidates = []
        for input_thing, output_grid in list_pairs:
            if to_map == input_thing:
                return output_grid

        arc_assert(False, "couldnt make a mapping")

    return lambda i: construct(f, i)

def _construct_mapping(f):
    def construct(f, g, invariant, input):
        # list of list of objects, most likely
        list1 = [f(grid) for grid in _input_grids(input)]
        # list of list of objects
        list2 = [g(grid) for grid in _output_grids(input)]

        list_zip = [zip(l1, l2) for l1, l2 in zip(list1, list2) if len(l1) ==
                len(l2)]
        list_pairs = [pair for l in list_zip for pair in l]

        # list of objects in test input
        list_to_map = f(_input(input))

        # for each object in list_to_map, if it equals something in list1, map
        # it to the corresponding element in list2. If multiple, choose the
        # largest.
        new_list = []
        for obj in list_to_map:
            # for those which match, we'll choose the largest one
            candidates = []
            # if the input object matches under invariance, we'll map to the
            # "fixed" output object. For example, if input was rotated, we
            # unrotate the output object and map to that.
            for input_obj, output_obj in list_pairs:
                equals_invariant, fixed_output_obj = test_and_fix_invariance(
                    input_obj, output_obj, obj, invariant)
                if equals_invariant:
                    # TODO might need to deflate this delta for size invariance
                    y1, x1 = input_obj.position
                    y2, x2 = output_obj.position
                    y3, x3 = obj.position
                    delta_y, delta_x = y2 - y1, x2 - x1
                    fixed_output_obj.position = y3 + delta_y, x3 + delta_x
                    candidates.append(fixed_output_obj)

            # in order to be valid, everything must get mapped! ?
            arc_assert(len(candidates) != 0)

            candidates = sorted(candidates, key=lambda o:
                    _area(o))
            # choose the largest match
            match = candidates[-1]
            new_list.append(match)

        return new_list

    return lambda g: lambda invariant: lambda input: construct(f, g, invariant, input)

def place_object(grid, obj):
    # print('obj: {}'.format(obj))
    # note: x, y, w, h should be flipped in reality. just go with it
    y, x = obj.position
    h, w = obj.grid.shape
    g_h, g_w = grid.shape
    # may need to crop the grid for it to fit
    # if negative, crop out the first parts
    o_x, o_y = max(0, -x), max(0, -y)
    # if negative, start at zero instead
    x, y = max(0, x), max(0, y)
    # this also affects the width/height
    w, h = w - o_x, h - o_y
    # if spills out sides, crop out the extra
    w, h = min(w, g_w - x), min(h, g_h - y)
    grid[y:y+h, x:x+w] = obj.grid[o_y: o_y + h, o_x: o_x + w]

def place_into_grid(grid, objects):
    for obj in objects:
        place_object(grid, obj)

    return Grid(grid)

def _place_into_grid(objects):
    return lambda input: place_into_grid(np.zeros(input.input_grid.grid.shape).astype('int'), objects)

def _place_into_input_grid(objects):
    return lambda input: place_into_grid(np.array(input.input_grid.grid), objects)

def _not_pixel(o):
    return o.grid.size != 1


def _number_of_objects(i):
    return len(_objects(i))


def grid_split(g):
    row_colors = [r[0] for r in g.grid if np.all(r == r[0])]
    column_colors = [c[0] for c in g.grid.T if np.all(c == c[0])]
    colors = row_colors + column_colors
    arc_assert(len(colors) > 0)
    color = np.argmax(np.bincount(colors))


def undo_grid_split(grid_split, objects):
    color, columns, rows, shape = grid_split
    h, w = shape
    grid = np.zeros(shape, dtype=int)
    for c in columns:
        grid[:,c] = np.full((h, 1), color)
    for r in rows:
        grid[r] = np.full((1, w), color)

    n = 0
    for i, r in enumerate([-1] + rows):
        for j, c in enumerate([-1] + cols):
            o = objects[n]
            o_w, o_h = o.grid.shape
            grid[r+1 : r+1 + o_h][c+1 : c+1 + o_w] = o.grid
            n += 1

    return Grid(grid)


def _grid_split_and_back(g):
    def grid_and_back(g, f):
        grid_split, objects = grid_split(g)
        objects = f(objects)
        return undo_grid_split(grid_split, objects)

    return lambda f: grid_and_back(g, f)


def _draw_line_slant_up(g):
    return lambda o: _draw_line(g)(o)(45)

def _draw_line_slant_down(g):
    return lambda o: _draw_line(g)(o)(315)

def _draw_line_down(g):
    return _draw_line(g)(270)

def _row(g):
    return lambda w: Grid(np.full((1, w), 1))

def _rectangle(o):
    # returns object as a rectangle
    arc_assert(_is_rectangle(o))
    return o

def _hollow(rec):
    new_grid = rec.grid[1:-1, 1:-1]
    y, x = rec.position
    return Grid(new_grid, position=(y+1, x+1), cutout=False)

def _shell(rec):
    new_grid = np.array(rec.grid)
    new_grid[1:-1, 1:-1] = 0
    return Grid(new_grid, position=rec.position, cutout=False)

def _enclose_with_ring(obj):
    def enclose(obj, color):
        y, x = obj.grid.shape
        new_grid = np.full((y+2, x+2), color)
        new_grid[1:-1, 1:-1] = obj.grid
        y, x = obj.position
        return Grid(new_grid, position=(y-1,x-1))

    return lambda color: enclose(obj, color)

def _fill_rectangle(obj):
    def fill(obj, color):
        grid = np.array(obj.grid)
        grid[::] = color
        return Grid(grid, obj.position)

    return lambda color: fill(obj, color)



def _is_rectangle(o):
    # object is a rectangle of the perimeter of nonblank colors forms a
    # rectangle. Inside may be blank.
    grid = o.grid
    y_range, x_range = np.nonzero(grid)
    cut = grid[min(y_range):max(y_range) + 1, min(x_range):max(x_range) + 1]
    border = []
    border += list(cut[0, :-1])     # Top row (left to right), not the last element.
    border += list(cut[:-1, -1])    # Right column (top to bottom), not the last element.
    border += list(cut[-1, :0:-1])  # Bottom row (right to left), not the last element.
    border += list(cut[1:, 0])    # Left column (top to bottom), not the first element.
    is_rec = len(border) == sum([c != 0 for c in border])
    return is_rec


def _is_rectangle_not_pixel(o):
    return _not_pixel(o) and _is_rectangle(o)


def _hblock(i):
    def block(length, color):
        arc_assert(length >= 1 and length <= 60)
        shape = (1, length)
        return Grid(np.full(shape, color))
    return lambda c: block(i, c)


def _vblock(i):
    def block(length, color):
        arc_assert(length >= 1 and length <= 60)
        shape = (length, 1)
        return Grid(np.full(shape, color))
    return lambda c: block(i, c)


def num_mistakes(width, height, down_shift, right_shift, grid,
        occlusion_color):
    arc_assert(down_shift == 0 or right_shift == 0)
    base_grid = {(y, x): None for x in range(width) for y in range(height)}

    def project(y, x):
        a = y % height
        b = x % width
        h_shifts = y // height
        w_shifts = x // width
        y2 = (a - down_shift * w_shifts) % height
        x2 = (b - right_shift * h_shifts) % width
        return y2, x2

    shift = down_shift or right_shift
    mistakes = 0
    for (y, x), val in np.ndenumerate(grid):
        if val != occlusion_color:
            y2, x2 = project(y, x)
            if base_grid[(y2, x2)] is None:
                base_grid[(y2, x2)] = val
            elif base_grid[(y2, x2)] != val:
                mistakes += 1

    tile = np.full((height, width), occlusion_color)
    for (y, x), val in base_grid.items():
        tile[(y, x)] = val if val else occlusion_color

    return mistakes, tile

def tile_grid(tile, down_shift, right_shift, shape):
    arc_assert(down_shift == 0 or right_shift == 0)

    H, W = shape
    h, w = tile.shape
    w_repeats = math.ceil(W / w)
    h_repeats = math.ceil(H / h)

    if down_shift == 0 and right_shift == 0:
        return np.kron(np.ones((h_repeats, w_repeats)), tile)[:H, :W]
    if down_shift != 0:
        tile = np.kron(np.ones((h_repeats+1, 1)), tile)
        # moving right shifts down by down_shift
        panels = [tile[ (d*down_shift) % h : H + (d*down_shift % h) ]
                for d in range(0, -w_repeats, -1)]
        return np.concatenate(panels, axis=1)[:, :W]

    else: # right_shift
        tile = np.kron(np.ones((1, w_repeats+1)), tile)
        # moving down shifts right by down_shift
        panels = [tile[:, (d*right_shift) % w : W + (d*right_shift % w) ]
                for d in range(0, -h_repeats, -1)]
        return np.concatenate(panels, axis=0)[:H]


def _tile_to_fill2(g):
    def tile_fill(g, occlusion_color):
        grid = g.grid
        for shift in range(0, int(max(grid.shape)/2)):
            for h in range(1, len(grid)-1):
                for w in range(1, len(grid[0])-1):
                    if shift <= h/2:
                        n, tile = num_mistakes(w, h, shift, 0, grid, occlusion_color)
                        if n == 0 and np.sum(tile == occlusion_color) == 0:
                            print('down, zero mistakes with {}'.format((h, w, shift)))
                            return tile_grid(tile, shift, 0, grid.shape)
                    if shift <= w/2:
                        n, tile = num_mistakes(w, h, 0, shift, grid, occlusion_color)
                        if n == 0 and np.sum(tile == occlusion_color) == 0:
                            print('right, zero mistakes with {}'.format((h, w, shift)))
                            return tile_grid(tile, 0, shift, grid.shape)

        arc_assert(False)

    return lambda col: Grid(tile_fill(g, col).astype(int))

def _tile_to_fill(g):
    '''
        Given a grid with occlusions, find grid size which maximizes the color
        matching, and then fill the grid.
    '''
    def tile_split(grid, shape):
        h, w = shape
        # extend beyond
        H, W = grid.shape
        new_h, new_w = math.ceil(H / h)*h, math.ceil(W / w)*w
        assert new_h % h == 0
        assert new_w % w == 0
        grid2 = np.full((new_h, new_w), -1)
        grid2[:H, :W] = grid

        h_repeats = int(new_h / h)
        w_repeats = int(new_w / w)
        return [grid2[int(i*h) : int((i+1)*h), int(j*w) : int((j+1)*w)] for i in range(h_repeats)
                for j in range(w_repeats)]


    def num_mistakes(tiling, grid, occlusion_color):
        return np.sum(tiling == grid) - np.sum(grid == occlusion_color)


#     def tile_fill2(g, occlusion_color):
#         grid = g.grid
#         # print('grid: {}'.format(grid))
#         # choose candidate row with least occlusions
#         c = np.argmin(np.sum(grid == occlusion_color, axis=0))
#         r = np.argmin(np.sum(grid == occlusion_color, axis=1))
#         col = grid[:,c]
#         row = grid[r]
#         # find w_repeat and h_repeat
#         # minimum with perfect overlap, ignoring
#         h_choice = -1
#         for h in range(1, len(col)):
#             repeats = math.ceil(len(col) / h)
#             tile = col[0:h]
#             g = np.kron(tile, np.ones((1, repeats)))[0:len(col)]
#             mistakes = num_mistakes(g, col, occlusion_color)
#             if mistakes == 0:
#                 h_choice = h
#                 h_tile = tile
#                 h_compare_tile = np.kron(tile, np.ones((1, repeats+1)))
#                 break

#         w_choice = -1
#         for w in range(1, len(row)):
#             repeats = math.ceil(len(row) / w)
#             tile = row[0:w]
#             g = np.kron(tile, np.ones((1, repeats)))[0:len(row)]
#             mistakes = num_mistakes(g, row, occlusion_color)
#             if mistakes == 0:
#                 w_choice = w
#                 w_tile = tile
#                 h_compare_tile = np.kron(tile, np.ones((1, repeats+1)))
#                 break

#         arc_assert(h_choice != -1 or w_choice != -1)
#         # for whichever one worked, tile it nicely.
#         if h_choice != -1:
#             # find the width of the tile.
#             # delta which maximizes
#             for w in range(2, len(grid[0])):
#                 for delta in range(0, h_choice):
#                     tiling = h_compare_tile[delta:delta+len(grid)]
#                     mistakes = num_mistakes(tiling, grid[:, w-1])
#                     if matches

#         w_repeat =
#         for h in range(1, len(grid)):
#             for w in range(1, len(grid[0])):

    def tile_fill(g, occlusion_color):
        grid = g.grid
        # print('grid: {}'.format(grid))
        options = []
        for h in range(1, len(grid)):
            for w in range(1, len(grid[0])):
                mistakes = 0
                # only correct if all match, except occlusion areas
                d = {(i, j): {occlusion_color: -1} for i in range(h) for j in range(w)}
                for tile in tile_split(grid, (h, w)):
                    # -1 represents spilling over.
                    for (i, j), val in np.ndenumerate(tile):
                        if val != -1 and val != occlusion_color:
                            if val not in d[(i, j)]:
                                if len(d[(i, j)]) != 1:
                                    mistakes += 1
                                d[(i, j)][val] = 1
                            else:
                                d[(i, j)][val] += 1
                # print((h, w))
                # print('mistakes: {}'.format(mistakes))

                most_likely_tile = np.zeros((h, w))
                for i in range(h):
                    for j in range(w):
                        possible = d[(i, j)].items()
                        possible = sorted(possible, key=lambda p: -p[1])
                        most_likely_tile[i, j] = possible[0][0]

                options.append(((h, w), mistakes, most_likely_tile))

        # choose h, w with the least mistakes
        options = sorted(options, key=lambda t: t[1])
        h, w = options[0][0]
        # print('chosen: {}'.format((h, w)))
        tile = options[0][2]
        # print('tile: {}'.format(tile))
        H, W = grid.shape
        h_repeat = math.ceil(H / h)
        w_repeat = math.ceil(W / w)
        ones = np.ones((h_repeat, w_repeat))
        oversized = np.kron(ones, tile).astype(int)
        filled = oversized[:H, :W]
        return Grid(filled)

    return lambda c: tile_fill(g, c)


# def rotate_to_fill(grid, center_position, occlusion_color):



## making the actual primitives

colors = {
    'color'+str(i): Primitive("color"+str(i), tcolor, i) for i in range(0, MAX_COLOR + 1)
    }
directions = {
    'dir'+str(i): Primitive('dir'+str(i), tdir, i) for i in range(0, 360, 45)
    }

ints = {
    str(i): Primitive(str(i), tint, i) for i in range(0, MAX_INT + 1)
    }
bools = {
    "True": Primitive("True", tboolean, True),
    "False": Primitive("False", tboolean, False)
    }

list_primitives = {
    "get": Primitive("get", arrow(tlist(t0), tint, t0), _get),
    "get_first": Primitive("get_first", arrow(tlist(t0), t0), _get_first),
    "get_last": Primitive("get_last", arrow(tlist(t0), t0), _get_last),
    "list_length": Primitive("list_length", arrow(tlist(t0), tint), _length),
    "sort_incr": Primitive("sort_incr", arrow(tlist(t0), arrow(t0, tint), tlist(t0)), _sort_incr),
    "sort_decr": Primitive("sort_decr", arrow(tlist(t0), arrow(t0, tint), tlist(t0)), _sort_decr),
    "map": Primitive("map", arrow(arrow(tgrid, tgrid), tlist(tgrid), tlist(tgrid)), _map),
    "filter_list": Primitive("filter_list", arrow(tlist(t0), arrow(t0, tboolean), tlist(t0)), _filter_list),
    "compare": Primitive("compare", arrow(arrow(t0, t1), t0, t0, tboolean), _compare),
    "zip": Primitive("zip", arrow(tlist(t0), tlist(t1), arrow(t0, t1, t2), tlist(t2)), _zip),
    "reverse_list": Primitive("reverse_list", arrow(tlist(t0), tlist(t0)), _reverse),
    "apply_colors": Primitive("apply_colors", arrow(tlist(tgrid), tlist(tcolor)), _apply_colors)
    }

line_primitives = {
    "draw_connecting_line": Primitive("draw_connecting_line", arrow(toriginal, tlist(tobject), tgrid), _draw_connecting_line),
    "draw_line": Primitive("draw_line", arrow(tgrid, tgrid, tdir, tgrid), _draw_line),
    "draw_line_slant_down": Primitive("draw_line_slant_down", arrow(toriginal, tobject, tgrid), _draw_line_slant_down),
    "draw_line_slant_up": Primitive("draw_line_slant_up", arrow(toriginal, tobject, tgrid), _draw_line_slant_up),
    "draw_line_down": Primitive("draw_line_down", arrow(tgrid, tgrid), _draw_line_down),

}

grid_primitives = {
    "map_i_to_j": Primitive("map_i_to_j", arrow(tgrid, tcolor, tcolor, tgrid), _map_i_to_j),
    "find_in_list": Primitive("find_in_list", arrow(tlist(tgrid), tint), _find_in_list),
    "find_in_grid": Primitive("find_in_grid", arrow(tgrid, tgrid, tposition), _find_in_grid),
    "filter_color": Primitive("filter_color", arrow(tgrid, tcolor, tgrid), _filter_color),
    "colors": Primitive("colors", arrow(tgrid, tlist(tcolor)), _colors),
    "num_colors": Primitive("num_colors", arrow(tgrid, tint), _num_colors),
    # "color": Primitive("color", arrow(tobject, tcolor), _color),
    "color": Primitive("color", arrow(tgrid, tcolor), _color),
    "objects": Primitive("objects", arrow(tgrid, tlist(tobject)), _objects),
    "objects_by_color": Primitive("objects_by_color", arrow(tgrid, tlist(tgrid)), _objects_by_color),
    # "object": Primitive("object", arrow(toriginal, tgrid), _object),
    "object": Primitive("object", arrow(tgrid, tgrid), _object),
    "objects2": Primitive("objects2", arrow(tgrid, tbase_bool, tbase_bool, tlist(tgrid)), _objects2),
    # this one crops. The above doesn't
    "objects3": Primitive("objects2", arrow(tgrid, tbase_bool, tbase_bool, tlist(tgrid)), _objects_no_crop),
    "pixel2": Primitive("pixel2", arrow(tcolor, tgrid), _pixel2),
    "pixel": Primitive("pixel", arrow(tint, tint, tgrid), _pixel),
    "list_of": Primitive("list_of", arrow(tgrid, tgrid, tlist(tgrid)), _list_of),
    "list_of_one": Primitive("list_of_one", arrow(tgrid, tlist(tgrid)), _list_of_one),
    "pixels": Primitive("pixels", arrow(tgrid, tlist(tgrid)), _pixels),
    "set_shape": Primitive("set_shape", arrow(tgrid, tposition, tgrid), _set_shape),
    "shape": Primitive("shape", arrow(tgrid, tposition), _shape),
    "y_mirror": Primitive("y_mirror", arrow(tgrid, tgrid), _y_mirror),
    "x_mirror": Primitive("x_mirror", arrow(tgrid, tgrid), _x_mirror),
    "hflip": Primitive("hflip", arrow(tgrid, tgrid), _y_mirror),
    "vflip": Primitive("vflip", arrow(tgrid, tgrid), _x_mirror),
    "reflect_down": Primitive("reflect_down", arrow(tgrid, tgrid), _reflect_down),
    "crop_down": Primitive("crop_down", arrow(tgrid, tgrid), _crop_down),
    "rotate_ccw": Primitive("rotate_ccw", arrow(tgrid, tgrid), _rotate_ccw),
    "rotate_cw": Primitive("rotate_cw", arrow(tgrid, tgrid), _rotate_cw),
    "has_x_symmetry": Primitive("has_x_symmetry", arrow(tgrid, tboolean), _has_x_symmetry),
    "has_y_symmetry": Primitive("has_y_symmetry", arrow(tgrid, tboolean), _has_y_symmetry),
    "has_rotational_symmetry": Primitive("has_rotational_symmetry", arrow(tgrid, tboolean), _has_rotational_symmetry),
    "hblock": Primitive("hblock", arrow(tint, tcolor, tgrid), _hblock),
    "vblock": Primitive("vblock", arrow(tint, tcolor, tgrid), _vblock),
    }

input_primitives = {
    # "input": Primitive("input", arrow(tinput, toriginal), _input),
    "input": Primitive("input", arrow(tinput, tgrid), _input),
    # "grid": Primitive("grid", arrow(toriginal, tgrid), lambda i: i),
    "inputs": Primitive("inputs", arrow(tinput, tlist(tgrid)), _input_grids),
    "outputs": Primitive("outputs", arrow(tinput, tlist(tgrid)), _output_grids),
    "find_corresponding": Primitive("find_corresponding", arrow(tinput, tgrid, tgrid), _find_corresponding)
    }

list_consolidation = {
    # "vstack": Primitive("vstack", arrow(tlist(tgrid), tgrid), _vstack),
    # "hstack": Primitive("hstack", arrow(tlist(tgrid), tgrid), _hstack),
    "overlay": Primitive("overlay", arrow(tgrid, tgrid, tgrid), _overlay),
    "stack_overlay": Primitive("stack_overlay", arrow(tlist(tgrid), tgrid), _stack_overlay),
    "combine_grids_horizontally": Primitive("combine_grids_horizontally", arrow(tgrid, tgrid, tgrid), _combine_grids_horizontally),
    "combine_grids_vertically": Primitive("combine_grids_vertically", arrow(tgrid, tgrid, tgrid), _combine_grids_vertically),
    "vstack": Primitive("vstack", arrow(tgrid, tgrid, tgrid), _combine_grids_horizontally),
    "hstack": Primitive("hstack", arrow(tgrid, tgrid, tgrid), _combine_grids_horizontally),
    "hstack_pair": Primitive("hstack_pair", arrow(tgrid, tgrid, tgrid),
        _combine_grids_horizontally),
    "vstack_pair": Primitive("vstack_pair", arrow(tgrid, tgrid, tgrid),
        _combine_grids_vertically),
    }

boolean_primitives = {
    "and": Primitive("and", arrow(tboolean, tboolean, tboolean), _and),
    "or": Primitive("or", arrow(tboolean, tboolean, tboolean), _or),
    "not": Primitive("not", arrow(tboolean, tboolean), _not),
    "ite": Primitive("ite", arrow(tboolean, t0, t0, t0), _ite),
    "eq": Primitive("eq", arrow(t0, t0, tboolean), _eq)
    }

object_primitives = {
    "position": Primitive("position", arrow(tgrid, tposition), _position),
    "x": Primitive("x", arrow(tgrid, tint), _x),
    "y": Primitive("y", arrow(tgrid, tint), _y),
    "color_in": Primitive("color_in", arrow(tgrid, tcolor, tgrid), _color_in),
    "color_in_grid": Primitive("color_in_grid", arrow(toutput, tcolor, toutput), _color_in_grid),
    "flood_fill": Primitive("flood_fill", arrow(tgrid, tcolor, tgrid), _flood_fill),
    "size": Primitive("size", arrow(tgrid, tint), _size),
    "area": Primitive("area", arrow(tgrid, tint), _area),
    "move_down": Primitive("move_down", arrow(tgrid, tgrid), _move_down),
    "move_down2": Primitive("move_down2", arrow(tgrid, tgrid), _move_down2),
    }

misc_primitives = {
    "inflate": Primitive("inflate", arrow(tgrid, tint, tgrid), _inflate),
    "deflate": Primitive("deflate", arrow(tgrid, tgrid), _deflate_detect_scale),
    "kronecker": Primitive("kronecker", arrow(tgrid, tgrid, tgrid), _kronecker),
    "top_half": Primitive("top_half", arrow(tgrid, tgrid), _top_half),
    "bottom_half": Primitive("bottom_half", arrow(tgrid, tgrid), _bottom_half),
    "left_half": Primitive("left_half", arrow(tgrid, tgrid), _left_half),
    "right_half": Primitive("right_half", arrow(tgrid, tgrid), _right_half),
    }

simon_new_primitives = {
    "equals_exact": Primitive("equals_exact", arrow(tgrid, tgrid, tboolean), _equals_exact),
    "color_transform": Primitive("color_transform", arrow(tgrid, tgrid), _color_transform),
    "equals_invariant": Primitive("equals_invariant", arrow(tgrid, tgrid, tinvariant, tboolean), _equals_invariant),
    "construct_mapping": Primitive("construct_mapping", arrow(arrow(tgrid, tlist(tgrid)), arrow(tgrid, tlist(tgrid)), tinvariant, tinput, tlist(tgrid)), _construct_mapping),
    "construct_mapping2": Primitive("construct_mapping2", arrow(tinvariant, tinput, tgrid), _construct_mapping2),
    "construct_mapping3": Primitive("construct_mapping3", arrow(arrow(tgrid, t0), tinput, tgrid), _construct_mapping3),
    "size_invariant": Primitive("size_invariant", tinvariant, "size"),
    "no_invariant": Primitive("no_invariant", tinvariant, "none"),
    "rotation_invariant": Primitive("rotation_invariant", tinvariant, "rotation"),
    "color_invariant": Primitive("color_invariant", tinvariant, "color"),
    "rows": Primitive("rows", arrow(tgrid, tlist(tgrid)), _rows),
    "columns": Primitive("columns", arrow(tgrid, tlist(tgrid)), _columns),
    "place_into_input_grid": Primitive("place_into_input_grid", arrow(tlist(tgrid), tinput, tgrid), _place_into_input_grid),
    "place_into_grid": Primitive("place_into_grid", arrow(tlist(tgrid), tinput, tgrid), _place_into_grid),
    # "output": Primitive("output", arrow(tgrid, toutput), lambda i: i),
    "contains_color": Primitive("contains_color", arrow(tgrid, tcolor,
        tboolean), _contains_color),
    "T": Primitive("T", tbase_bool, True),
    "F": Primitive("F", tbase_bool, False),
    "not_pixel": Primitive("not_pixel", arrow(tgrid, tboolean), _not_pixel),
    "number_of_objects": Primitive("number_of_objects", arrow(tgrid, tint),
        _number_of_objects),
    "fill_rectangle": Primitive("fill_rectangle", arrow(tgrid, tcolor, tgrid), _fill_rectangle),
    "shell": Primitive("shell", arrow(tgrid, tgrid), _shell),
    "hollow": Primitive("hollow", arrow(tgrid, tgrid), _hollow),
    "is_rectangle": Primitive("is_rectangle", arrow(tgrid, tboolean),
        _is_rectangle),
    "is_rectangle_not_pixel": Primitive("is_rectangle", arrow(tgrid, tboolean),
        _is_rectangle_not_pixel),
    "enclose_with_ring": Primitive("enclose_with_ring", arrow(tgrid, tcolor, tgrid),
        _enclose_with_ring),
}

sylee_new_primitives = {
    #"object_frequency": Primitive("object_frequency", arrow(tgrid, tgrid, tint), _object_frequency),
    "max_object_frequency": Primitive("max_object_frequency", arrow(tgrid, tgrid), _max_object_frequency),
    "min_object_frequency": Primitive("min_object_frequency", arrow(tgrid, tgrid), _min_object_frequency),
}

pixelwise_primitives = {"stack_xor": Primitive("stack_xor", arrow(tlist(tgrid), tgrid), _stack_xor),
    "stack_and": Primitive("stack_and", arrow(tlist(tgrid), tgrid), _stack_and),
    #"apply_function": Primitive('apply_function', arrow(t0,arrow(t0,t1),t1), _apply_function),
    "complement": Primitive("complement",arrow(tgrid,tcolor,tgrid),_complement),
    "return_subgrids": Primitive("return_subgrids",arrow(tgrid,tlist(tgrid)),_return_subgrids),
    "grid_split": Primitive("grid_split",arrow(tgrid,tlist(tgrid)),_grid_split),
    #"grid_split_2d": Primitive("grid_split_2d",arrow(tgrid,tlist(tgrid)),_grid_split_2d)
    }

primitive_dict = {**colors, **directions, **ints, **bools, **list_primitives,
        **line_primitives,
        **grid_primitives, **input_primitives, **list_consolidation,
        **boolean_primitives, **object_primitives, **misc_primitives,
        **simon_new_primitives, **sylee_new_primitives, **pixelwise_primitives}

primitives = list(primitive_dict.values())

def generate_ocaml_primitives(primitives=None):
    if primitives == None:
        primitives = primitive_dict.values()

    with open("solvers/program.ml", "r") as f:
        contents = f.readlines()

    start_ix = min([i for i in range(len(contents)) if contents[i][0:7] == '(* AUTO'])
    end_ix = min([i for i in range(len(contents)) if contents[i][0:11] == '(* END AUTO'])

    non_auto_contents = contents[0:start_ix+1] + contents[end_ix:]
    # get the existing primitive names. We won't auto-create any primitives
    # whose name matches an existing name.
    existing_primitives = parse_primitive_names(non_auto_contents)

    lines = [p.ocaml_string() + '\n' for p in primitives
            if p.name not in existing_primitives]

    for p in primitives:
        if p.name in existing_primitives:
            print('Primitive {} already exists, skipping ocaml code generation for it'.format(p.name))

    contents = contents[0:start_ix+1] + lines + contents[end_ix:]

    with open("solvers/program.ml", "w") as f:
        f.write(''.join(contents))


def parse_primitive_names(ocaml_contents):
    contents = ''.join([c[:-1] for c in ocaml_contents if c[0:2] + c[-3:-1] != '(**)'])
    contents = contents.split('primitive "')[1:]
    primitives = [p[:p.index('"')] for p in contents if '"' in p]
    return primitives