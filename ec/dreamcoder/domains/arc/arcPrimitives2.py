from dreamcoder.program import *
from dreamcoder.type import arrow, baseType, tint, tlist, t0, t1, t2, tbool
from dreamcoder.task import Task
from dreamcoder.grammar import Grammar
from dreamcoder.program import Primitive

from scipy.ndimage import measurements
from math import sin,cos,radians,copysign
import numpy as np

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
        assert type(grid) == type(np.array([1])), 'bad grid type'
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
    def __init__(self, grid, position=(0,0), input_grid=None, cutout=False):
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
        super().__init__(grid, position)


class Input():
    """
        Combines i/o examples into one input, so that we can synthesize a solution
        which looks at different examples at once

    """
    def __init__(self, input_grid, training_examples):
        assert type(input_grid) == type(np.array([1])), 'bad grid type '+str(type(input_grid))
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
    return Object(g.grid, (0,0), g.input_grid, cutout=False)

def _move_down(g):
    # o.grid = np.roll(o.grid, 1, axis=0)
    # return Grid(o.grid)

    o = _get(_objects(g))(0)
    newg = Grid(g.grid)
    newg.grid[o.grid==1]=0 # remove object from old grid
    o.grid = np.roll(o.grid, 1, axis=0) # move down object 
    return _overlay(newg)(o) # add object back to grid

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
    pixel_grid = [[Pixel(g.grid[i:i+1, j:j+1], (i + g.pos[0], j + g.pos[1])) 
            for i in range(len(g.grid))]
            for j in range(len(g.grid[0]))]
    # flattens nested list into single list
    return [item for sublist in pixel_grid for item in sublist]

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
def _input(i): return i.input_grid

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
        stackedgrid += g.grid * (stackedgrid == 0)

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
        grid[grid != 0] = c
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

        return _stack_no_crop(grids)

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
    # sort colors by frequency and map accordingly
    counts = np.unique(obj.grid, return_counts=True)
    # list of (element, frequency) tuples
    counts = list(zip(*counts))
    # sort with most common first
    counts = sorted(counts, key=lambda t: -t[1])
    # now it's just the colors, sorted by frequency
    colors = list(zip(*counts))[0]
    # map colors based on frequency
    
    return Grid(map_multiple(obj.grid, colors, range(len(colors))))
    

def test_and_fix_invariance(input_obj, output_obj, source_obj, invariant):
    # returns tuple (is_equivalent, fixed_output_obj)
    # where is_equivalent is a boolean for whether input_obj == source_obj under
    # the invariance, and fixed_output_obj is the output_obj with the invariance
    # fixed 
    if _equals_exact(input_obj)(source_obj):
        return True, Object(output_obj.grid, (0, 0), output_obj.input_grid,
                cutout=False)

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
            return True, Object(map_multiple(output_obj.grid, corresponding_colors, colors))
        else:
            return False, None
    elif invariant == 'size':
        if len(input_obj.grid) > len(source_obj.grid):
            # need to deflate output_obj
            scale = len(input_obj.grid) / len(source_obj.grid)
            if scale != int(scale):
                return False, None

            if _equals_exact(_deflate(input_obj))(source_obj):
                return _deflate(output_obj)(scale)
            else:
                return False, None
        else:
            # need to inflate output_obj
            scale = len(input_obj.grid) / len(source_obj.grid)
            if scale != int(scale):
                return False, None

            if _equals_exact(_inflate(input_obj))(source_obj):
                return _inflate(output_obj)(scale)
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

def _construct_mapping2(f):
    def construct(f, g, invariant, input):
        obj_fn = lambda g: [Object(g.grid, position=(0,0), input_grid=g.grid,
            cutout=False)]
        return _construct_mapping(obj_fn)(obj_fn)(invariant)(input)[0]

    return lambda g: lambda inv: lambda i: construct(f, g, inv, i)

def _construct_mapping4(i):
    def construct(i, inf):
        obj_fn = lambda g: [Object(g.grid, position=(0,0), input_grid=g.grid,
            cutout=False)]
        return _construct_mapping3(obj_fn)(i)(inf)(lambda i: i)(lambda i: lambda
                j: j)[0]
    return lambda inf: construct(i, inf)

def _construct_mapping3(f):
    def construct(f, input, in_feature_fn, out_feature_fn, final_fn):
        # list of list of objects, most likely
        list1 = [f(grid) for grid in _input_grids(input)]
        # list of list of objects
        list2 = [f(grid) for grid in _output_grids(input)]

        list_zip = [zip(l1, l2) for l1, l2 in zip(list1, list2) if len(l1) ==
                len(l2)]

        list_pairs = [pair for l in list_zip for pair in l]
        list_pairs = [((g1, in_feature_fn(g1)), out_feature_fn(g2)) for (g1, g2) in list_pairs]

        # print('list_pairs: {}'.format(list_pairs))
        # list of objects in test input
        list_to_map = f(_input(input))

        # for each object in list_to_map, if it equals something in list1, map
        # it to the corresponding element in list2. If multiple, choose the
        # largest.
        new_list = []
        for obj in list_to_map:
            found_match = False
            target_feature = in_feature_fn(obj)
            # if the feature matches, we'll apply final_fn to it.
            for (in_obj, in_feature), out_feature in list_pairs:
                if in_feature == target_feature:
                    new_list.append(final_fn(obj)(out_feature))
                    found_match = True
                    break # go to next object

            arc_assert(found_match)

        return new_list

    return lambda i: lambda inf: lambda outf: lambda finalf: construct(
            f, i, inf, outf, finalf)

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
                    x1, y1 = input_obj.position
                    x2, y2 = output_obj.position
                    x3, y3 = obj.position
                    delta_x, delta_y = x2 - x1, y2 - y1
                    fixed_output_obj.position = x3 + delta_x, y3 + delta_y
                    fixed_output_obj.input_grid = obj.input_grid
                    candidates.append(fixed_output_obj)

            # in order to be valid, everything must get mapped! ?
            arc_assert(len(candidates) != 0)

            candidates = sorted(candidates, key=lambda o:
                    _area(o))
            # choose the largest match
            match = candidates[-1]
            new_list.append(match)

        # print('new_list: {}'.format(new_list))
        return new_list

    return lambda g: lambda invariant: lambda input: construct(f, g, invariant, input)

def _place_into_grid(objects):
    grid = np.zeros(objects[0].input_grid.shape, dtype=int)
    # print('grid: {}'.format(grid))
    for obj in objects:
        # print('obj: {}'.format(obj))
        # note: x, y, w, h should be flipped in reality. just go with it
        y, x = obj.position
        # print('x, y: {}'.format((x, y)))
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
        # print('x, y = {}, {}, o_x, o_y = {}, {}, w, h = {}, {}'.format(x, y,
            # o_x, o_y, w, h))

        grid[y:y+h, x:x+w] = obj.grid[o_y: o_y + h, o_x: o_x + w]

    return Grid(grid)

def _place_into_input_grid(objects):
    grid = np.copy(objects[0].input_grid)
    for obj in objects:
        # note: x, y, w, h should be flipped in reality. just go with it
        x, y = obj.position
        w, h = obj.grid.shape
        # x or y might be negative, in which case we only need the later part.
        if x < 0:
            obj.grid = obj.grid[-x:,:]
            x = 0
        if y < 0:
            obj.grid = obj.grid[:,-y:]
            y = 0
        grid[x:x+w, y:y+h] = obj.grid

    return Grid(grid)



def _not_pixel(o):
    return o.grid.size != 1
    
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
    "True": Primitive("True", tbool, True),
    "False": Primitive("False", tbool, False)
    }

list_primitives = {
    "get": Primitive("get", arrow(tlist(t0), tint, t0), _get),
    "get_first": Primitive("get_first", arrow(tlist(t0), t0), _get_first),
    "get_last": Primitive("get_last", arrow(tlist(t0), t0), _get_last),
    "length": Primitive("length", arrow(tlist(t0), tint), _length),
    "remove_head": Primitive("remove_head", arrow(tlist(t0), t0), _remove_head),
    "sortby": Primitive("sortby", arrow(tlist(t0), arrow(t0, t1), tlist(t0)), _sortby),
    "map": Primitive("map", arrow(arrow(t0, t1), tlist(t0), tlist(t1)), _map),
    "filter_list": Primitive("filter_list", arrow(tlist(t0), arrow(t0, tbool), tlist(t0)), _filter_list),
    "compare": Primitive("compare", arrow(arrow(t0, t1), t0, t0, tbool), _compare),    
    "zip": Primitive("zip", arrow(tlist(t0), tlist(t1), arrow(t0, t1, t2), tlist(t2)), _zip),    
    "reverse": Primitive("reverse", arrow(tlist(t0), tlist(t0)), _reverse),
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
    "color": Primitive("color", arrow(tobject, tcolor), _color),
    "objects": Primitive("objects", arrow(tgrid, tlist(tobject)), _objects),
    "objects_by_color": Primitive("objects_by_color", arrow(tgrid, tlist(tgrid)), _objects_by_color),
    "group_objects_by_color": Primitive("group_objects_by_color", arrow(toriginal, tlist(tlist(tobject))), _group_objects_by_color),
    "object": Primitive("object", arrow(tgrid, tgrid), _object),
    "pixel2": Primitive("pixel2", arrow(tcolor, tgrid), _pixel2),
    "pixel": Primitive("pixel", arrow(tint, tint, tgrid), _pixel),
    "list_of": Primitive("list_of", arrow(tgrid, tgrid, tlist(tgrid)), _list_of),
    "pixels": Primitive("pixels", arrow(tgrid, tlist(tgrid)), _pixels),
    "set_shape": Primitive("set_shape", arrow(tgrid, tposition, tgrid), _set_shape),
    "shape": Primitive("shape", arrow(tgrid, tposition), _shape),
    "y_mirror": Primitive("y_mirror", arrow(tgrid, tgrid), _y_mirror),
    "x_mirror": Primitive("x_mirror", arrow(tgrid, tgrid), _x_mirror),
    "reflect_down": Primitive("reflect_down", arrow(tgrid, tgrid), _reflect_down),
    "rotate_ccw": Primitive("rotate_ccw", arrow(tgrid, tgrid), _rotate_ccw),
    "rotate_cw": Primitive("rotate_cw", arrow(tgrid, tgrid), _rotate_cw),
    "has_x_symmetry": Primitive("has_x_symmetry", arrow(tgrid, tbool), _has_x_symmetry),
    "has_y_symmetry": Primitive("has_y_symmetry", arrow(tgrid, tbool), _has_y_symmetry),
    "has_rotational_symmetry": Primitive("has_rotational_symmetry", arrow(tgrid, tbool), _has_rotational_symmetry),
    }

input_primitives = {
    "input": Primitive("input", arrow(tinput, toriginal), _input),
    "inputs": Primitive("inputs", arrow(tinput, tlist(tgrid)), _input_grids),
    "outputs": Primitive("outputs", arrow(tinput, tlist(tgrid)), _output_grids),
    "find_corresponding": Primitive("find_corresponding", arrow(tinput, tgrid, tgrid), _find_corresponding)
    }

list_consolidation = {
    "vstack": Primitive("vstack", arrow(tlist(tgrid), toutput), _vstack),
    "hstack": Primitive("hstack", arrow(tlist(tgrid), toutput), _hstack),
    "overlay": Primitive("overlay", arrow(tgrid, tgrid, tgrid), _overlay),
    "positionless_stack": Primitive("positionless_stack", arrow(tlist(tgrid), toutput), _positionless_stack),
    "stack": Primitive("stack", arrow(tlist(tgrid), toutput), _stack),
    "stack_no_crop": Primitive("stack_no_crop", arrow(tlist(tgrid), tgrid), _stack_no_crop),
    "combine_grids_horizontally": Primitive("combine_grids_horizontally", arrow(tgrid, tgrid, tgrid), _combine_grids_horizontally),
    "combine_grids_vertically": Primitive("combine_grids_vertically", arrow(tgrid, tgrid, tgrid), _combine_grids_vertically),
    }

boolean_primitives = {
    "and": Primitive("and", arrow(tbool, tbool, tbool), _and),
    "or": Primitive("or", arrow(tbool, tbool, tbool), _or),
    "not": Primitive("not", arrow(tbool, tbool), _not),
    "ite": Primitive("ite", arrow(tbool, t0, t0, t0), _ite),
    "eq": Primitive("eq", arrow(t0, t0, tbool), _eq)
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
    }

misc_primitives = {
    "inflate": Primitive("inflate", arrow(tgrid, tgrid), _inflate),
    "top_half": Primitive("top_half", arrow(tgrid, tgrid), _top_half),
    "bottom_half": Primitive("bottom_half", arrow(tgrid, tgrid), _bottom_half),
    "left_half": Primitive("left_half", arrow(tgrid, tgrid), _left_half),
    "right_half": Primitive("right_half", arrow(tgrid, tgrid), _right_half),
    }

simon_new_primitives = {
    "equals_exact": Primitive("equals_exact", arrow(tgrid, tgrid, tboolean), _equals_exact),
    "color_transform": Primitive("color_transform", arrow(tgrid, tgrid), _color_transform),
    "equals_invariant": Primitive("equals_invariant", arrow(tgrid, tgrid, tboolean), _equals_invariant),
    "construct_mapping": Primitive("construct_mapping", arrow(arrow(tgrid, tlist(tgrid)), arrow(tgrid, tlist(tgrid)), tinvariant, tinput, tlist(tgrid)), _construct_mapping),
    "construct_mapping3": Primitive("construct_mapping3", arrow(arrow(tgrid, tlist(tgrid)), tinput, arrow(tgrid, t0), arrow(tgrid, t1), arrow(tgrid, t1, tgrid), tlist(tgrid)), _construct_mapping3),
    "construct_mapping2": Primitive("construct_mapping2", arrow(arrow(tgrid, tgrid), arrow(tgrid, tgrid), tinvariant, tinput, tgrid), _construct_mapping2),
    "construct_mapping4": Primitive("construct_mapping4", arrow(tinput, arrow(tgrid, t0), toutput), _construct_mapping4),
    "size_invariant": Primitive("size_invariant", tinvariant, "size"),
    "no_invariant": Primitive("no_invariant", tinvariant, "none"),
    "rotation_invariant": Primitive("rotation_invariant", tinvariant, "rotation"),
    "color_invariant": Primitive("color_invariant", tinvariant, "color"),
    "rows": Primitive("rows", arrow(tgrid, tlist(tgrid)), _rows),
    "columns": Primitive("columns", arrow(tgrid, tlist(tgrid)), _columns),
    "place_into_input_grid": Primitive("place_into_input_grid", arrow(tlist(tgrid), toutput), _place_into_input_grid),
    "place_into_grid": Primitive("place_into_grid", arrow(tlist(tgrid), toutput), _place_into_grid),
    "output": Primitive("output", arrow(tgrid, toutput), lambda i: i),
    "contains_color": Primitive("contains_color", arrow(tgrid, tcolor,
        tboolean), _contains_color),
    "T": Primitive("T", tbase_bool, True),
    "F": Primitive("F", tbase_bool, False),
}


primitive_dict = {**colors, **directions, **ints, **bools, **list_primitives,
        **line_primitives,
        **grid_primitives, **input_primitives, **list_consolidation,
        **boolean_primitives, **object_primitives, **misc_primitives,
        **simon_new_primitives}

primitives = list(primitive_dict.values())
