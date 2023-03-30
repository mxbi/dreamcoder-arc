from dreamcoder.domains.arc.arcPrimitives import * 
from dreamcoder.domains.arc.arcPrimitives import Grid
import numpy as np

def _overlay_and(grid1,grid2):
        #both grids must have value in overlay
        arr1, arr2 = grid1.grid,grid2.grid
        arc_assert(np.shape(arr1)==np.shape(arr2),'overlayed two grids of unequal size')
        new_arr = np.zeros(np.shape(arr1))
        for i in range(np.shape(arr1)[0]):
            for j in range(np.shape(arr1)[1]):
                if arr1[i][j]!=0 and arr2[i][j]!=0:
                    new_arr[i][j] = arr1[i][j]
        return Grid(new_arr)
    
def _overlay_xor(grid1,grid2):
        #only one grid can have value in overlay
        arr1, arr2 = grid1.grid,grid2.grid
        arc_assert(np.shape(arr1)==np.shape(arr2),'overlayed two grids of unequal size')
        new_arr = np.zeros(np.shape(arr1))
        for i in range(np.shape(arr1)[0]):
            for j in range(np.shape(arr1)[1]):
                if arr1[i][j]!=0 and arr2[i][j]==0:
                    new_arr[i][j] = arr1[i][j]
                elif arr1[i][j]==0 and arr2[i][j]!=0:
                    new_arr[i][j]=arr2[i][j]
        return Grid(new_arr)
    

def _stack_main(grids,function):
    arc_assert(len(grids),'need at least one grid')
    if len(grids)==1:
        return grids[0]
    #elif len(grids)==0:
    #    return []
    new_grid = Grid(grids[0].grid)
    for grid in grids[1:]:
        new_grid = function(new_grid,grid)
    return new_grid

def _stack_and(grids):
    #stacks so that both grids must have pixesl there (and)
    return _stack_main(grids,_overlay_and)
def _stack_xor(grids):
    #stacks so that only one grid can have pixels there
    return _stack_main(grids,_overlay_xor)
          



############################
#SOLVING PIXELWISE NO BORDERS#
##############################


def _horiz_vert_subdivides(grid):
    #this function is used to find the subgrids of a grid.
    #for examples of what a subgrid is, look at the four quadrants
    # of task 179
    def _horizontal_subdivide(grid):
        def check_for_one_color(cells):
            #checks cells to see if they have one color
            #only takes in one row
            # cells is type np array not grid
            cells_shape = np.shape(cells)
            cells_width = cells_shape[0]
            color = None
            for j in range(cells_width):
                if cells[j]!=0: #cells can be either black or a different color
                    if color is None:
                        color = cells[j]
                    elif color!=cells[j]:
                        return False
            return True
        shape = np.shape(grid)
        height, width = shape                  
        for subdivide in range(1,width//2+1):
            if width%subdivide!=0:
                continue
            else:
                increment = width/subdivide
                spots = [int(increment*i) for i in range(0,subdivide+1)]
                iteration = True
                for i in range(height):
                    for s in range(len(spots)-1):
                        if not check_for_one_color(grid[i][spots[s]:spots[s+1]]):
                            iteration = False
                            break
                    if iteration==False:
                        break
                else:
                    return width/subdivide
    return _horizontal_subdivide(np.rot90(grid)),_horizontal_subdivide(grid)#gets both vertical and horizontal


def _return_subgrids(griddy):
    #returns list of subgrids as divided in dimensions 
    #used for subgrids without bars
    #for examples of what a subgrid is, look at the four quadrants
    # of task 179
    grid = griddy.grid
    dim = _horiz_vert_subdivides(grid)
    if type(dim[0])!=int or type(dim[1])!=int:
        return np.asarray([griddy])
    height,width = np.shape(grid)
    grid_list = []
    for i in range(int(height/dim[1])):
        for j in range(int(width/dim[0])):
            posh = int(dim[1]*i)
            posw = int(dim[0]*j)
            grid_list.append(Grid(grid[posh:posh+int(dim[1]),posw:posw+int(dim[0])]))
    return np.asarray(grid_list)


#GENERAL PRIMITIVES

def _complement(griddy):
    #every element in grid that is black becomes color and all others become black
    #ADD THIS IN
    def complement(color):
        grid = griddy.grid
        new_grid = np.zeros(np.shape(grid))
        for i in range(np.shape(grid)[0]):
            for j in range(np.shape(grid)[1]):
                if grid[i][j]==0:
                    new_grid[i][j] = color
        return Grid(new_grid)
    return complement

#####################
#GRIDSPLITTING
#####################

def _filter_color_neg1(griddy):
    #returns grid of only one color
    #all other elements are -1
    #make this a function i only use
    grid = griddy.grid
    def setter(color):
        new_grid = np.full(np.shape(grid),-1)
        for i in range(np.shape(new_grid)[0]):
            for j in range(np.shape(new_grid)[1]):
                if grid[i,j]==color:
                    
                    new_grid[i,j]=color
        return Grid(new_grid)
    return setter

def _all_one_color(griddy):
    #no black rows
    #checks to see if grid is all one color
    # if so returns that color, else returns false
    grid = griddy.grid
    for color in range(0,10):
        if np.all(grid==color):
            return color
    return False

def _get_midlines(griddy):
    #returns the borders if grid has borders, false otherwise
    #assumes input grid only has one color or black
    def mini(grid):
        if np.array_equal(grid,np.full(np.shape(grid),-1)):
            return False
        horiz = []
        vert = []
        for row in range(np.shape(grid)[0]):
            if _all_one_color(Grid(grid[row,:])):
                horiz.append(row)
        for col in range(np.shape(grid)[1]):
            if _all_one_color(Grid(grid[:,col])):
                vert.append(col)
        for i in range(np.shape(grid)[0]):
            for j in range(np.shape(grid)[1]):
                if grid[i,j]!=-1 and i not in horiz and j not in vert:
                    return False
        return {'horiz':horiz,
                'vert':vert}
    for color in range(1,10):
        new_grid = _filter_color_neg1(griddy)(color)
        result = mini(new_grid.grid)
        if result is not False:
            return result
    return False
    
   
    


def _grid_split(grid):
    return _grid_split_2d(grid).flatten()

def _grid_split_2d(griddy):
    #returns 2d array of grids
    grid = griddy.grid
    borders = _get_midlines(griddy)
    if borders is False:
        return np.array([Grid(grid)])
    grid_list = []
    horizontal = [-1]+borders['horiz']+[np.shape(grid)[0]]
    vertical = [-1]+borders['vert']+[np.shape(grid)[1]]
    for i in range(len(horizontal)-1):
        mini = []
        for j in range(len(vertical)-1):
            horiz1 = horizontal[i]+1
            horiz2 = horizontal[i+1]
            vert1 = vertical[j]+1
            vert2 = vertical[j+1]
            if abs(horiz1-horiz2)>1 and abs(vert1-vert2)>1:
                #in case two borders are right next to one another, we dont want both
                mini.append(Grid(grid[horiz1:horiz2,vert1:vert2]))
        if mini:
            grid_list.append(np.asarray(mini))
    return np.asarray(grid_list) #valid?


def _apply_function(x,f): #do we have apply_function already?
#HAD TO PUT THIS BELOW GRID CLASS, BC ARCPIXELWISEPRIMITIVES RELIES ON THE GRID CLASS AS WELL
    return f(x)


    #WHAT SHOULD I DO FOR THE TYPING OF APPLY_FUNCTION?
    #do i use tlist even if im returning an array? for example return subgrids, grid_split_2d (2d not even1d)
    #stack function is defunct
    #no task testing things
    #is my current file structure ok in arcPixelwisePrimitives? will importing grid class and stuf be fine, do i have to declare?
