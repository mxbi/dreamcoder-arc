import dreamcoder.domains.arc.arcPrimitives as p
from dreamcoder.domains.arc.makeTasks import get_arc_task
# from dreamcoder.domains.arc.arcInput import load_task

def check_solves(task, program):
    for i, ex in enumerate(task.examples):
        inp, out = ex[0][0], ex[1]
        predicted = program(inp)
        if predicted != out:
            print('didnt solve: {}'.format(task.name))
            print('Failed example ' + str(i) + ': input=')
            print(p._input(inp))
            print('output=')
            print(out)
            print('predicted=')
            print(predicted)
            # assert False, 'did NOT pass!'
            print('Did not pass')
            return
    print('Passed {}'.format(task.name))

def task128():
    """
    Color the whole grid the color of most frequent color
    """
    task = get_arc_task(128)

    def program(i):
        g = p._input(i)
        # c = p._color(p._get_last(p._sort(
        #         p._map(p._colors(g))(p._filter_color(g)) # color separated grids
        #     )(p._area)))
        out = p._flood_fill(g)(p._color(g))
        return out

        # g = p._input(i)
        # c = p._color(p._get_last(p._sort(
        #         p._map(p._colors(g))(p._filter_color(g)) # color separated grids
        #     )(p._area)))
        # out = p._flood_fill(g)(c)
        # return out

    check_solves(task, program)



def task140():
    """
    From a point, draw diagonal lines in all four directions, the same color as that point
    """
    task = get_arc_task(140)

    def program(i):
        # draw line slant up should take the object type
        o = p._get(p._objects(p._input(i)))(0)
        return p._color_in_grid(
                p._overlay (p._draw_line_slant_up(p._input(i))(o)) (p._draw_line_slant_down(p._input(i))(o))
            )(p._color(p._input(i)))
                
        # return p._color_in_grid(
        #     p._map(p._draw_line(p._input(i))(p._get(p._objects(p._input(i)))(0)),
        #         [315,45]
        #         )(p._color(p._get(p._objects(p._input(i)))(0)))

        # return p._color_in_grid(
        #     p._draw_line(
        #             p._draw_line(p._input(i))(p._get(p._objects(p._input(i)))(0))(315)
        #         )(p._get(p._objects(p._input(i)))(0))(45)
        #     )(p._color(p._get(p._objects(p._input(i)))(0)))

        # return p._color_in_grid(
        #     p._overlay(
        #             p._draw_line(p._input(i))(p._get(p._objects(p._input(i)))(0))(45)
        #         )(
        #             p._draw_line(p._input(i))(p._get(p._objects(p._input(i)))(0))(315)
        #         )
        #     )(p._color(p._get(p._objects(p._input(i)))(0)))


        # o = p._get(p._objects(p._input(i)))(0) # get first object
        # line1 = p._draw_line(p._input(i))(o)(45) # draw first line
        # line2 = p._draw_line(p._input(i))(o)(315) # draw second line
        # bothlines = p._overlay(line1)(line2) # stack the lines
        # return p._color_in_grid(bothlines)(p._color(o)) # color them

    check_solves(task, program)




def task36():
    """
    connect points of same color
    """

    task = get_arc_task(36)

    def program(i):

        return p._stack_no_crop(
                p._map  
                ( p._draw_connecting_line(p._input(i)) ) 
                ( p._group_objects_by_color(p._input(i)) )
            )

    check_solves(task, program)

    



def run():
    # task128()
    task140()
    task36()
