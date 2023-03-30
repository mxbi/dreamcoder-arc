import numpy as np
from dreamcoder.task import Task
from dreamcoder.type import arrow
import dreamcoder.domains.arc.arcPrimitives as p
from dreamcoder.domains.arc.arcPrimitives import Grid, Input, tgrid, tinput, toutput
from dreamcoder.type import arrow, baseType, tint, tlist, t0, t1, t2, tbool
from dreamcoder.domains.arc.arcPrimitives import _map_i_to_j, _get, _list_of, _pixels, _objects, _stack_overlay
from dreamcoder.domains.arc.arcInput import export_tasks
from dreamcoder.program import Primitive
from dreamcoder.grammar import Grammar
from dreamcoder.dreamcoder import commandlineArguments, ecIterator


def shuffle_task(seed=0, n=6):
    lengths = np.random.randint(1,high=n+1, size=n)
    sorted_lengths = sorted(lengths, key=lambda l: l)
    input_rows = []
    output_rows = []
    for l1, l2, in zip(lengths, sorted_lengths):
        input_rows.append(np.full((1,n), fill_value=0))
        output_rows.append(np.full((1,n), fill_value=0))

        color = min(l1, 10)
        a = np.full((1,n), fill_value=0)
        a[0,:l1] = np.full(l1, fill_value=color)
        input_rows.append(a)

        color = min(l2, 10)
        a = np.full((1,n), fill_value=0)
        a[0,:l2] = np.full(l2, fill_value=color)
        output_rows.append(a)

    input_rows.append(np.full((1,n), fill_value=0))
    output_rows.append(np.full((1,n), fill_value=0))

    inp = np.concatenate(input_rows)
    outp = np.concatenate(output_rows)

    return Grid(inp), Grid(outp), lengths




def task1(seed=0):
    rng = np.random.default_rng(seed)
    inp = rng.integers(0, high=9, size=(20, 20))
    i1, o1, i2, o2 = rng.integers(0, high=5, size=4)
    # print('{} -> {}, {} -> {}'.format(i1, o1, i2, o2))
    outp = _map_i_to_j(Grid(inp))(i1)(o1)
    outp = _map_i_to_j(outp)(i2)(o2)
    outp = outp.grid

    examples = [{'input': inp, 'output': outp}]
    examples = [((Input(ex["input"], examples),),
        Grid(ex["output"])) for ex in examples]

    task = Task('task1_' + str(seed),
            arrow(tinput, tgrid),
            examples)
    return task

def task2(seed=0):
    obj = np.array([[1,1,1],[1,1,1],[1,1,1]])
    num_objects = 10
    grid = np.zeros((20, 20))
    rng = np.random.default_rng(seed)

    for i in range(num_objects):
        x, y = rng.integers(0, high=grid.shape[0]-3, size=2)
        grid[x:x+obj.shape[0], y:y+obj.shape[1]] += obj

    inp = grid
    succeeded = False
    attempts = 0
    while not succeeded:
        try:
            attempts += 1
            # high controls how hard enumeration is.
            o1, i1, o2, i2 = rng.integers(0, high=5, size=4)
            # print('o1, i1, o2, i2: {}'.format((o1, i1, o2, i2)))
            obj1 = _get(_objects(Grid(inp)))(o1)
            # obj2 = _get(_objects(Grid(inp)))(o2)
            pix1 = _get(_pixels(obj1))(i1)
            # pix2 = _get(_pixels(obj2))(i2)
            # outp = _stack_no_crop(_list_of(obj1)(obj2))
            outp = _absolute_grid(pix1)
            outp = outp.grid
            succeeded = True
        except ValueError:
            # print('attempts: {}'.format(attempts))
            succeeded = False

    examples = [{'input': inp, 'output': outp}]
    examples = [((Input(ex["input"], examples),),
        Grid(ex["output"])) for ex in examples]

    task = Task('task2_' + str(seed),
            arrow(tinput, tgrid),
            examples)
    return task


def _order(n, prev=None):
    if prev is None:
        prev = []
    if n == 1:
        return lambda x: prev + [x]
    else:
        return lambda x: _order(n-1, prev=prev + [x])

def _arrange(objects):
    def arrange(objects1, order):
        # if given same key, put larger obj first so its wrong
        return [x for _, x in sorted(zip(order, objects1), key=lambda t: (t[0],
            -np.sum(t[1].grid)))]
    return lambda order: arrange(objects, order)


def _reassemble(objects):
    lengths = [p._area(o) for o in objects]
    colors = [p._color(o) for o in objects]
    n = len(objects)
    rows = []
    for l, c in zip(lengths, colors):
        rows.append(np.full((1,n), fill_value=0))

        a = np.full((1,n), fill_value=0)
        a[0,:l] = np.full(l, fill_value=c)
        rows.append(a)

    rows.append(np.full((1,n), fill_value=0))

    grid = np.concatenate(rows)
    return Grid(grid)


def shuffle_solution(lengths):
    n = len(lengths)
    def program(i):
        objects = p._objects(i)
        order = _order(n)
        # feed order into the lambda function.
        # after this, order is a list
        for l in lengths:
            order = order(l)
        # order = [l for l in lengths]
        return _reassemble(_arrange(objects)(order))

    return program


def check_solves(program, input_grid, output_grid):
    prediction = program(input_grid)
    if p._equals_exact(prediction)(output_grid):
        print('solved')
    else:
        print('input_grid: {}'.format(input_grid))
        print('output_grid: {}'.format(output_grid))
        print('predicted: {}'.format(prediction))
        assert False





def run():
    tasks = [task1(), task2()]
    export_tasks('/home/salford/to_copy/', tasks)

def shuffle_tasks(n, num_tasks):
    examples = [shuffle_task(n=n) for i in range(num_tasks)]
    tasks = [Task(' '.join(str(l) for l in ls),
        arrow(tinput, toutput),
        [((Input(i.grid, []),), Grid(o.grid))] )
        for ix, (i, o, ls) in enumerate(examples)]

    # for t in tasks:
        # t.n = n

    # for t, _, (_, _, lengths) in enumerate(zip(tasks, examples)):
        # t.numbers = lengths

    return tasks

def task_size(): # hack so main.py can access for generating training examples
    return 6

def test_shuffle():
    for n in range(10):
        for i in range(10):
            inp, outp, lengths = shuffle_task(n=5)
            program = shuffle_solution(lengths)
            check_solves(program, inp, outp)


def run_shuffle():
    # test_shuffle()
    # if we put this at the top we get a circular dependency import error
    from dreamcoder.domains.arc.main import ArcNet2
    n = task_size()
    num_tasks = 50
    tasks = shuffle_tasks(n, num_tasks)

    primitives = [
            Primitive("input", arrow(tinput, tgrid), lambda i: i.input_grid),
            Primitive("objects", arrow(tgrid, tlist(tgrid)), p._objects),
            Primitive("order", arrow(*([tint] * n + [tlist(tint)])), _order(n)),
            Primitive("arrange", arrow(tlist(tgrid), tlist(tint), tlist(tgrid)),
                _arrange),
            Primitive("reassemble", arrow(tlist(tgrid), toutput), _reassemble),
            *[Primitive(str(i), tint, i) for i in range(n)],
            ]

    grammar = Grammar.uniform(primitives)

    # generic command line options
    args = commandlineArguments(
        enumerationTimeout=60,
        # activation='tanh',
        aic=.1, # LOWER THAN USUAL, to incentivize making primitives
        iterations=5,
        recognitionTimeout=440,
        featureExtractor=ArcNet2,
        a=3,
        maximumFrontier=10,
        topK=1,
        pseudoCounts=30.0,
        # helmholtzRatio=0.5,
        # structurePenalty=.1, # HIGHER THAN USUAL, to incentivize making primitives
        solver='python',
        auxiliary=True,
        contextual=True,
        CPUs=5
        )

    # export_tasks('/home/salford/to_copy/', tasks)

    # iterate over wake and sleep cycles for our task
    generator = ecIterator(grammar,
                           tasks,
                           testingTasks=[],
                           **args)
    for i, _ in enumerate(generator):
        print('ecIterator count {}'.format(i))
