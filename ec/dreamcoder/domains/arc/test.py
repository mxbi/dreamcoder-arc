import datetime
import os
import random

import binutil

from dreamcoder.dreamcoder import commandlineArguments, ecIterator
from dreamcoder.grammar import Grammar
from dreamcoder.program import Primitive
from dreamcoder.task import Task
from dreamcoder.type import arrow, tint
from dreamcoder.utilities import numberOfCPUs

from dreamcoder.domains.arc.arcPrimitives import *
from dreamcoder.domains.arc.makeTasks import get_arc_task
import dreamcoder.domains.arc.arcPrimitives as p
from dreamcoder.domains.arc.main import ArcNet
from dreamcoder.domains.arc.arcPrimitives import primitive_dict as pd
from dreamcoder.dreamcoder import commandlineArguments, ecIterator

def test_tile():
    print(p.tile_grid(np.array([[1,2,3],[4,5,6]]), 0, 1, (15, 15)))

def test_tile_to_fill():
    task = get_arc_task(109)
    # for ex in range(len(task.examples)):
    i, o = task.examples[1][0][0].input_grid, task.examples[1][1]
    print(p.num_mistakes(6, 7, 1, 0, i.grid, 0))
    assert False
    predicted = p._tile_to_fill2(i)(0)
    if not p._equals_exact(predicted)(o):
        print('i: {}'.format(i))
        print('o: {}'.format(o))
        print('predicted: {}'.format(predicted))
        print(predicted.grid*(predicted.grid != o.grid))
        assert False
    else:
        print('passed!')


def test_construct_mapping():
    primitives = [
            pd['objects2'],
            pd['T'], pd['F'],
            pd['input'],
            pd['rotation_invariant'],
            pd['size_invariant'],
            pd['color_invariant'],
            pd['no_invariant'],
            # pd['place_into_input_grid'],
            pd['place_into_grid'],
            pd['rows'],
            pd['columns'],
            # pd['output'],
            # pd['area'],
            pd['construct_mapping'],
            pd['vstack'],
            pd['hstack'],
            # pd['construct_mapping2'],
            # pd['construct_mapping3'],
    ]

    grammar = Grammar.uniform(primitives)


    # generic command line options
    args = commandlineArguments(
        # enumerationTimeout=60, 
        enumerationTimeout=300,
        # activation='tanh',
        aic=.1, # LOWER THAN USUAL, to incentivize making primitives
        iterations=1, 
        # recognitionTimeout=120, 
        # featureExtractor=ArcNet,
        a=3, 
        maximumFrontier=10, 
        topK=1, 
        pseudoCounts=30.0,
        # helmholtzRatio=0.5, 
        # structurePenalty=.1, # HIGHER THAN USUAL, to incentivize making primitives
        solver='python',
        CPUs=1,
        no_consolidation=True,
        )

    copy_one_tasks = [11, 14, 15, 80, 81, 94, 159, 281, 316, 330, 72, 261, 301, 234]
    training = [get_arc_task(i) for i in copy_one_tasks]

    # iterate over wake and sleep cycles for our task
    generator = ecIterator(grammar,
                           training,
                           testingTasks=[],
                           outputPrefix='./experimentOutputs/arc/',
                           **args)

    for i, result in enumerate(generator):
        print('ecIterator count {}'.format(i))

    print('should have solved all {} tasks'.format(len(copy_one_tasks)))


def test_construct_mapping2():
    primitives = [
        pd['objects2'],
        pd['T'], pd['F'],
        pd['input'],
        pd['rotation_invariant'],
        pd['size_invariant'],
        pd['color_invariant'],
        pd['no_invariant'],
        pd['place_into_input_grid'],
        pd['place_into_grid'],
        pd['get_first'],
        # pd['rows'],
        # pd['columns'],
        # pd['output'],
        # pd['size'],
        # pd['area'],
        pd['construct_mapping'],
        # pd['vstack'],
        # pd['hstack'],
        # pd['construct_mapping2'],
        pd['construct_mapping3'],
        pd['list_of_one'],
        pd['area'],
        pd['has_y_symmetry'],
        pd['list_length'],
        pd['filter_list'],
        pd['contains_color'],
        pd['color2'],
    ]

    grammar = Grammar.uniform(primitives)


    # generic command line options
    args = commandlineArguments(
        enumerationTimeout=100, 
        # activation='tanh',
        aic=.1, # LOWER THAN USUAL, to incentivize making primitives
        iterations=1, 
        # recognitionTimeout=120, 
        # featureExtractor=ArcNet,
        a=3, 
        maximumFrontier=10, 
        topK=1, 
        pseudoCounts=30.0,
        # helmholtzRatio=0.5, 
        # structurePenalty=.1, # HIGHER THAN USUAL, to incentivize making primitives
        solver='python',
        CPUs=1,
        no_consolidation=True,
        )

    copy_two_tasks = [103, 55, 166, 47, 185, 398, 102, 297, 352, 372]
    training = [get_arc_task(i) for i in copy_two_tasks]

    # iterate over wake and sleep cycles for our task
    generator = ecIterator(grammar,
                           training,
                           testingTasks=[],
                           outputPrefix='./experimentOutputs/arc/',
                           **args)

    for i, result in enumerate(generator):
        print('ecIterator count {}'.format(i))

    print('should have solved all {} tasks'.format(len(copy_two_tasks)))

def test_inflate():
    primitives = [
            # pd['objects2'],
            # pd['T'], pd['F'],
            pd['input'],
            pd['object'],
            # pd['rotation_invariant'],
            # pd['size_invariant'],
            # pd['color_invariant'],
            # pd['no_invariant'],
            # pd['place_into_input_grid'],
            # pd['place_into_grid'],
            # pd['rows'],
            # pd['columns'],
            # pd['output'],
            # pd['size'],
            # pd['area'],
            # pd['construct_mapping'],
            # pd['vstack'],
            # pd['hstack'],
            # pd['construct_mapping2'],
            # pd['construct_mapping3'],
            pd['area'],
            pd['kronecker'],
            pd['inflate'],
            pd['deflate'],
            pd['2'],
            pd['3'],
            pd['num_colors'],
    ]

    grammar = Grammar.uniform(primitives)

    # generic command line options
    args = commandlineArguments(
        enumerationTimeout=300, 
        # activation='tanh',
        aic=.1, # LOWER THAN USUAL, to incentivize making primitives
        iterations=1, 
        # recognitionTimeout=120, 
        # featureExtractor=ArcNet,
        a=3, 
        maximumFrontier=10, 
        topK=1, 
        pseudoCounts=30.0,
        # helmholtzRatio=0.5, 
        # structurePenalty=.1, # HIGHER THAN USUAL, to incentivize making primitives
        solver='python',
        CPUs=1,
        no_consolidation=True,
        )

    inflate_tasks = [0, 194, 216, 222, 268, 288, 306, 383]
    training = [get_arc_task(i) for i in inflate_tasks]

    # iterate over wake and sleep cycles for our task
    generator = ecIterator(grammar,
                           training,
                           testingTasks=[],
                           outputPrefix='./experimentOutputs/arc/',
                           **args)

    for i, result in enumerate(generator):
        print('ecIterator count {}'.format(i))

    print('should have solved all {} tasks'.format(len(inflate_tasks)))


def test_symmetry():
    primitives = [
        pd['object'],
        pd['x_mirror'],
        pd['y_mirror'],
        pd['rotate_cw'],
        pd['rotate_ccw'],
        pd['left_half'],
        pd['right_half'], 
        pd['top_half'],
        pd['bottom_half'],
        pd['overlay'],
        pd['combine_grids_vertically'],
        pd['combine_grids_horizontally'], 
        pd['input'],
        # pd['output'],
    ]

    grammar = Grammar.uniform(primitives)

    # generic command line options
    args = commandlineArguments(
        enumerationTimeout=300, 
        # activation='tanh',
        aic=.1, # LOWER THAN USUAL, to incentivize making primitives
        iterations=1, 
        # recognitionTimeout=120, 
        # featureExtractor=ArcNet,
        a=3, 
        maximumFrontier=10, 
        topK=1, 
        pseudoCounts=30.0,
        # helmholtzRatio=0.5, 
        # structurePenalty=.1, # HIGHER THAN USUAL, to incentivize making primitives
        solver='python',
        CPUs=1,
        no_consolidation=True,
        )

    # symmetry_tasks = [30, 38, 52, 56, 66, 70, 82, 86, 105, 108, 112, 115, 116, 139, 141, 149, 151, 154, 163, 171, 176, 178, 179, 209, 210, 240, 241, 243, 248, 310, 346, 359, 360, 379, 371, 384]
    symmetry_tasks = [30, 38, 86, 112, 115, 139, 149, 154, 163, 171, 176, 178, 209, 240, 248, 310, 359, 379, 384]

    training = [get_arc_task(i) for i in symmetry_tasks]

    # iterate over wake and sleep cycles for our task
    generator = ecIterator(grammar,
                           training,
                           testingTasks=[],
                           outputPrefix='./experimentOutputs/arc/',
                           **args)

    for i, result in enumerate(generator):
        print('ecIterator count {}'.format(i))

    print('should have solved all {} tasks'.format(len(symmetry_tasks)))

def test_helmholtz():
    symmetry_tasks = [30, 38, 86, 112, 115, 139, 149, 154, 163, 171, 176, 178, 209, 240, 248, 310, 359, 379, 384]
    copy_one_tasks = [11, 14, 15, 80, 81, 94, 159, 281, 316, 330, 72, 261, 301, 234]
    copy_two_tasks = [103, 55, 166, 47, 185, 398, 102, 297, 352, 372]
    inflate_tasks = [0, 194, 216, 222, 268, 288, 306, 383]
    tasks = symmetry_tasks + copy_one_tasks + copy_two_tasks + inflate_tasks
    tasks = list(set(tasks))

    symmetry_primitives = [
        pd['object'],
        pd['x_mirror'],
        pd['y_mirror'],
        pd['rotate_cw'],
        pd['rotate_ccw'],
        pd['left_half'],
        pd['right_half'], 
        pd['top_half'],
        pd['bottom_half'],
        pd['overlay'],
        pd['combine_grids_vertically'],
        pd['combine_grids_horizontally'], 
        pd['input'],
        # pd['output'],
    ]

    inflate_primitives = [
            pd['input'],
            pd['object'],
            pd['area'],
            pd['kronecker'],
            pd['inflate'],
            pd['deflate'],
            pd['2'],
            pd['3'],
            pd['num_colors'],
    ]

    copy_two_primitives = [
        pd['objects2'],
        pd['T'], pd['F'],
        pd['input'],
        pd['rotation_invariant'],
        pd['size_invariant'],
        pd['color_invariant'],
        pd['no_invariant'],
        pd['place_into_input_grid'],
        pd['place_into_grid'],
        pd['construct_mapping'],
        pd['construct_mapping3'],
        pd['get_first'],
        pd['list_of_one'],
        pd['area'],
        pd['has_y_symmetry'],
        pd['list_length'],
        pd['filter_list'],
        pd['contains_color'],
        pd['color2'],
    ]

    copy_primitives = [
            pd['objects2'],
            pd['T'], pd['F'],
            pd['input'],
            pd['rotation_invariant'],
            pd['size_invariant'],
            pd['color_invariant'],
            pd['no_invariant'],
            pd['place_into_grid'],
            pd['rows'],
            pd['columns'],
            pd['construct_mapping'],
            pd['vstack'],
            pd['hstack'],
    ]

    primitives = copy_primitives + copy_two_primitives + inflate_primitives + symmetry_primitives
    primitives = list(set(primitives))

    grammar = Grammar.uniform(primitives)

    training = [get_arc_task(i) for i in tasks]

    # generic command line options
    args = commandlineArguments(
        enumerationTimeout=100, 
        # activation='tanh',
        # aic=.1, # LOWER THAN USUAL, to incentivize making primitives
        iterations=20, 
        recognitionTimeout=3600, 
        featureExtractor=ArcNet,
        auxiliary=True, # train our feature extractor too
        contextual=True, # use bi-gram model, not unigram
        a=3, 
        maximumFrontier=10, 
        topK=1, 
        pseudoCounts=30.0,
        helmholtzRatio=1.0,  # percent that are random programs
        # structurePenalty=.1, # HIGHER THAN USUAL, to incentivize making primitives
        solver='python',
        CPUs=15,
        no_consolidation=True,
        )

    # iterate over wake and sleep cycles for our task
    generator = ecIterator(grammar,
                           training,
                           testingTasks=[],
                           outputPrefix='./experimentOutputs/arc/',
                           **args)

    for i, result in enumerate(generator):
        print('ecIterator count {}'.format(i))



    



def test():
    # test_tile()
    # test_tile_to_fill()
    test_helmholtz()
    # test_construct_mapping()
    # test_construct_mapping2()
    # test_inflate()
    # test_symmetry()
