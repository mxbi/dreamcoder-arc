
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

from dreamcoder.domains.arc.arcInput import export_tasks
from dreamcoder.domains.arc.arcInput import export_dc_demo, make_consolidation_dict
from dreamcoder.domains.arc.makeTasks import get_arc_task
from dreamcoder.domains.arc.main import ArcNet

from dreamcoder.domains.arc.arcPrimitives import *
from dreamcoder.domains.arc.arcPrimitives import primitive_dict as p
from dreamcoder.domains.arc.arcPrimitives import generate_ocaml_primitives

from dreamcoder.domains.arc.recognition_test import run_shuffle

# set the primitives to work with
primitives = [
        p['object'],
        p['x_mirror'],
        # p['y_mirror'],
        p['rotate_cw'],
        # p['rotate_ccw'],
        p['left_half'],
        # p['right_half'], 
        # p['top_half'],
        # p['bottom_half'],
        p['overlay'],
        p['combine_grids_vertically'],
        # p['combine_grids_horizontally'], 
        p['input'],
    ]

# make a starting grammar to enumerate over
grammar = Grammar.uniform(primitives)

# generic command line options
args = commandlineArguments(
    enumerationTimeout=10, 
    aic=0.1,
    iterations=1, 
    recognitionTimeout=120, 
    # featureExtractor=ArcNet,
    a=3, 
    maximumFrontier=10, 
    topK=5, 
    pseudoCounts=30.0,
    structurePenalty=0.1,
    solver='python',
    CPUs=48,
    )

symmetry_tasks = [30, 38, 52, 56, 66, 70, 82, 86, 105, 108, 112, 115, 116, 139, 141, 149, 151, 154, 163, 171, 176, 178, 179, 209, 210, 240, 241, 243, 248, 310, 345, 359, 360, 379, 371, 384]
training = [get_arc_task(i) for i in symmetry_tasks]

# task = training[0]
# task.examples = [(x, x) for x, y in task.examples]
# training[0] = task

# iterate over wake and sleep cycles for our task
generator = ecIterator(grammar,
                       training,
                       testingTasks=[],
                       outputPrefix='./experimentOutputs/arc/',
                       **args)

# run the DreamCoder learning process for the set number of iterations
for i, result in enumerate(generator):
    print('ecIterator count {}'.format(i))