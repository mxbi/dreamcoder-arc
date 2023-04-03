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

from dreamcoder.domains.arc.makeTasks import get_arc_task, get_arc_train_tasks
from dreamcoder.domains.arc.main import ArcNet

from dreamcoder.domains.arc.arcPrimitivesMikel import *
from dreamcoder.domains.arc.arcPrimitivesMikel import primitive_dict as p
from dreamcoder.domains.arc.arcPrimitivesMikel import generate_ocaml_primitives

generate_ocaml_primitives()

# set the primitives to work with
p = primitive_dict#.values()
primitives = p.values()
# primitives = [
#         p['object'],
#         p['x_mirror'],
#         # p['y_mirror'],
#         p['rotate_cw'],
#         # p['rotate_ccw'],
#         p['left_half'],
#         p['overlay'],
#         p['combine_grids_vertically'],
#     ]

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
    compressor='ocaml',
    CPUs=48,
    )

# symmetry_tasks = range(400) #[30, 38, 52, 56, 66, 70, 82, 86, 105, 108, 112, 115, 116, 139, 141, 149, 151, 154, 163, 171, 176, 178, 179, 209, 210, 240, 241, 243, 248, 310, 345, 359, 360, 379, 371, 384]
# training = [(i) for i in symmetry_tasks]
training = get_arc_train_tasks()

# iterate over wake and sleep cycles for our task
generator = ecIterator(grammar,
                       training,
                       testingTasks=[],
                       outputPrefix='./experimentOutputs/arc/',
                       **args)

# run the DreamCoder learning process for the set number of iterations
for i, result in enumerate(generator):
    print('ecIterator count {}'.format(i))

# consolidation_dict = make_consolidation_dict(result)
# export_dc_demo('/home/salford/to_copy/arc_demo_9.json', training, consolidation_dict)

