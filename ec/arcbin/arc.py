import datetime
import os
import random
import numpy as np

import binutil

import arckit

print(arckit.load_data())

from dreamcoder.ec import commandlineArguments, ecIterator
from dreamcoder.grammar import Grammar
from dreamcoder.program import Primitive
from dreamcoder.task import Task
from dreamcoder.type import arrow, tint
from dreamcoder.utilities import numberOfCPUs

args = commandlineArguments(
    enumerationTimeout=10, activation='tanh',
    iterations=10, recognitionTimeout=3600,
    a=3, maximumFrontier=10, topK=2, pseudoCounts=30.0,
    helmholtzRatio=0.5, structurePenalty=1.,
    CPUs=numberOfCPUs())

def _rot90(x): return lambda x: np.rot90(x)

primitives = [
    Primitive("rot90", arrow(tint, tint), _rot90),
]