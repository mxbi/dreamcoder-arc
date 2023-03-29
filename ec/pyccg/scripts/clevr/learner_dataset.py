#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : learner_dataset.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 01/16/2019
#
# Distributed under terms of the MIT license.

import random

from pyccg.chart import WeightedCCGChartParser, printCCGDerivation
from pyccg.model import Model
from pyccg.lexicon import Lexicon
from pyccg.logic import TypeSystem, Ontology, Expression
from pyccg.word_learner import WordLearner


########
# Ontology: defines a type system, constants, and predicates available for use
# in logical forms.


class Object(object):
  def __init__(self, shape, size, material):
    self.shape = shape
    self.size = size
    self.material = material

  def __hash__(self):
    return hash((self.shape, self.size, self.material))

  def __eq__(self, other):
    return other.__class__ == self.__class__ and hash(self) == hash(other)

  def __str__(self):
    return "Object(%s, %s, %s)" % (self.shape, self.size, self.material)


def fn_unique(xs):
  true_xs = [x for x, matches in xs.items() if matches]
  assert len(true_xs) == 1
  return true_xs[0]


def fn_exists(xs):
  true_xs = [x for x, matches in xs.items() if matches]
  return len(true_xs) > 0


types = TypeSystem(["object", "boolean", "shape", "size"])

functions = [
  types.new_function("has_shape", ("object", "shape", "boolean"), lambda x, s: x.shape == s),
  types.new_function("unique", (("object", "boolean"), "object"), fn_unique),
  types.new_function("object_exists", (("object", "boolean"), "boolean"), fn_exists),
]

constants = [
  types.new_constant("sphere", "shape"),
  types.new_constant("cube", "shape"),
  types.new_constant("cylinder", "shape"),
  types.new_constant("true", "boolean"),
]

ontology = Ontology(types, functions, constants)


#######
# Lexicon: defines an initial set of word -> (syntax, meaning) mappings.
# Weights are initialized uniformly by default.


initial_lex = Lexicon.fromstring(r"""
  :- S, N

  any => S/N {\x.object_exists(x)}
  _dummy_noun => N {\x.true}
""", ontology, include_semantics=True)


#######
# VQA Dataset: defines the dataset.


class VQADataset(object):
    """
    A dummy dataset contains tuples of (scene, question, answer).
    Each scene contains only one objects with one of the three shapes ('sphere', 'cube' and 'cylinder').
    There are three types of questions: "any sphere", "any cube", "any cylinder".
    The answer is True if the shape of interest in the question match the shape of the visual object.
    """
    def __init__(self, dataset_size):
        super().__init__()
        self.dataset_size = dataset_size
        self.data = list()
        self._gen_data()

    def _gen_data(self):
        def gen_shape():
            return random.choice(['sphere', 'cube', 'cylinder'])

        for i in range(self.dataset_size):
            scene = dict(objects=[
                Object(gen_shape(), 'big', 'rubber')
            ])

            soi = gen_shape()  # shape-of-interest
            question = 'any ' + soi
            answer = scene['objects'][0].shape == soi

            self.data.append((scene, question, answer))

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


def learn_dataset(initial_lex, dataset_size):
    """"Learn the lexicon from a randomly generated data."""
    dataset = VQADataset(dataset_size)

    def iter_data():
        """Helper function for iterating dataset and inject the ontology to make the scene a model for execution."""
        for v, q, a in dataset:
            yield (Model(v, ontology), q.split(), a)

    learner = WordLearner(initial_lex, update_perceptron_algo='reinforce')
    try:
        for v, q, a in iter_data():
            learner.update_with_distant(q, v, a)
    except Exception as e:
        print(e)
        raise e
    finally:
        del iter_data

    return learner.lexicon


lex = learn_dataset(initial_lex, 1000)

print('Learned lexicon:')
print('-' * 120)
lex.debug_print()
print()

print('Example:')
print('-' * 120)
parser = WeightedCCGChartParser(lex)
results = parser.parse("any cube".split())
printCCGDerivation(results[0])

root_token, _ = results[0].label()
print(root_token.semantics())

