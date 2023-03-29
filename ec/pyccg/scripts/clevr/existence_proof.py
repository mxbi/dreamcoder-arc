#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demonstrates that CLEVR sentences are actually parse-able under a CCG
syntax.
"""

import random

import nose
from nose.tools import *

from pyccg.chart import WeightedCCGChartParser, printCCGDerivation
from pyccg.model import Model
from pyccg.lexicon import Lexicon
from pyccg.logic import TypeSystem, Ontology, Expression
from pyccg.word_learner import WordLearner


########
# Ontology: defines a type system, constants, and predicates available for use
# in logical forms.


dummy_fn_1 = lambda x: x
dummy_fn_2 = lambda x, y: x

types = TypeSystem(["object", "boolean", "shape", "size"])

functions = [
  types.new_function("has_shape", ("object", "shape", "boolean"), dummy_fn_2),
  types.new_function("unique", (("object", "boolean"), "object"), dummy_fn_1),
  types.new_function("object_exists", (("object", "boolean"), "boolean"), dummy_fn_1),
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


lexicon = Lexicon.fromstring(r"""
  :- S, N

  the => S/N {\x.unique(x)}
  the => N/N {\x.unique(x)}
  the => S/N {\x.x}
  the => N/N {\x.x}

  object => N {scene}
  objects => N {scene}

  metallic => N/N {\x.filter(material,x,metal)}
  shiny => N/N {\x.filter(material,x,metal)}

  big => N/N {\x.filter(size,x,large)}

  purple => N/N {\x.filter(color,x,purple)}

  material => N {material}
  shape => N {shape}
  color => N {color}
  size => N {size}

  same => N/N/N {\a o.same(a,o)}
  as => N/N {\x.x}

  with => S\N/N {\p x.filter_(x,p)}
  of => N\N/N {\o a.query(a,o)}
  that => N\N/S {\p x.filter_(x,p)}

  how_many => S/S/N {\x p.count(x,p)}
  what_number_of => S/S/N {\x p.count(x,p)}

  what => S/S {\x.x}
  is => S/N {\x.x}
  is => S/S {\x.x}
  are => S/N {\x.x}
  are => S/S {\x.x}

  # TODO this is wrong -- actually a very complicated and interesting operator..
  # Need to rule out the particular object given in the complement
  # in "what number of other objects are the same size as the purple shiny object"
  other => N/N {\x.x}
""", ontology, include_semantics=True)


#######
# Define test cases.

sentences = [
  ("the big metallic object",
   r"unique(filter(size,filter(material,scene,metal),large))"),

  ("the same shape as the big metallic object",
   r"same(shape,unique(filter(size,filter(material,scene,metal),large)))"),

  ("objects with the same shape as the big metallic object",
   r"filter_(scene,same(shape,unique(filter(size,filter(material,scene,metal),large))))"),

  ("what is the material of the big purple object",
   r"query(material,unique(filter(size,filter(color,scene,purple),large)))"),

  ("what is the object that is the same size as the purple shiny object",
   r"unique(filter_(scene,same(size,unique(filter(color,filter(material,scene,metal),purple)))))"),

  ("what is the shape of the object that is the same size as the purple shiny object",
   r"query(shape,unique(filter_(scene,same(size,unique(filter(color,filter(material,scene,metal),purple))))))"),

  ("how_many objects are the same size as the purple shiny object",
   r"count(scene,same(size,unique(filter(color,filter(material,scene,metal),purple))))"),

  ("what_number_of objects are the same size as the purple shiny object",
   r"count(scene,same(size,unique(filter(color,filter(material,scene,metal),purple))))"),

]


######
# Define tests.

def _test_case(parser, sentence, lf, scene=None):
  model = Model(scene or {"objects": []}, ontology)
  results = parser.parse(sentence.split())
  for result in results:
    semantics = result.label()[0].semantics()
    print("Predicted semantics:", semantics)
    if str(semantics) == lf:
      printCCGDerivation(result)
      return True
  ok_(False, "Expected successful parse with correct lf. Found %i parses; none matched LF." % len(results))


def test_cases():
  parser = WeightedCCGChartParser(lexicon)
  for sentence, lf in sentences:
    yield _test_case, parser, sentence, lf


if __name__ == '__main__':
  nose.main(defaultTest=__file__)
