"""
Basic example of CCG learning in a CLEVR-like domain.
"""

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


types = TypeSystem(["object", "boolean", "shape", "size"])

functions = [
  types.new_function("has_shape", ("object", "shape", "boolean"), lambda x, s: x.shape == s),

  types.new_function("unique", (("object", "boolean"), "object"), fn_unique),
]

constants = [
  types.new_constant("sphere", "shape"),
  types.new_constant("cube", "shape"),
  types.new_constant("cylinder", "shape"),
]

ontology = Ontology(types, functions, constants)


#######
# Lexicon: defines an initial set of word -> (syntax, meaning) mappings.
# Weights are initialized uniformly by default.

lex = Lexicon.fromstring(r"""
  :- N

  the => N/N {\x.unique(x)}
  ball => N {\x.has_shape(x,sphere)}
""", ontology, include_semantics=True)


#######
# Execute on a scene.

scene = {
  "objects": [
    Object("sphere", "big", "rubber"),
    Object("cube", "small", "metal"),
    Object("cylinder", "small", "rubber"),
  ]
}

model = Model(scene, ontology)
print("the ball")
print(model.evaluate(Expression.fromstring(r"unique(\x.has_shape(x,sphere))")))

######
# Parse an utterance and execute.

learner = WordLearner(lex)

# Update with distant supervision.
learner.update_with_distant("the cube".split(), model, scene['objects'][1])

parser = learner.make_parser()
results = parser.parse("the cube".split())
printCCGDerivation(results[0])

root_token, _ = results[0].label()
print(root_token.semantics())
print(model.evaluate(root_token.semantics()))
