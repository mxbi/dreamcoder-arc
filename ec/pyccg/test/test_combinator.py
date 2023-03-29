from nose.tools import *

from pyccg.combinator import *
from pyccg.lexicon import Lexicon, Token
from pyccg.logic import TypeSystem, Ontology


def _make_mock_lexicon():
  types = TypeSystem(["obj", "boolean"])
  functions = [
    types.new_function("unique", (("obj", "boolean"), "obj"), lambda x: x[0]),
    types.new_function("twoplace", ("boolean", ("obj", "boolean"), "obj"), lambda a, b: b[0]),
    types.new_function("dog", ("obj", "boolean"), lambda x: x["dog"]),
    types.new_function("not_", ("boolean", "boolean"), lambda a: not a),
    types.new_function("enlarge", ("obj", "obj"), lambda x: x),
  ]
  constants = [types.new_constant("true", "boolean")]
  ontology = Ontology(types, functions, constants)

  lex = Lexicon.fromstring(r"""
    :- S, N

    the => N/N {\x.unique(x)}
    thee => N\N {\x.unique(x)}
    twoplace => N/N {\x.twoplace(true,x)}
    twoplacee => N\N {\x.twoplace(true,x)}
    abc => N/N {\a.not_(a)}
    def => N/N {\b.not_(b)}
    qrs => N/N {\a.enlarge(a)}
    tuv => N/N {\b.enlarge(b)}
    twoarg => N/N/N {\a b.twoplace(a,b)}
    doggish => N/N {\x.dog(x)}
    dog => N {dog}
    true => N {true}

    # NB, won't typecheck
    cat => N {unique}
    """, ontology=ontology, include_semantics=True)

  # TODO hack: this needs to be integrated into lexicon construction..
  for w in ["the", "twoplace", "thee", "twoplacee"]:
    e = lex._entries[w][0]
    sem = e.semantics()
    tx = lex.ontology.infer_type(sem, "x")
    sem.variable.type = tx
    lex.ontology.typecheck(sem)

  return lex


def _test_case_binary(op, lex, left_entry, right_entry, allowed, expected_categ, expected_semantics):
  A = lex._entries[left_entry][0]
  B = lex._entries[right_entry][0]

  if allowed:
    ok_(op.can_combine(A, B),
        "Should allow %s of %s / %s" % (op, A, B))
    ret = list(op.combine(A, B))
    eq_(len(ret), 1)

    categ, semantics = ret[0]
    eq_(str(categ), expected_categ)
    eq_(str(semantics.normalize()), expected_semantics)
  else:
    ok_(not op.can_combine(A, B),
        "Should not allow %s of %s / %s" % (op, A, B))


def test_forward_application():
  lex = _make_mock_lexicon()

  cases = [
    ("the", "dog", True, "N", r"unique(dog)"),
    ("twoplace", "dog", True, "N", r"twoplace(true,dog)"),
    ("the", "cat", False, None, None),
    ("twoplace", "cat", False, None, None),
    ("twoarg", "true", True, "(N/N)", r"\z1.twoplace(true,z1)")
  ]

  for left, right, allowed, categ, semantics in cases:
    yield _test_case_binary, ForwardApplication, lex, left, right, allowed, categ, semantics

def test_backward_application():
  lex = _make_mock_lexicon()

  cases = [
    ("dog", "thee", True, "N", r"unique(dog)"),
    ("dog", "twoplacee", True, "N", r"twoplace(true,dog)"),
    ("cat", "thee", False, None, None),
    ("cat", "twoplacee", False, None, None),
  ]

  for left, right, allowed, categ, semantics in cases:
    yield _test_case_binary, BackwardApplication, lex, left, right, allowed, categ, semantics


def test_forward_composition():
  lex = _make_mock_lexicon()

  cases = [
    ("abc", "def", True, "(N/N)", r"\z1.not_(not_(z1))"),
    ("qrs", "tuv", True, "(N/N)", r"\z1.enlarge(enlarge(z1))"),
    ("abc", "tuv", False, None, None),
    ("qrs", "def", False, None, None),
  ]

  for left, right, allowed, categ, semantics in cases:
    yield _test_case_binary, ForwardComposition, lex, left, right, allowed, categ, semantics


def test_forward_substitution():
  lex = _make_mock_lexicon()

  cases = [
    ("twoarg", "doggish", True, "(N/N)", r"\z1.twoplace(z1,dog(z1))"),
  ]

  for left, right, allowed, categ, semantics in cases:
    yield _test_case_binary, ForwardSubstitution, lex, left, right, allowed, categ, semantics

