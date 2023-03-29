from nose.tools import *

from frozendict import frozendict

from pyccg.logic import Expression, Ontology, TypeSystem, Function
from pyccg.model import *


def _make_mock_ontology():
  def fn_unique(xs):
    true_xs = [x for x, matches in xs.items() if matches]
    assert len(true_xs) == 1
    return true_xs[0]

  types = TypeSystem(["object", "boolean"])
  functions = [
    types.new_function("left_of", ("object", "object", "boolean"), lambda a, b: a["x"] < b["x"]),
    types.new_function("unique", (("object", "boolean"), "object"), fn_unique),
    types.new_function("cube", ("object", "boolean"), lambda x: x["shape"] == "cube"),
    types.new_function("sphere", ("object", "boolean"), lambda x: x["shape"] == "sphere"),

    types.new_function("and_", ("boolean", "boolean", "boolean"), lambda x, y: x and y),
  ]

  constants = []

  ontology = Ontology(types, functions, constants)

  return ontology


def test_model_constants():
  """
  Test evaluating with constant values.
  """
  types = TypeSystem(["num"])

  functions = [
    types.new_function("add", ("num", "num", "num"), lambda a, b: str(int(a) + int(b)))
  ]
  constants = [types.new_constant("1", "num"), types.new_constant("2", "num")]

  ontology = Ontology(types, functions, constants)
  model = Model(scene={"objects": []}, ontology=ontology)

  cases = [
    ("Test basic constant evaluation", r"1", "1"),
    ("Test constants as arguments to functions", r"add(1,1)", "2"),
  ]

  def test(msg, expr, expected):
    print("ret", model.evaluate(Expression.fromstring(expr)))
    eq_(model.evaluate(Expression.fromstring(expr)), expected, msg=msg)

  for msg, expr, expected in cases:
    yield test, msg, expr, expected


def test_model_induced_functions():
  """
  Test evaluating a model with an ontology which has induced functions.
  """

  fake_scene = {
    "objects": ["foo", "bar"],
  }

  types = TypeSystem(["a"])
  functions = [
      types.new_function("test", ("a", "a"), lambda x: True),
      types.new_function("test2", ("a", "a"), Expression.fromstring(r"\x.test(test(x))")),
  ]
  ontology = Ontology(types, functions, [])

  model = Model(scene=fake_scene, ontology=ontology)

  cases = [
    ("Test basic call of an abstract function", r"\a.test2(a)", {"foo": True, "bar": True}),
    ("Test embedded call of an abstract function", r"\a.test(test2(a))", {"foo": True, "bar": True}),
  ]

  def test(msg, expr, expected):
    eq_(model.evaluate(Expression.fromstring(expr)), expected)

  for msg, expr, expected in cases:
    yield test, msg, expr, expected


def test_model_partial_application():
  types = TypeSystem(["obj"])
  functions = [
    types.new_function("lotsofargs", ("obj", "obj", "obj"), lambda a, b: b),
  ]
  constants = [
      types.new_constant("obj1", "obj"),
      types.new_constant("obj2", "obj"),
  ]
  ontology = Ontology(types, functions, constants)

  scene = {"objects": []}
  model = Model(scene, ontology)

  eq_(model.evaluate(Expression.fromstring(r"(lotsofargs(obj1))(obj2)")), "obj2")


def test_model_stored_partial_application():
  types = TypeSystem(["obj"])
  functions = [
    types.new_function("lotsofargs", ("obj", "obj", "obj"), lambda a, b: b),
  ]
  constants = [
      types.new_constant("obj1", "obj"),
      types.new_constant("obj2", "obj"),
  ]
  ontology = Ontology(types, functions, constants)
  ontology.add_functions([types.new_function("partial", ("obj", "obj"), Expression.fromstring(r"lotsofargs(obj2)"))])

  scene = {"objects": []}
  model = Model(scene, ontology)

  eq_(model.evaluate(Expression.fromstring(r"partial(obj1)")), "obj1")


def test_nested_lambda():
  """
  Test evaluation of nested lambda expressions.
  """
  ontology = _make_mock_ontology()

  scene = {"objects": [
    frozendict(x=3, shape="sphere"),
    frozendict(x=4, shape="cube"),
  ]}
  model = Model(scene, ontology)

  eq_(model.evaluate(Expression.fromstring(r"unique(\x.left_of(x,unique(\y.cube(y))))")),
      scene["objects"][0])
  eq_(model.evaluate(Expression.fromstring(r"sphere(unique(\x.left_of(x,unique(\y.cube(y)))))")),
      True)


def test_base_function():
  """
  Support domain enumeration when a function appears as a constant in "base"
  form.
  """
  ontology = _make_mock_ontology()
  scene = {"objects": [
    frozendict(x=3, shape="sphere"),
    frozendict(x=4, shape="cube"),
  ]}
  model = Model(scene, ontology)

  eq_(model.evaluate(Expression.fromstring(r"unique(cube)")), scene["objects"][1])


def test_property_function_cache():
  ontology = _make_mock_ontology()
  scene = {"objects": [
    frozendict(x=3, shape="sphere"),
    frozendict(x=4, shape="cube"),
  ]}
  model = Model(scene, ontology)

  ok_("unique" in model._property_function_cache,
      "Should prepare to cache `unique` function")
  eq_(len(model._property_function_cache["unique"]), 0)

  expr = Expression.fromstring(r"unique(\x.sphere(x))")
  expected = scene["objects"][0]

  eq_(model.evaluate(expr), expected)
  ok_(len(model._property_function_cache["unique"]) > 0,
      "Cache should be populated after call")

  eq_(model.evaluate(expr), expected, "Cached evaluation returns the same value")
