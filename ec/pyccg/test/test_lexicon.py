from unittest.mock import MagicMock

from nose.tools import *

from pyccg.lexicon import *
from pyccg import logic as l
from pyccg.chart import WeightedCCGChartParser
from pyccg.util import Distribution

from nltk.ccg.lexicon import FunctionalCategory, PrimitiveCategory, Direction


def test_filter_lexicon_entry():
  lex = Lexicon.fromstring(r"""
    :- NN, DET, ADJ

    DET :: NN/NN
    ADJ :: NN/NN

    the => DET {\x.unique(x)}
    sphere => NN {filter_shape(scene,sphere)}
    sphere => NN {filter_shape(scene,cube)}
    """, include_semantics=True)

  lex_filtered = filter_lexicon_entry(lex, "sphere", "the sphere".split(), "unique(filter_shape(scene,sphere))")

  entries = lex_filtered.categories("sphere")
  assert len(entries) == 1

  eq_(str(entries[0].semantics()), "filter_shape(scene,sphere)")


def test_multiple_starts():
  """
  Support lexicons which allow any of several root categories, signaled in
  `fromstring` with a colon.
  """
  lex = Lexicon.fromstring(r"""
    :- S:N,P

    the => N/N
    boy => N
    eats => S\N
    """)

  eq_(len(lex.start_categories), 2)

  parser = WeightedCCGChartParser(lex)

  assert len(parser.parse("the boy".split())) > 0
  assert len(parser.parse("the boy eats".split())) > 0


def test_get_semantic_arity():
  from nltk.ccg.lexicon import augParseCategory
  lex = Lexicon.fromstring(r"""
    :- NN, DET, ADJ

    DET :: NN/NN
    ADJ :: NN/NN

    the => DET {\x.unique(x)}
    sphere => NN {filter_shape(scene,sphere)}
    sphere => NN {filter_shape(scene,cube)}
    """, include_semantics=True)

  cases = [
      (r"NN", 0),
      (r"NN/NN", 1),
      (r"NN\NN", 1),
      (r"(NN\NN)/NN", 2),
  ]

  def test_case(cat, expected):
    eq_(get_semantic_arity(augParseCategory(cat, lex._primitives, lex._families)[0]),
        expected, msg=str(cat))

  for cat, expected in cases:
    yield test_case, cat, expected


def _make_lexicon_with_derived_category():
  lex = Lexicon.fromstring(r"""
  :- S, NP

  the => S/NP {\x.unique(x)}

  # Have an entry taking the same argument type twice.
  derp => S/NP/NP {\a b.derp(a,b)}
  dorp => NP/NP {\a.dorp(a)}

  foo => NP {\x.foo(x)}
  bar => NP {\x.bar(x)}
  baz => NP {\x.baz(x)}
  """, include_semantics=True)

  # Induce a derived category involving `foo` and `bar`.
  involved_tokens = [lex._entries["foo"][0], lex._entries["bar"][0]]
  derived_categ = lex.add_derived_category(involved_tokens)

  return lex, involved_tokens, derived_categ


def test_propagate_derived_category():
  lex, involved_tokens, name = _make_lexicon_with_derived_category()
  assert name in lex._derived_categories

  old_baz_categ = lex._entries["baz"][0].categ()

  categ, _ = lex._derived_categories[name]

  lex.propagate_derived_category(name)

  eq_(set(str(entry.categ()) for entry in lex._entries["foo"]), {"NP", str(categ)})
  eq_(set(str(entry.categ()) for entry in lex._entries["bar"]), {"NP", str(categ)})
  eq_(set(str(entry.categ()) for entry in lex._entries["baz"]), {str(old_baz_categ)},
      msg="Propagation of derived category should not affect `baz`, which has a "
          "category which is the same as the base of the derived category")

  eq_(len(lex._entries["the"]), 2,
      msg="Derived category propagation should have created a new functional "
          "category entry for the higher-order `the`. Only %i entries." % len(lex._entries["the"]))

  # Should try one of each possible replacement with a derived category,
  # yielding 3 entries for derp
  eq_(set(str(entry.categ()) for entry in lex._entries["derp"]),
      set(["((S/NP)/NP)", "((S/NP)/D0{NP})", "((S/D0{NP})/NP)", "((S/D0{NP})/D0{NP})"]))
  eq_(set(str(entry.categ()) for entry in lex._entries["dorp"]),
      set(["(NP/NP)", "(D0{NP}/D0{NP})", "(NP/D0{NP})", "(D0{NP}/NP)"]))


def test_propagate_functional_category():
  """
  Validate that functional categories are correctly propagated.
  """

  # This is very tricky! Suppose have a derived functional category `X/Y` and
  # there are other entries `S/X`. After propagation, we want there to be some
  # explicit type lifted form `S/(D0/Y)` where `D0 = (X/Y)`.
  lex = Lexicon.fromstring(r"""
  :- S, NN, PP

  put => S/NN/PP
  it => NN
  on => PP/NN
  the_table => NN
  """)

  involved_tokens = [lex._entries["on"][0]]
  derived_categ = lex.add_derived_category(involved_tokens)
  lex.propagate_derived_category(derived_categ)

  eq_(set(str(entry.categ()) for entry in lex._entries["on"]),
      set(["(D0{PP}/NN)", "(PP/NN)"]))
  eq_(set(str(entry.categ()) for entry in lex._entries["put"]),
      set(["((S/NN)/PP)", "((S/NN)/%s)" % lex._derived_categories[derived_categ][0]]))


def test_propagate_derived_category_distinctively():
  """
  Derived categories with functional originating categories should not be
  propagated onto other entries with the same functional category type!
  """
  lex = Lexicon.fromstring(r"""
  :- S, PP, NP

  the => S/NP {\x.unique(x)}

  derp => PP/NP {\a.derp(a)}
  dorp => PP/NP {\a.dorp(a)}
  darp => PP/NP {\a.darp(a)}
  durp => PP/NP {\a.durp(a)}
  """, include_semantics=True)

  # Induce a derived category involving `foo` and `bar`.
  involved_tokens = [lex._entries["derp"][0], lex._entries["dorp"][0]]
  derived_categ = lex.add_derived_category(involved_tokens)
  lex.propagate_derived_category(derived_categ)

  # Sanity checks
  eq_(len(lex._entries["derp"]), 2)
  eq_(len(lex._entries["dorp"]), 2)

  # Critical checks: these tokens not participating in the derived category
  # should not receive hard-propagation, since their category is the same as
  # the originating category of the derived category.
  eq_(len(lex._entries["darp"]), 1)
  eq_(len(lex._entries["durp"]), 1)


def test_soft_propagate_root_categories():
  """
  Derived categories with the root category as a base should only
  soft-propagate. We shouldn't see lexical entries hard-propagated -- it should
  only show up in `total_category_masses` and downstream methods.
  """
  lex = Lexicon.fromstring(r"""
  :- S, PP, NP

  the => S/NP {\x.unique(x)}
  that => S/NP/PP {\x y.unique(x)}

  derp => PP/NP {\a.derp(a)}
  dorp => PP/NP {\a.dorp(a)}
  darp => PP/NP {\a.darp(a)}
  durp => PP/NP {\a.durp(a)}
  """, include_semantics=True)

  # Induce a derived category involving `the`.
  involved_tokens = [lex._entries["the"][0]]
  derived_categ = lex.add_derived_category(involved_tokens)
  cat_obj, _ = lex._derived_categories[derived_categ]
  lex.propagate_derived_category(derived_categ)

  # Sanity checks
  eq_(len(lex._entries["the"]), 2)
  eq_(len(lex._entries["that"]), 1,
      "Derived category with root base should not hard-propagate.")

  expected = set_yield(lex.parse_category("S/NP/PP"), cat_obj)
  ok_(expected not in lex.total_category_masses(soft_propagate_roots=False))
  ok_(expected in lex.total_category_masses(soft_propagate_roots=True))


def test_get_lf_unigrams():
  lex = Lexicon.fromstring(r"""
    :- NN

    the => NN/NN {\x.unique(x)}
    sphere => NN {\x.and_(object(x),sphere(x))}
    cube => NN {\x.and_(object(x),cube(x))}
    """, include_semantics=True)

  expected = {
    "NN": Counter({"and_": 2 / 6, "object": 2 / 6, "sphere": 1 / 6, "cube": 1 / 6, None: 0.0, "unique": 0.0}),
    "(NN/NN)": Counter({"unique": 1, "cube": 0.0, "object": 0.0, "sphere": 0.0, "and_": 0.0, None: 0.0})
  }

  ngrams = lex.lf_ngrams_given_syntax(order=1, smooth=False)
  for categ, dist in ngrams.dists.items():
    eq_(dist, expected[str(categ)])


def test_get_yield():
  from nltk.ccg.lexicon import augParseCategory
  lex = Lexicon.fromstring(r"""
    :- S, NN, PP

    on => PP/NN
    the => S/NN
    the => NN/NN
    sphere => NN
    sphere => NN
    """)

  cases = [
      ("S", "S"),
      ("S/NN", "S"),
      (r"NN\PP/NN", "NN"),
      (r"(S\NN)/NN", "S"),
  ]

  def test_case(cat, cat_yield):
    eq_(get_yield(lex.parse_category(cat)), lex.parse_category(cat_yield))

  for cat, cat_yield in cases:
    yield test_case, cat, cat_yield


def test_set_yield():
  from nltk.ccg.lexicon import augParseCategory
  lex = Lexicon.fromstring(r"""
    :- S, NN, PP

    on => PP/NN
    the => S/NN
    the => NN/NN
    sphere => NN
    sphere => NN
    """)

  cases = [
      ("S", "NN", "NN"),
      ("S/NN", "NN", "NN/NN"),
      (r"NN\PP/NN", "S", r"S\PP/NN"),
      (r"(S\NN)/NN", "NN", r"(NN\NN)/NN"),
  ]

  def test_case(cat, update, expected):
    source = lex.parse_category(cat)
    updated = set_yield(source, update)

    eq_(str(updated), str(lex.parse_category(expected)))

  for cat, update, expected in cases:
    yield test_case, cat, update, expected


def test_attempt_candidate_parse():
  """
  Find parse candidates even when the parse requires composition.
  """
  lex = Lexicon.fromstring(r"""
  :- S, N

  gives => S\N/N/N {\o x y.give(x, y, o)}
  John => N {\x.John(x)}
  Mark => N {\x.Mark(x)}
  it => N {\x.T}
  """, include_semantics=True)
  # TODO this doesn't actually require composition .. get one which does

  cand_category = lex.parse_category(r"S\N/N/N")
  cand_expressions = [l.Expression.fromstring(r"\o x y.give(x,y,o)")]
  dummy_vars = {"sends": l.Variable("F000")}
  results = attempt_candidate_parse(lex, ["sends"], [cand_category],
                                    "John sends Mark it".split(),
                                    dummy_vars)

  ok_(len(list(results)) > 0)


def _make_simple_mock_ontology():
  types = l.TypeSystem(["boolean", "obj"])
  functions = [
      types.new_function("and_", ("boolean", "boolean", "boolean"), lambda x, y: x and y),
      types.new_function("foo", ("obj", "boolean"), lambda x: True),
      types.new_function("bar", ("obj", "boolean"), lambda x: True),
      types.new_function("not_", ("boolean", "boolean"), lambda x: not x),

      types.new_function("invented_1", (("obj", "boolean"), "obj", "boolean"), lambda f, x: x is not None and f(x)),

      types.new_function("threeplace", ("obj", "obj", "boolean", "boolean"), lambda x, y, o: True),
  ]
  constants = [types.new_constant("baz", "boolean"), types.new_constant("qux", "obj")]

  ontology = l.Ontology(types, functions, constants, variable_weight=0.1)
  return ontology


def test_fromstring_typechecks():
  """
  Ensure that `Lexicon.fromstring` type-checks and assigns types to provided
  entries.
  """
  ontology = _make_simple_mock_ontology()
  lex = Lexicon.fromstring(r"""
  :- S
  foo => S {\x.and_(x,baz)}
  """, ontology=ontology, include_semantics=True)

  foo_entry = lex._entries["foo"][0]
  eq_(foo_entry.semantics().type, ontology.types["boolean", "boolean"])
  eq_(foo_entry.semantics().variable.type, ontology.types["boolean"])


def test_fromstring_typecheck_failure():
  ontology = _make_simple_mock_ontology()
  def should_raise():
    lex = Lexicon.fromstring(r"""
    :- S
    foo => S {\x.and_(x,qux)}
    """, ontology=ontology, include_semantics=True)

  assert_raises(l.TypeException, should_raise)


def test_add_entry_typecheck():
  """
  Ensure that dynamically added entries are type-checked / type-assigned.
  """
  ontology = _make_simple_mock_ontology()
  lex = Lexicon.fromstring(r"""
  :- S
  foo => S {\x.and_(x,baz)}
  """, ontology=ontology, include_semantics=True)

  lex.add_entry("bar", lex.parse_category("S"), l.Expression.fromstring(r"\x.and_(x,x)"))
  bar_entry = lex.get_entries("bar")[0]
  eq_(bar_entry.semantics().type, ontology.types["boolean", "boolean"])
  eq_(bar_entry.semantics().variable.type, ontology.types["boolean"])


def test_add_entry_typecheck_failure():
  ontology = _make_simple_mock_ontology()
  lex = Lexicon.fromstring(r"""
  :- S
  foo => S {\x.and_(x,baz)}
  """, ontology=ontology, include_semantics=True)

  def should_raise():
    lex.add_entry("bar", lex.parse_category("S"), l.Expression.fromstring(r"\x.and_(x,qux)"))
  assert_raises(l.TypeException, should_raise)


def test_zero_shot_type_request():
  """
  predict_zero_shot should infer the types of missing semantic forms and use as
  specific a possible type request when invoking `Ontology.iter_expressions`
  """
  ontology = _make_simple_mock_ontology()
  lex = Lexicon.fromstring(r"""
  :- S, N
  bar => N {baz}
  blah => S/N {baz}
  """, ontology=ontology, include_semantics=True)

  # setup: we observe a sentence "foo bar". ground truth semantics for 'foo' is
  # \x.and(x,baz)

  # Mock ontology.predict_zero_shot
  mock = MagicMock(return_value=[])
  ontology.iter_expressions = mock

  tokens = ["foo"]
  candidate_syntaxes = {"foo": Distribution.uniform([lex.parse_category("S/N")])}
  sentence = "foo bar".split()

  predict_zero_shot(lex, tokens, candidate_syntaxes, sentence, ontology,
                    model=None, likelihood_fns=[])

  eq_(len(mock.call_args_list), 1)
  args, kwargs = mock.call_args
  eq_(kwargs["type_request"], ontology.types["boolean", "e"])
