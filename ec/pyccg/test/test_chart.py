from nose.tools import *

from pyccg.chart import *
from pyccg.lexicon import Lexicon
from pyccg.logic import *


def _make_lexicon_with_derived_category():
  lex = Lexicon.fromstring(r"""
  :- S, NP

  the => S/NP {\x.unique(x)}

  foo => NP {\x.foo(x)}
  bar => NP {\x.bar(x)}
  baz => NP {\x.baz(x)}
  """, include_semantics=True)
  old_lex = lex.clone()

  # Induce a derived category involving `foo` and `bar`.
  involved_tokens = [lex._entries["foo"][0], lex._entries["bar"][0]]
  derived_categ = lex.add_derived_category(involved_tokens)

  return old_lex, lex, involved_tokens, derived_categ


def test_parse_with_derived_category():
  """
  Ensure that we can still parse with derived categories.
  """

  old_lex, lex, involved_tokens, categ_name = _make_lexicon_with_derived_category()
  lex.propagate_derived_category(categ_name)

  old_results = WeightedCCGChartParser(old_lex).parse("the foo".split())
  results = WeightedCCGChartParser(lex).parse("the foo".split())

  eq_(len(results), len(results))
  eq_(results[0].label()[0].semantics(), old_results[0].label()[0].semantics())


def test_parse_with_derived_root_category():
  """
  Ensure that we can parse with a derived category whose base is the root
  category.
  """
  lex = Lexicon.fromstring(r"""
      :- S, N
      the => S/N {\x.unique(x)}
      foo => N {\x.foo(x)}
      """, include_semantics=True)

  involved_tokens = [lex._entries["the"][0]]
  derived_categ = lex.add_derived_category(involved_tokens)
  lex.propagate_derived_category(derived_categ)
  derived_categ_obj, _ = lex._derived_categories[derived_categ]

  results = WeightedCCGChartParser(lex).parse("the foo".split())
  eq_(set(str(result.label()[0].categ()) for result in results),
      {"S", str(derived_categ_obj)})


def test_parse_oblique():
  """
  Test parsing a verb with an oblique PP -- this shouldn't require type raising?
  """

  lex = Lexicon.fromstring(r"""
  :- S, NP, PP

  place => S/PP/NP
  it => NP
  on => PP/NP
  the_table => NP
  """)

  parser = WeightedCCGChartParser(lex, ApplicationRuleSet)
  printCCGDerivation(parser.parse("place it on the_table".split())[0])


def test_parse_oblique_raised():
  lex = Lexicon.fromstring(r"""
  :- S, NP, PP

  place => S/NP/(PP/NP)/NP
  it => NP
  on => PP/NP
  the_table => NP
  """)

  parser = WeightedCCGChartParser(lex, DefaultRuleSet)
  printCCGDerivation(parser.parse("place it on the_table".split())[0])


def test_get_derivation_tree():
  lex = Lexicon.fromstring(r"""
  :- S, N

  John => N
  saw => S\N/N
  Mary => N
  """)

  parser = WeightedCCGChartParser(lex, ruleset=DefaultRuleSet)
  top_parse = parser.parse("Mary saw John".split())[0]

  from io import StringIO
  stream = StringIO()
  get_clean_parse_tree(top_parse).pretty_print(stream=stream)

  eq_([line.strip() for line in stream.getvalue().strip().split("\n")],
      [line.strip() for line in r"""
         S
  _______|_______
 |             (S\N)
 |        _______|____
 N   ((S\N)/N)        N
 |       |            |
Mary    saw          John""".strip().split("\n")])


def test_parse_typechecking():
  """
  Chart parser linked to an ontology should (by default) not produce
  sentence-level LFs which fail typechecks.
  """
  types = TypeSystem(["agent", "action", "object"])
  functions = [
    types.new_function("see", ("agent", "agent", "action"), lambda a, b: ("see", a, b)),
    types.new_function("request", ("agent", "object", "action"), lambda a, b: ("request", a, b)),
  ]
  constants = [types.new_constant("john", "agent"),
               types.new_constant("mary", "agent"),
               types.new_constant("help", "object")]
  ontology = Ontology(types, functions, constants)

  lex = Lexicon.fromstring(r"""
  :- S, N

  John => N {john}
  saw => S\N/N {see}
  saw => S\N/N {request}
  requested => S\N/N {request}
  Mary => N {mary}
  """, ontology=ontology, include_semantics=True)

  parser = WeightedCCGChartParser(lex, ruleset=ApplicationRuleSet)

  parses = parser.parse("Mary saw John".split())
  parse_lfs = [str(parse.label()[0].semantics()) for parse in parses]
  from pprint import pprint
  pprint(parse_lfs)

  ok_(r"see(john,mary)" in parse_lfs,
      "Parses of 'Mary saw John' should include typechecking see(john,mary)")
  ok_(r"request(john,mary)" not in parse_lfs,
      "Parses of 'Mary saw John' should not include non-typechecking request(john,mary)")


def test_parse_typechecking_complex():
  """
  Chart parser linked to an ontology should (by default) not produce
  sentence-level LFs which fail typechecks.
  """
  types = TypeSystem(["object", "boolean"])
  functions = [
    types.new_function("unique", (("object", "boolean"), "object"), lambda objs: [x for x, v in objs.items() if v][0]),
    types.new_function("big", ("object", "boolean"), lambda o: o['size'] == "big"),
    types.new_function("box", ("object", "boolean"), lambda o: o["shape"] == "box"),
    types.new_function("and_", ("boolean", "boolean", "boolean"), lambda a, b: a and b),
    types.new_function("apply", (("object", "boolean"), "object", "boolean"), lambda f, o: f(o)),
  ]
  constants = []
  ontology = Ontology(types, functions, constants)

  lex = Lexicon.fromstring(r"""
  :- S, N

  the => S/N {unique}
  the => N/N {unique}
  big => N/N {\f x.and_(apply(f,x),big(x))}
  box => N {box}
  """, ontology=ontology, include_semantics=True)

  parser = WeightedCCGChartParser(lex, ruleset=ApplicationRuleSet)

  parses = parser.parse("the the big box".split())
  eq_(len(parses), 0, "Should disallow non-typechecking parses for 'the the big box'")
