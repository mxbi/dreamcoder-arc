from copy import deepcopy
import itertools

from nltk.ccg import chart as nchart
from nltk.parse.chart import AbstractChartRule, Chart, EdgeI
from nltk.tree import Tree
import numpy as np

from pyccg import Token
from pyccg.combinator import *
from pyccg.logic import *


printCCGDerivation = nchart.printCCGDerivation


# Based on the EdgeI class from NLTK.
# A number of the properties of the EdgeI interface don't
# transfer well to CCGs, however.
class CCGEdge(EdgeI):
    def __init__(self, span, categ, rule, semantics=None):
        self._span = span
        self._categ = categ
        self._rule = rule
        self._semantics = semantics
        self._comparison_key = (span, categ, rule, semantics)

    # Accessors
    def lhs(self): return self._categ
    def span(self): return self._span
    def start(self): return self._span[0]
    def end(self): return self._span[1]
    def length(self): return self._span[1] - self.span[0]
    def rhs(self): return ()
    def dot(self): return 0
    def is_complete(self): return True
    def is_incomplete(self): return False
    def nextsym(self): return None

    def categ(self): return self._categ
    def rule(self): return self._rule
    def semantics(self): return self._semantics

class CCGLeafEdge(EdgeI):
    '''
    Class representing leaf edges in a CCG derivation.
    '''
    def __init__(self, pos, token, leaf):
        self._pos = pos
        self._token = token
        self._leaf = leaf
        self._comparison_key = (pos, token.categ(), leaf)

    # Accessors
    def lhs(self): return self._token.categ()
    def span(self): return (self._pos, self._pos+1)
    def start(self): return self._pos
    def end(self): return self._pos + 1
    def length(self): return 1
    def rhs(self): return self._leaf
    def dot(self): return 0
    def is_complete(self): return True
    def is_incomplete(self): return False
    def nextsym(self): return None

    def token(self): return self._token
    def categ(self): return self._token.categ()
    def semantics(self): return self._token.semantics()
    def leaf(self): return self._leaf


class CCGChartRule(AbstractChartRule):

  def set_ontology(self, ontology):
    if hasattr(self, "_combinator"):
      self._combinator.set_ontology(ontology)


class BinaryCombinatorRule(CCGChartRule):
    '''
    Class implementing application of a binary combinator to a chart.
    Takes the directed combinator to apply.
    '''
    NUMEDGES = 2
    def __init__(self,combinator):
        self._combinator = combinator

    # Apply a combinator
    def apply(self, chart, grammar, left_edge, right_edge):
        # The left & right edges must be touching.
        if not (left_edge.end() == right_edge.start()):
            return

        # Check if the two edges are permitted to combine.
        # If so, generate the corresponding edge.
        if self._combinator.can_combine(left_edge, right_edge):
            for categ, semantics in self._combinator.combine(left_edge, right_edge):
                new_edge = CCGEdge(span=(left_edge.start(), right_edge.end()),
                                   categ=categ, semantics=semantics,
                                   rule=self._combinator)
                if chart.insert(new_edge,(left_edge,right_edge)):
                    yield new_edge

    # The representation of the combinator (for printing derivations)
    def __str__(self):
        return "%s" % self._combinator

# Type-raising must be handled slightly differently to the other rules, as the
# resulting rules only span a single edge, rather than both edges.
class ForwardTypeRaiseRule(CCGChartRule):
    '''
    Class for applying forward type raising
    '''
    NUMEDGES = 2

    def __init__(self):
       self._combinator = ForwardT
    def apply(self, chart, grammar, left_edge, right_edge):
        if not (left_edge.end() == right_edge.start()):
            return

        for categ, semantics in self._combinator.combine(left_edge, right_edge):
            new_edge = CCGEdge(span=left_edge.span(), categ=categ, semantics=semantics,
                               rule=self._combinator)
            if chart.insert(new_edge,(left_edge,)):
                yield new_edge

    def __str__(self):
        return "%s" % self._combinator

class BackwardTypeRaiseRule(CCGChartRule):
    '''
    Class for applying backward type raising.
    '''
    NUMEDGES = 2

    def __init__(self):
       self._combinator = BackwardT
    def apply(self, chart, grammar, left_edge, right_edge):
        if not (left_edge.end() == right_edge.start()):
            return

        for categ, semantics in self._combinator.combine(left_edge, right_edge):
            new_edge = CCGEdge(span=right_edge.span(), categ=categ, semantics=semantics,
                               rule=self._combinator)
            if chart.insert(new_edge,(right_edge,)):
                yield new_edge

    def __str__(self):
        return "%s" % self._combinator


# Common sets of combinators used for English derivations.
ApplicationRuleSet = [BinaryCombinatorRule(ForwardApplication),
                        BinaryCombinatorRule(BackwardApplication)]
CompositionRuleSet = [BinaryCombinatorRule(ForwardComposition),
                        BinaryCombinatorRule(BackwardComposition),
                        BinaryCombinatorRule(BackwardBx)]
SubstitutionRuleSet = [BinaryCombinatorRule(ForwardSubstitution),
                        BinaryCombinatorRule(BackwardSx)]
TypeRaiseRuleSet = [ForwardTypeRaiseRule(), BackwardTypeRaiseRule()]

# The standard English rule set.
DefaultRuleSet = ApplicationRuleSet + CompositionRuleSet + \
                    SubstitutionRuleSet + TypeRaiseRuleSet


class CCGChart(Chart):
  def __init__(self, tokens):
    Chart.__init__(self, tokens)

  # Constructs the trees for a given parse. Unfortnunately, the parse trees need to be
  # constructed slightly differently to those in the default Chart class, so it has to
  # be reimplemented
  def _trees(self, edge, complete, memo, tree_class):
    assert complete, "CCGChart cannot build incomplete trees"

    if edge in memo:
      return memo[edge]

    if isinstance(edge,CCGLeafEdge):
      word = tree_class(edge.token(), [self._tokens[edge.start()]])
      leaf = tree_class((edge.token(), "Leaf"), [word])
      memo[edge] = [leaf]
      return [leaf]

    memo[edge] = []
    trees = []

    for cpl in self.child_pointer_lists(edge):
      child_choices = [self._trees(cp, complete, memo, tree_class)
                       for cp in cpl]
      for children in itertools.product(*child_choices):
        lhs = (Token(self._tokens[edge.start():edge.end()], edge.lhs(), edge.semantics()), str(edge.rule()))
        trees.append(tree_class(lhs, children))

    memo[edge] = trees
    return trees


class WeightedCCGChartParser(nchart.CCGChartParser):
  """
  CCG chart parser building off of the basic NLTK parser.

  Current extensions:

  1. Weighted inference (with weights on lexicon)
  2. Exhaustive search in cases where lexicon entries have ambiguous
  semantics. By default, NLTK ignores entries which have different
  semantics but share syntactic categories.
  """

  def __init__(self, lexicon, ruleset=None, *args, **kwargs):
    if ruleset is None:
      ruleset = ApplicationRuleSet

    if lexicon.ontology is not None:
      ruleset = deepcopy(ruleset)
      for rule in ruleset:
        rule.set_ontology(lexicon.ontology)

    super().__init__(lexicon, ruleset, *args, **kwargs)

  def _parse_inner(self, chart):
    """
    Run a chart parse on a chart with the edges already filled in.
    """

    # Select a span for the new edges
    for span in range(2,chart.num_leaves()+1):
      for start in range(0,chart.num_leaves()-span+1):
        # Try all possible pairs of edges that could generate
        # an edge for that span
        for part in range(1,span):
          lstart = start
          mid = start + part
          rend = start + span

          for left in chart.select(span=(lstart,mid)):
            for right in chart.select(span=(mid,rend)):
              # Generate all possible combinations of the two edges
              for rule in self._rules:
                edges_added_by_rule = 0
                for newedge in rule.apply(chart,self._lexicon,left,right):
                  edges_added_by_rule += 1

    # Attempt parses with the lexicon's start category as the root, or any
    # derived category which has the start category as base.
    parses = []
    for start_cat in self._lexicon.start_categories:
      parses.extend(chart.parses(start_cat))
    return parses

  def parse(self, tokens, return_aux=False):
    """
    Args:
      tokens: list of string tokens
      return_aux: return auxiliary information (`weights`, `valid_edges`)

    Returns:
      parses: list of CCG derivation results
      if return_aux, the list is actually a tuple with `parses` as its first
      element and the other following elements:
        weight: float parse weight
        edges: `tokens`-length list of the edge tokens used to generate this
          parse
    """
    tokens = list(tokens)
    lex = self._lexicon

    # Collect potential leaf edges for each index. May be multiple per
    # token.
    edge_cands = [[CCGLeafEdge(i, l_token, token) for l_token in lex.categories(token)]
                   for i, token in enumerate(tokens)]

    # Run a parse for each of the product of possible leaf nodes,
    # and merge results.
    results = []
    used_edges = []
    for edge_sequence in itertools.product(*edge_cands):
      chart = CCGChart(list(tokens))
      for leaf_edge in edge_sequence:
        chart.insert(leaf_edge, ())

      partial_results = list(self._parse_inner(chart))
      results.extend(partial_results)

      if return_aux:
        # Track which edge values were used to generate these parses.
        used_edges.extend([edge_sequence] * len(partial_results))

    # Score using Bayes' rule, calculated with lexicon weights.
    cat_priors = self._lexicon.observed_category_distribution()
    total_cat_masses = self._lexicon.total_category_masses()
    def score_parse(parse):
      score = 0.0
      for _, token in parse.pos():
        if total_cat_masses[token.categ()] == 0:
          return -np.inf
        # TODO not the same scoring logic as in novel word induction .. an
        # ideal Bayesian model would have these aligned !! (No smoothing here)
        likelihood = max(token.weight(), 1e-6) / total_cat_masses[token.categ()]
        logp = 0.5 * np.log(cat_priors[token.categ()])
        logp += np.log(likelihood)

        score += logp
      return score

    results = sorted(results, key=score_parse, reverse=True)
    if not return_aux:
      return results
    return [(parse, score_parse(parse), used_edges_i)
            for parse, used_edges_i in zip(results, used_edges)]


def get_clean_parse_tree(ccg_chart_result):
  """
  Get a clean parse tree representation of a CCG derivation, as returned by
  `CCGChartParser.parse`.
  """
  def traverse(node):
    if not isinstance(node, Tree):
      return

    label = node.label()
    if isinstance(label, tuple):
      token, op = label
      node.set_label(str(token.categ()))

    for i, child in enumerate(node):
      if len(child) == 1:
        new_preterminal = child[0]
        new_preterminal.set_label(str(new_preterminal.label().categ()))
        node[i] = new_preterminal
      else:
        traverse(child)

  ret = ccg_chart_result.copy(deep=True)
  traverse(ret)

  return ret
