"""
Tools for updating and expanding lexicons, dealing with logical forms, etc.
"""

from collections import defaultdict, Counter
from copy import copy, deepcopy
from functools import reduce
import itertools
import logging
import queue
import re
import sys

from nltk.ccg import lexicon as ccg_lexicon
from nltk.ccg.api import PrimitiveCategory, FunctionalCategory, AbstractCCGCategory
import numpy as np
from tqdm import tqdm, trange

from pyccg import chart, Token
from pyccg.combinator import category_search_replace
from pyccg import logic as l
from pyccg.util import ConditionalDistribution, Distribution, UniquePriorityQueue, \
    NoParsesError, tuple_unordered


L = logging.getLogger(__name__)


#------------
# Regular expressions used for parsing components of the lexicon
#------------

# Parses a primitive category and subscripts
PRIM_RE = re.compile(r'''([A-Za-z]+)(\[[A-Za-z,]+\])?''')

# Separates the next primitive category from the remainder of the
# string
NEXTPRIM_RE = re.compile(r'''([A-Za-z]+(?:\[[A-Za-z,]+\])?)(.*)''')

# Separates the next application operator from the remainder
APP_RE = re.compile(r'''([\\/])([.,]?)([.,]?)(.*)''')

# Parses the definition of the right-hand side (rhs) of either a word or a family
LEX_RE = re.compile(r'''([\S_]+)\s*(::|[-=]+>)\s*(.+)''', re.UNICODE)

# Parses the right hand side that contains category and maybe semantic predicate
RHS_RE = re.compile(r'''([^{}<>]*[^ {}<>])\s*(\{[^}]+\})?\s*(<-?\d*(?:\.\d+)?>)?''', re.UNICODE)

# Parses the semantic predicate
SEMANTICS_RE = re.compile(r'''\{([^}]+)\}''', re.UNICODE)

# Strips comments from a line
COMMENTS_RE = re.compile('''([^#]*)(?:#.*)?''')


class Lexicon(ccg_lexicon.CCGLexicon):

  def __init__(self, starts, primitives, families, entries, ontology=None):
    """
    Create a new Lexicon.

    Args:
      start: Start symbol(s). All valid parses must have a root node of this
        category. Either a string (single start) or a sequence (multiple
        allowed starts).
      primitives:
      families:
      entries: Lexical entries. Dict mapping from word strings to lists of
        `Token`s.
    """
    starts = [starts] if isinstance(starts, str) else starts
    self._starts = [ccg_lexicon.PrimitiveCategory(start) for start in starts]
    self._primitives = primitives
    self._families = families
    self._entries = entries

    self.ontology = ontology

    self._derived_categories = {}
    self._derived_categories_by_base = defaultdict(set)
    self._derived_categories_by_source = {}

  @classmethod
  def fromstring(cls, lex_str, ontology=None, include_semantics=False,
                 default_weight=0.001):
    """
    Convert string representation into a lexicon for CCGs.
    """
    ccg_lexicon.CCGVar.reset_id()
    primitives, starts = [], []
    families = {}
    entries = defaultdict(list)
    for line in lex_str.splitlines():
      # Strip comments and leading/trailing whitespace.
      line = COMMENTS_RE.match(line).groups()[0].strip()
      if line == "":
        continue

      if line.startswith(':-'):
        # A line of primitive categories.
        # The first one is the target category
        # ie, :- S, N, NP, VP
        primitives = primitives + [prim.strip() for prim in line[2:].strip().split(',')]

        # But allow multiple target categories separated by a colon in the first element:
        # ie, :- S:N,NP,VP
        starts = primitives[0].split(":")
        primitives = starts + primitives[1:]
      else:
        # Either a family definition, or a word definition
        (ident, sep, rhs) = LEX_RE.match(line).groups()
        (catstr, semantics_str, weight) = RHS_RE.match(rhs).groups()
        (cat, var) = ccg_lexicon.augParseCategory(catstr, primitives, families)

        if sep == '::':
          # Family definition
          # ie, Det :: NP/N
          families[ident] = (cat, var)
          # TODO weight?
        else:
          semantics = None
          if include_semantics is True:
            if semantics_str is None:
              raise AssertionError(line + " must contain semantics because include_semantics is set to True")
            else:
              semantics = l.Expression.fromstring(ccg_lexicon.SEMANTICS_RE.match(semantics_str).groups()[0])

              # Assign types.
              if ontology is not None:
                ontology.typecheck(semantics)

          weight = float(weight[1:-1]) if weight is not None else default_weight

          # Word definition
          # ie, which => (N\N)/(S/NP)
          entries[ident].append(Token(ident, cat, semantics, weight=weight))
    return cls(starts, primitives, families, entries,
               ontology=ontology)

  def get_entries(self, word):
    return self._entries.get(word, [])

  def set_entries(self, word, entries):
    """
    Set the list of entries for a wordform `word`.

    Arguments:
      word: String wordform
      entries: List of `(category, semantics, weight)` tuples
    """
    self._entries[word] = []
    for category, semantics, weight in entries:
      self.add_entry(word, category, semantics=semantics, weight=weight)

  def add_entry(self, word, category, semantics=None, weight=None):
    """
    Add a `Token` entry for the wordform `word`.

    Arguments:
      word: String wordform
      category: Syntactic category
      semantics: Meaning representation
      weight: Float weight
    """
    # Typecheck and assign types in semantic representation.
    if semantics is not None and self.ontology is not None:
      self.ontology.typecheck(semantics)

    token = Token(word, category, semantics=semantics, weight=weight)
    self._entries[word].append(token)

  def __eq__(self, other):
    return isinstance(other, Lexicon) and self._starts == other._starts \
        and self._primitives == other._primitives and self._families == other._families \
        and self._entries == other._entries \
        and self._derived_categories == other._derived_categories

  def clone(self, retain_semantics=True):
    """
    Return a clone of the current lexicon instance.
    """
    ret = deepcopy(self)

    if not retain_semantics:
      for entry_tokens in ret._entries.values():
        for token in entry_tokens:
          token._semantics = None

    return ret

  def prune(self, max_entries=3):
    """
    Prune low-weight entries from the lexicon in-place.

    Args:
      min_weight: Minimal weight for entries which should be retained.

    Returns:
      prune_count: Number of lexical entries which were pruned.
    """
    prune_count = 0
    for token in self._entries:
      entries_t = [token for token in self._entries[token] if token.weight() > 0]
      entries_t = sorted(entries_t, key=lambda t: t.weight())[:max_entries]
      prune_count += len(self._entries[token]) - len(entries_t)
      self._entries[token] = entries_t

    return prune_count

  def debug_print(self, stream=sys.stdout):
    for token, entries in self._entries.items():
      for entry in entries:
        stream.write("%.3f %s\n" % (entry.weight(), entry))

  def parse_category(self, cat_str):
    return ccg_lexicon.augParseCategory(cat_str, self._primitives, self._families)[0]

  @property
  def primitive_categories(self):
    return set([self.parse_category(prim) for prim in self._primitives])

  @property
  def observed_categories(self):
    """
    Find categories (both primitive and functional) attested in the lexicon.
    """
    return set([token.categ()
                for token_list in self._entries.values()
                for token in token_list])

  def total_category_masses(self, exclude_tokens=frozenset(),
                            soft_propagate_roots=False):
    """
    Return the total weight mass assigned to each syntactic category. Shifts
    masses such that the minimum mass is zero.

    Args:
      exclude_tokens: Exclude entries with this token from the count.
      soft_propagate_roots: Soft-propagate derived root categories. If there is
        a derived root category `D0{S}` and some lexical entry `S/N`, even if
        no entry has the category `D0{S}`, we will add a key to the returned
        counter with category `D0{S}/N` (and zero weight).

    Returns:
      masses: `Distribution` mapping from category types to masses. The minimum
        mass value is zero and the maximum is unbounded.
    """
    ret = Distribution()
    # Track categories with root yield.
    rooted_cats = set()

    for token, entries in self._entries.items():
      if token in exclude_tokens:
        continue
      for entry in entries:
        c_yield = get_yield(entry.categ())
        if c_yield in self._starts:
          rooted_cats.add((c_yield, entry.categ()))

        if entry.weight() > 0.0:
          ret[entry.categ()] += entry.weight()

    if soft_propagate_roots:
      for c_yield, rooted_cat in rooted_cats:
        for derived_root_cat in self._derived_categories_by_base[c_yield]:
          soft_prop_cat = set_yield(rooted_cat, derived_root_cat)
          # Ensure key exists.
          ret.setdefault(soft_prop_cat, 0.0)

    return ret

  def observed_category_distribution(self, exclude_tokens=frozenset(),
                                     soft_propagate_roots=False):
    """
    Return a distribution over categories calculated using the lexicon weights.
    """
    ret = self.total_category_masses(exclude_tokens=exclude_tokens,
                                     soft_propagate_roots=soft_propagate_roots)
    return ret.normalize()

  @property
  def start_categories(self):
    """
    Return primitive categories which are valid root nodes.
    """
    return self._starts + \
        list(itertools.chain.from_iterable(self._derived_categories_by_base[start]
                                           for start in self._starts))

  def start(self):
    raise NotImplementedError("use #start_categories instead.")

  def category_semantic_arities(self, soft_propagate_roots=False):
    """
    Get the arities of semantic expressions associated with each observed
    syntactic category.
    """
    # If possible, lean on the type system to help determine expression arity.
    get_arity = (self.ontology and self.ontology.get_expr_arity) \
        or get_semantic_arity

    entries_by_categ = {
      category: set(entry for entry in itertools.chain.from_iterable(self._entries.values())
                    if entry.categ() == category)
      for category in self.observed_categories
    }

    ret = {}
    rooted_cats = set()
    for category, entries in entries_by_categ.items():
      if get_yield(category) in self._starts:
        rooted_cats.add((get_yield(category), category))
      ret[category] = set(get_arity(entry.semantics()) for entry in entries)

    if soft_propagate_roots:
      for c_yield, rooted_cat in rooted_cats:
        for derived_root_cat in self._derived_categories_by_base[c_yield]:
          new_syn = set_yield(rooted_cat, derived_root_cat)
          ret.setdefault(new_syn, ret[rooted_cat])

    return ret

  def add_derived_category(self, involved_tokens, source_name=None):
    name = "D%i" % len(self._derived_categories)

    # The derived category will have as its base the yield of the source
    # category. For example, tokens of type `PP/NP` will lead to a derived
    # category of type `PP`.
    categ = DerivedCategory(name, get_yield(involved_tokens[0].categ()),
                            source_name=source_name)
    self._primitives.append(categ)
    self._derived_categories[name] = (categ, set(involved_tokens))
    self._derived_categories_by_base[categ.base].add(categ)

    if source_name is not None:
      self._derived_categories_by_source[source_name] = categ

    return name

  def propagate_derived_category(self, name):
    categ, involved_entries = self._derived_categories[name]
    originating_category = next(iter(involved_entries)).categ()
    new_entries = defaultdict(list)

    # Replace all lexical entries directly involved with the derived category.
    for word, entry_list in self._entries.items():
      for entry in entry_list:
        if entry in involved_entries:
          # Replace the yield of the syntactic category with our new derived
          # category. (For primitive categories, setting the yield is
          # equivalent to just changing the category.)
          new_entry = entry.clone()
          new_entry._categ = set_yield(entry.categ(), categ)
          new_entries[word].append(new_entry)

    # Create duplicates of all entries with functional categories involving the
    # base of the derived category.
    #
    # For example, if we have an entry of syntactic category `S/N/PP` and we
    # have just created a derived category `D0` based on `N`, we need to make
    # sure there is now a corresponding candidate entry of type `S/D0/PP`.

    replacements = {}
    # HACK: don't propagate S
    if categ.base.categ() != "S":
      for word, entries in self._entries.items():
        for entry in entries:
          if not isinstance(entry.categ(), FunctionalCategory):
            # This step only applies to functional categories.
            continue
          elif entry.categ() == originating_category:
            # We've found an entry which has a category with the same category
            # as that of the tokens involved in this derived category's
            # creation. Don't propagate -- this is exactly what allows us to
            # separate the involved tokens from other members of the same
            # category.
            #
            # e.g. if we've just derived a category from some PP/NP entries,
            # don't propagate the PP yield onto other PP/NP entries which were
            # not involved in the derived category creation.
            continue

          try:
            categ_replacements = replacements[entry.categ()]
          except KeyError:
            replacements[entry.categ()] = category_search_replace(
                entry.categ(), categ.base, categ)

            categ_replacements = replacements[entry.categ()]

          for replacement_category in categ_replacements:
            # We already know a replacement is necessary -- go ahead.
            new_entry = entry.clone()
            new_entry._categ = replacement_category
            new_entries[word].append(new_entry)

    for word, w_entries in new_entries.items():
      self._entries[word].extend(w_entries)


  def lf_ngrams(self, order=1, conditioning_fn=None, smooth=None):
    """
    Calculate n-gram statistics about the predicates present in the semantic
    forms in the lexicon.

    Args:
      order: n-gram order
      conditioning_fn: If non-`None`, returns conditional distributions mapping
        from the range of `conditioning_fn` to distributions over semantic
        predicates. This can be used to e.g. build distributions over
        predicates conditioned on syntactic category.
      smooth: If not `None`, add-k smooth the returned distributions using the
        provided float.
    """
    if order > 1:
      raise NotImplementedError()

    ret = ConditionalDistribution()
    for entry_list in self._entries.values():
      for entry in entry_list:
        keys = conditioning_fn(entry) if conditioning_fn is not None else [None]
        for key in keys:
          # Initialize the distribution, whether or not we will find any
          # predicates to count.
          ret.ensure_cond_support(key)

          for predicate in entry.semantics().predicates():
            ret[key][predicate.name] += entry.weight()

    if smooth is not None:
      support = ret.support
      for key in ret:
        for predicate in support:
          ret[key][predicate] += smooth
        ret[key][None] += smooth

    ret.normalize_all()

    if conditioning_fn is None:
      return ret[None]
    return ret

  def lf_ngrams_given_syntax(self, **kwargs):
    conditioning_fn = lambda entry: [entry.categ()]
    kwargs["conditioning_fn"] = conditioning_fn
    return self.lf_ngrams(**kwargs)

  def lf_ngrams_mixed(self, alpha=0.25, **kwargs):
    """
    Return conditional distributions over logical form n-grams conditioned on
    syntactic category, calculated by mixing two distribution classes: a
    distribution conditioned on the full syntactic category and a distribution
    conditioned on the yield of the category.
    """
    lf_syntax_ngrams = self.lf_ngrams_given_syntax(**kwargs)
    lf_support = lf_syntax_ngrams.support

    # Soft-propagate derived root categories.
    for syntax in list(lf_syntax_ngrams.dists.keys()):
      syn_yield = get_yield(syntax)
      if syn_yield in self._starts:
        for derived_root_cat in self._derived_categories_by_base[syn_yield]:
          new_yield = set_yield(syntax, derived_root_cat)
          if new_yield not in lf_syntax_ngrams:
            lf_syntax_ngrams[new_yield] = Distribution.uniform(lf_support)

    # Second distribution: P(pred | root)
    lf_yield_ngrams = self.lf_ngrams(
        conditioning_fn=lambda entry: [get_yield(entry.categ())], **kwargs)
    # Mix full-category and primitive-category predictions.
    lf_mixed_ngrams = ConditionalDistribution()
    for syntax in lf_syntax_ngrams:
      # # Mix distributions conditioned on the constituent primitives.
      # primitives = get_category_primitives(syntax)
      # prim_alpha = 1 / len(primitives)

      # Mix root-conditioned distribution and the full syntax-conditioned
      # distribution.
      yield_dist = lf_yield_ngrams[get_yield(syntax)]
      lf_mixed_ngrams[syntax] = lf_syntax_ngrams[syntax].mix(yield_dist, alpha)

    return lf_mixed_ngrams


class DerivedCategory(PrimitiveCategory):

  def __init__(self, name, base, source_name=None):
    self.name = name
    self.base = base
    self.source_name = source_name
    self._comparison_key = (name, base)

  def is_primitive(self):
    return True

  def is_function(self):
    return False

  def is_var(self):
    return False

  def categ(self):
    return self.name

  def substitute(self, subs):
    return self

  def can_unify(self, other):
    # The unification logic is critical here -- this determines how derived
    # categories are treated relative to their base categories.
    if other == self:
      return []

  def __str__(self):
    return "%s{%s}" % (self.name, self.base)

  def __repr__(self):
    return "%s{%s}{%s}" % (self.name, self.base, self.source_name)


def get_semantic_arity(category, arity_overrides=None):
  """
  Get the expected arity of a semantic form corresponding to some syntactic
  category.
  """
  arity_overrides = arity_overrides or {}
  if category in arity_overrides:
    return arity_overrides[category]

  if isinstance(category, PrimitiveCategory):
    return 0
  elif isinstance(category, FunctionalCategory):
    return 1 + get_semantic_arity(category.arg(), arity_overrides) \
      + get_semantic_arity(category.res(), arity_overrides)
  else:
    raise ValueError("unknown category type %r" % category)


def get_category_primitives(category):
  """
  Get the primitives involved in the given syntactic category.
  """
  if isinstance(category, PrimitiveCategory):
    return [category]
  elif isinstance(category, FunctionalCategory):
    return get_category_primitives(category.arg()) + \
        get_category_primitives(category.res())
  else:
    raise ValueError("unknown category type %r" % category)


def get_yield(category):
  """
  Get the primitive yield node of a syntactic category.
  """
  if isinstance(category, PrimitiveCategory):
    return category
  elif isinstance(category, FunctionalCategory):
    return get_yield(category.res())
  else:
    raise ValueError("unknown category type with instance %r" % category)


def set_yield(category, new_yield):
  if isinstance(category, PrimitiveCategory):
    return new_yield
  elif isinstance(category, FunctionalCategory):
    return FunctionalCategory(set_yield(category.res(), new_yield),
                              category.arg(), category.dir())
  else:
    raise ValueError("unknown category type of instance %r" % category)


def get_candidate_categories(lex, tokens, sentence, smooth=1e-3):
  """
  Find candidate categories for the given tokens which appear in `sentence` such
  that `sentence` yields a parse.

  Args:
    lex:
    tokens:
    sentence:
    smooth: If not `None`, add-k smooth the prior distribution over syntactic
      categories (where the float value of `smooth` specifies `k`).

  Returns:
    cat_dists: Dictionary mapping each token to a `Distribution` over
      categories.
  """
  assert set(tokens).issubset(set(sentence))

  # Make a minimal copy of `lex` which does not track semantics.
  lex = lex.clone(retain_semantics=False)

  # Remove entries for the queried tokens.
  for token in tokens:
    lex.set_entries(token, [])

  category_prior = lex.observed_category_distribution(
      exclude_tokens=set(tokens), soft_propagate_roots=True)
  if smooth is not None:
    for key in category_prior.keys():
      category_prior[key] += smooth
    category_prior = category_prior.normalize()
  L.debug("Smoothed category prior with soft root propagation: %s", category_prior)

  def score_cat_assignment(cat_assignment):
    """
    Assign a log-probability to a joint assignment of categories to tokens.
    """

    for token, category in zip(tokens, cat_assignment):
      lex.set_entries(token, [(category, None, 0.001)])

    # Attempt a parse.
    results = chart.WeightedCCGChartParser(lex, chart.DefaultRuleSet) \
        .parse(sentence, return_aux=True)
    if results:
      # Prior weight for category comes from lexicon.
      #
      # Might also downweight categories which require type-lifting parses by
      # default?
      score = 0
      for token, category in zip(tokens, cat_assignment):
        score += np.log(category_prior[category])

      # Likelihood weight comes from parse score
      score += np.log(sum(np.exp(weight)
                          for _, weight, _ in results))

      return score

    return -np.inf

  # NB does not cover the case where a single token needs multiple syntactic
  # interpretations for the sentence to parse
  cat_assignment_weights = {
    cat_assignment: score_cat_assignment(cat_assignment)
    for cat_assignment in itertools.product(category_prior.keys(), repeat=len(tokens))
  }

  cat_dists = defaultdict(Distribution)
  for cat_assignment, logp in cat_assignment_weights.items():
    for token, token_cat_assignment in zip(tokens, cat_assignment):
      cat_dists[token][token_cat_assignment] += np.exp(logp)

  # Normalize.
  cat_dists = {token: dist.normalize() for token, dist in cat_dists.items()}
  return cat_dists


def attempt_candidate_parse(lexicon, tokens, candidate_categories,
                            sentence, dummy_vars):
  """
  Attempt to parse a sentence, mapping `tokens` to new candidate
  lexical entries.

  Arguments:
    lexicon: Current lexicon. Will modify in place -- send a copy.
    tokens: List of string token(s) to be attempted.
    candidate_categories: List of candidate categories for each token (one per
      token).
    sentence: Sentence which we are attempting to parse.
  """

  get_arity = (lexicon.ontology and lexicon.ontology.get_expr_arity) \
      or get_semantic_arity

  # Prepare dummy variable which will be inserted into parse checks.
  token_vars = {token: copy(dummy_vars[token])
                for token in tokens}
  var_to_token = {var.name: token for token, var in token_vars.items()}
  sub_exprs = {token: l.FunctionVariableExpression(token_var)
               for token, token_var in token_vars.items()}

  lexicon = lexicon.clone()
  for token, syntax in zip(tokens, candidate_categories):
    lexicon.set_entries(token, [(syntax, sub_exprs[token], 1.0)])

  parse_results = []

  # First attempt a parse with only function application rules.
  results = chart.WeightedCCGChartParser(lexicon, ruleset=chart.ApplicationRuleSet) \
      .parse(sentence)

  for result in results:
    # Retrieve apparent types of each token's dummy var.
    apparent_types = defaultdict(list)
    def visit(expr):
      if isinstance(expr, l.FunctionVariableExpression):
        if expr.variable.name in var_to_token:
          apparent_types[var_to_token[expr.variable.name]].append(expr.type)
      elif isinstance(expr, l.ApplicationExpression):
        visit(expr.pred)
        for arg in expr.args:
          visit(arg)
      elif isinstance(expr, l.LambdaExpression):
        visit(expr.variable)
        visit(expr.term)

    sem = result.label()[0].semantics()
    visit(sem)

    # Make sure we inferred a type for every dummy var.
    assert(set(apparent_types.keys()) == set(token_vars.keys()))

    yield result, apparent_types

  # # Attempt to parse, allowing for function composition. In order to support
  # # this we need to pass a dummy expression which is a lambda.
  # arities = {expr: get_arity(expr) for expr in candidate_expressions}
  # max_arity = max(arities.values())

  # results, sub_expr_original = [], sub_expr
  # for arity in range(1, max(arities.values()) + 1):
  #   sub_expr = sub_expr_original

  #   variables = [l.Variable("z%i" % (i + 1)) for i in range(arity)]
  #   # Build curried application expression.
  #   term = sub_expr
  #   for variable in variables:
  #     term = l.ApplicationExpression(term, l.IndividualVariableExpression(variable))

  #   # Build surrounding lambda expression.
  #   sub_expr = term
  #   for variable in variables[::-1]:
  #     sub_expr = l.LambdaExpression(variable, sub_expr)

  #   lexicon._entries[token] = [Token(token, candidate_category, sub_expr)]
  #   results.extend(
  #       chart.WeightedCCGChartParser(lexicon, ruleset=chart.DefaultRuleSet).parse(sentence))

  # lexicon._entries[token] = []
  # return results, sub_target


def build_bootstrap_likelihood(lex, sentence, ontology,
                               alpha=0.25, meaning_prior_smooth=1e-3):
  """
  Prepare a likelihood function `p(meaning | syntax, sentence)` based on
  syntactic bootstrapping.

  Args:
    lex:
    sentence:
    ontology:
    alpha: Mixing parameter for bootstrapping distributions. See `alpha`
      parameter of `Lexicon.lf_ngrams_mixed`.

  Returns:
    likelihood_fn: A likelihood function to be used with `predict_zero_shot`.
  """
  # Prepare for syntactic bootstrap: pre-calculate distributions over semantic
  # form elements conditioned on syntactic category.
  lf_ngrams = lex.lf_ngrams_mixed(alpha=alpha, order=1,
                                  smooth=meaning_prior_smooth)
  for category in lf_ngrams:
    # Redistribute UNK probability uniformly across predicates not observed for
    # this category.
    unk_lf_prob = lf_ngrams[category].pop(None)
    unobserved_preds = set(f.name for f in ontology.functions) - set(lf_ngrams[category].keys())
    lf_ngrams[category].update({pred: unk_lf_prob / len(unobserved_preds)
                                for pred in unobserved_preds})

    L.info("% 20s %s", category,
           ", ".join("%.03f %s" % (prob, pred) for pred, prob
                     in sorted(lf_ngrams[category].items(), key=lambda x: x[1], reverse=True)))

  def likelihood_fn(tokens, categories, exprs, sentence_parse, model):
    likelihood = 0.0
    for token, category, expr in zip(tokens, categories, exprs):
      # Retrieve relevant bootstrap distribution p(meaning | syntax).
      cat_lf_ngrams = lf_ngrams[category]
      for predicate in expr.predicates():
        if predicate.name in cat_lf_ngrams:
          likelihood += np.log(cat_lf_ngrams[predicate.name])

    return likelihood

  return likelihood_fn


def likelihood_scene(tokens, categories, exprs, sentence_parse, model):
  """
  0-1 likelihood function, 1 when a sentence is true of the model and false
  otherwise.
  """
  try:
    return 0. if model.evaluate(sentence_parse) == True else -np.inf
  except:
    return -np.inf


def build_distant_likelihood(answer):
  """
  Prepare a likelihood function `p(meaning | syntax, sentence)` based on
  distant supervision.

  Args:
    answer: ground-truth answer

  Returns:
    likelihood_fn: A likelihood function to be used with `predict_zero_shot`.
  """
  def likelihood_fn(tokens, categories, exprs, sentence_parse, model):
    try:
      success = model.evaluate(sentence_parse) == answer
    except:
      success = None

    return 0.0 if success == True else -np.inf

  return likelihood_fn


def likelihood_2afc(tokens, categories, exprs, sentence_parse, models):
  """
  0-1 likelihood function for the 2AFC paradigm, where an uttered
  sentence is known to be true of at least one of two scenes.

  Args:
    models:

  Returns:
    log_likelihood:
  """
  model1, model2 = models
  try:
    model1_success = model1.evaluate(sentence_parse) == True
  except:
    model1_success = None
  try:
    model2_success = model2.evaluate(sentence_parse) == True
  except:
    model2_success = None

  return 0. if model1_success or model2_success else -np.inf


def predict_zero_shot(lex, tokens, candidate_syntaxes, sentence, ontology,
                      model, likelihood_fns, queue_limit=5):
  """
  Make zero-shot predictions of the posterior `p(syntax, meaning | sentence)`
  for each of `tokens`.

  Args:
    lex:
    tokens:
    candidate_syntaxes:
    sentence:
    ontology:
    model:
    likelihood_fns: Collection of likelihood functions
      `p(meanings | syntaxes, sentence, model)` used to score candidate
      meaning--syntax settings for a subset of `tokens`.  Each function should
      accept arguments `(tokens, candidate_categories, candidate_meanings,
      candidate_semantic_parse, model)`, where `tokens` are assigned specific
      categories given in `candidate_categories` and specific meanings given in
      `candidate_meanings`, yielding a single semantic analysis of the sentence
      `candidate_semantic_parse`. The function should return a log-likelihood
      `p(candidate_meanings | candidate_syntaxes, sentence, model)`.

  Returns:
    queues: A dictionary mapping each query token to a ranked sequence of
      candidates of the form
      `(logprob, (tokens, candidate_categories, candidate_semantics))`,
      describing a nonzero-probability novel mapping of a subset `tokens` to
      syntactic categories `candidate_categories` and meanings
      `candidate_semantics`. The log-probability value given is
      `p(meanings, syntaxes | sentence, model)`, under the relevant provided
      meaning likelihoods and the lexicon's distribution over syntactic forms.
    dummy_vars: TODO
  """

  get_arity = (lex.ontology and lex.ontology.get_expr_arity) \
      or get_semantic_arity

  # We will restrict semantic arities based on the observed arities available
  # for each category. Pre-calculate the necessary associations.
  category_sem_arities = lex.category_semantic_arities(soft_propagate_roots=True)

  def iter_expressions_for_arity(arity, max_depth=3):
    type_request = ontology.types[("e",) * (arity + 1)]
    return ontology.iter_expressions(max_depth=max_depth,
                                     type_request=type_request)

  def iter_expressions_for_category(cat):
    """
    Generate candidate semantic expressions for a lexical entry with the given
    syntactic category. (Forms type requests based on known associations
    between `cat` and semantic expressions.)
    """
    return itertools.chain.from_iterable(
        iter_expressions_for_arity(arity)
        for arity in category_sem_arities[cat])

  # Shared dummy variables which is included in candidate semantic forms, to be
  # replaced by all candidate lexical expressions and evaluated.
  dummy_vars = {token: l.Variable("F%03i" % i) for i, token in enumerate(tokens)}

  category_parse_results = {}
  candidate_queue = None
  for depth in trange(1, len(tokens) + 1, desc="Depths"):
    candidate_queue = UniquePriorityQueue(maxsize=queue_limit)

    token_combs = list(itertools.combinations(tokens, depth))
    for token_comb in tqdm(token_combs, desc="Token combinations"):
      token_syntaxes = [list(candidate_syntaxes[token].support) for token in token_comb]
      for syntax_comb in tqdm(itertools.product(*token_syntaxes),
                              total=np.prod(list(map(len, token_syntaxes))),
                              desc="Syntax combinations"):
        syntax_weights = [candidate_syntaxes[token][cat] for token, cat in zip(token_comb, syntax_comb)]
        if any(weight == 0 for weight in syntax_weights):
          continue

        # Attempt to parse with this joint syntactic assignment, and return the
        # resulting syntactic parses + sentence-level semantic forms, with
        # dummy variables in place of where the candidate expressions will go.
        results = attempt_candidate_parse(lex, token_comb,
                                          syntax_comb,
                                          sentence,
                                          dummy_vars)
        results = list(results)
        category_parse_results[syntax_comb] = results

        for result, apparent_types in results:
          candidate_exprs = [list(ontology.iter_expressions(max_depth=3, type_request=apparent_types[token][0]))
                             for token in token_comb]

          n_expr_combs = np.prod(list(map(len, candidate_exprs)))
          for expr_comb in tqdm(itertools.product(*candidate_exprs),
                                total=n_expr_combs,
                                desc="Expressions"):
            # Compute likelihood of this joint syntax--semantics assignment.
            likelihood = 0.0

            # Swap in semantic values for each token.
            sentence_semantics = result.label()[0].semantics()
            for token, token_expr in zip(token_comb, expr_comb):
              dummy_var = dummy_vars[token]
              sentence_semantics = sentence_semantics.replace(dummy_var, token_expr)
            sentence_semantics = sentence_semantics.simplify()

            # TODO find a better type-resolution solution. The current one will
            # be super slow -- we can pre-compute most of the relevant types
            # except for the item swapped in.
            try:
              lex.ontology.typecheck(sentence_semantics)
            except l.TypeException:
              continue

            # Compute p(meaning | syntax, sentence, parse)
            logp = sum(likelihood_fn(token_comb, syntax_comb, expr_comb,
                                    sentence_semantics, model)
                      for likelihood_fn in likelihood_fns)
            likelihood += np.exp(logp)

            # Add category priors.
            log_prior = sum(np.log(weight) for weight in syntax_weights)
            if log_prior == -np.inf or likelihood == 0:
              # Zero probability. Skip.
              continue

            joint_score = log_prior + np.log(likelihood)

            data = tuple_unordered([token_comb, syntax_comb, expr_comb])
            new_item = (joint_score, data)
            try:
              candidate_queue.put_nowait(new_item)
            except queue.Full:
              # See if this candidate is better than the worst item.
              worst = candidate_queue.get()
              if worst[0] < joint_score:
                replacement = new_item
              else:
                replacement = worst

              candidate_queue.put_nowait(replacement)

    if candidate_queue.qsize() > 0:
      # We have a result. Quit and don't search at higher depth.
      return candidate_queue, dummy_vars

  return candidate_queue, dummy_vars


def augment_lexicon(old_lex, query_tokens, query_token_syntaxes,
                    sentence, ontology, model, likelihood_fns,
                    beta=3.0,
                    **predict_zero_shot_args):
  """
  Augment a lexicon with candidate meanings for a given word using an abstract
  likelihood measure. (The induced meanings for the queried words must yield
  parses that have nonzero posterior probability, given the lexicon and
  `model`.)

  Candidate entries will be assigned relative weights according to a posterior
  distribution $P(word -> syntax, meaning | sentence, success_fn, lexicon)$. This
  distribution incorporates multiple prior and likelihood terms:

  1. A prior over syntactic categories (derived internally, by inspection of
     the current lexicon)
  2. A likelihood over meanings (specified by `likelihood_fns`)

  Arguments:
    old_lex: Existing lexicon which needs to be augmented. Does not write
      in-place.
    query_words: Set of tokens for which we need to search for novel lexical
      entries.
    query_word_syntaxes: Possible syntactic categories for each of the query
      words, as returned by `get_candidate_categories`.
    sentence: Token list sentence.
    ontology: Available logical ontology -- used to enumerate possible logical
      forms.
    model: Scene model which evaluates logical forms to answers.
    likelihood_fns: Sequence of functions describing zero-shot likelihoods
      `p(meanings | syntaxes, sentence, model)`. See `predict_zero_shot` for
      more information.
    beta: Total mass to assign to novel candidate lexical entries per each
      wordform. (Mass will be divided according to the posterior distribution
      given above.)
  """

  # Target lexicon to be returned.
  lex = old_lex.clone()

  ranked_candidates, dummy_vars = \
      predict_zero_shot(lex, query_tokens, query_token_syntaxes, sentence,
                        ontology, model, likelihood_fns, **predict_zero_shot_args)

  candidates = sorted(ranked_candidates.queue, key=lambda item: -item[0])
  new_entries = {token: Counter() for token in query_tokens}
  # Calculate marginal p(syntax, meaning | sentence) for each token.
  for logp, (tokens, syntaxes, meanings) in candidates:
    for token, syntax, meaning in zip(tokens, syntaxes, meanings):
      new_entries[token][syntax, meaning] += np.exp(logp)

  if all(len(candidates) == 0 for candidates in new_entries.values()):
    raise NoParsesError("Failed to derive any meanings for tokens %s."
                        % query_tokens, sentence)

  # Construct a new lexicon.
  for token, candidates in new_entries.items():
    total_mass = sum(candidates.values())
    if len(candidates) > 0:
      lex.set_entries(token,
                      [(syntax, meaning, weight / total_mass * beta)
                       for (syntax, meaning), weight in candidates.items()])

      L.info("Inferred %i novel entries for token %s:", len(candidates), token)
      for entry, weight in sorted(candidates.items(), key=lambda x: x[1], reverse=True):
        L.info("%.4f %s", weight / total_mass * beta, entry)

  return lex


def augment_lexicon_distant(old_lex, query_tokens, query_token_syntaxes,
                            sentence, ontology, model, likelihood_fns, answer,
                            **augment_kwargs):
  """
  Augment a lexicon with candidate meanings for a given word using distant
  supervision. (Find word meanings such that the whole sentence meaning yields
  an expected `answer`.)

  For argument documentation, see `augment_lexicon`.
  """
  likelihood_fns = (build_distant_likelihood(answer),) + \
      tuple(likelihood_fns)

  return augment_lexicon(old_lex, query_tokens, query_token_syntaxes,
                         sentence, ontology, model, likelihood_fns,
                         **augment_kwargs)


def augment_lexicon_cross_situational(old_lex, query_tokens, query_token_syntaxes,
                                      sentence, ontology, model, likelihood_fns,
                                      **augment_kwargs):
  likelihood_fns = (likelihood_scene,) + tuple(likelihood_fns)
  return augment_lexicon(old_lex, query_tokens, query_token_syntaxes,
                         sentence, ontology, model, likelihood_fns,
                         **augment_kwargs)


def augment_lexicon_2afc(old_lex, query_tokens, query_token_syntaxes,
                         sentence, ontology, models, likelihood_fns,
                         **augment_kwargs):
  """
  Augment a lexicon with candidate meanings for a given word using 2AFC
  supervision. (We assume that the uttered sentence is true of at least one of
  the 2 scenes given in the tuple `models`.)

  For argument documentation, see `augment_lexicon`.
  """
  likelihood_fns = (likelihood_2afc,) + tuple(likelihood_fns)
  return augment_lexicon(old_lex, query_tokens, query_token_syntaxes,
                         sentence, ontology, models, likelihood_fns,
                         **augment_kwargs)


def filter_lexicon_entry(lexicon, entry, sentence, lf):
  """
  Filter possible syntactic/semantic mappings for a given lexicon entry s.t.
  the given sentence renders the given LF, holding everything else
  constant.

  This process is of course not fail-safe -- the rest of the lexicon must
  provide the necessary definitions to guarantee that any valid parse can
  result.

  Args:
    lexicon: CCGLexicon
    entry: string word
    sentence: list of word tokens, must contain `entry`
    lf: logical form string
  """
  if entry not in sentence:
    raise ValueError("Sentence does not contain given entry")

  entry_idxs = [i for i, val in enumerate(sentence) if val == entry]
  parse_results = chart.WeightedCCGChartParser(lexicon).parse(sentence, True)

  valid_cands = [set() for _ in entry_idxs]
  for _, _, edge_cands in parse_results:
    for entry_idx, valid_cands_set in zip(entry_idxs, valid_cands):
      valid_cands_set.add(edge_cands[entry_idx])

  # Find valid interpretations across all uses of the word in the
  # sentence.
  valid_cands = list(reduce(lambda x, y: x & y, valid_cands))
  if not valid_cands:
    raise ValueError("no consistent interpretations of word found.")

  new_lex = lexicon.clone()
  new_lex.set_entries(entry, [(cand.token().categ(), cand.token().semantics(), cand.token().weight())
                              for cand in valid_cands])

  return new_lex
