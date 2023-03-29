"""
Combinators and related utilities for operating on CCG syntactic types and
semantic forms.
"""

from abc import ABCMeta, abstractmethod
from copy import deepcopy
import itertools

from nltk.ccg.api import FunctionalCategory, PrimitiveCategory
from six import add_metaclass

from pyccg import logic as l


@add_metaclass(ABCMeta)
class UndirectedBinaryCombinator(object):
  """
  Abstract class for representing a binary combinator.
  Merely defines functions for checking if the function and argument
  are able to be combined, and what the resulting category is.

  Note that as no assumptions are made as to direction, the unrestricted
  combinators can perform all backward, forward and crossed variations
  of the combinators; these restrictions must be added in the rule
  class.
  """

  def __init__(self):
    self._ontology = None

  def set_ontology(self, ontology):
    self._ontology = ontology

  @abstractmethod
  def can_combine(self, function, argument):
    pass

  @abstractmethod
  def combine(self, function, argument):
    pass


@add_metaclass(ABCMeta)
class DirectedBinaryCombinator(object):
  """
  Wrapper for the undirected binary combinator.
  It takes left and right categories, and decides which is to be
  the function, and which the argument.
  It then decides whether or not they can be combined.
  """

  def __init__(self):
    self._ontology = None

  def set_ontology(self, ontology):
    self._ontology = ontology

  @abstractmethod
  def can_combine(self, left, right):
    pass

  @abstractmethod
  def combine(self, left, right):
    pass


class ForwardCombinator(DirectedBinaryCombinator):
  """
  Class representing combinators where the primary functor is on the left.

  Takes an undirected combinator, and a predicate which adds constraints
  restricting the cases in which it may apply.
  """

  def __init__(self, combinator, predicate, suffix=''):
    super().__init__()
    self._combinator = combinator
    self._predicate = predicate
    self._suffix = suffix

  def set_ontology(self, ontology):
    self._combinator.set_ontology(ontology)

  def can_combine(self, left, right):
    return self._combinator.can_combine(left, right) and self._predicate(
      left, right
    )

  def combine(self, left, right):
    for result in self._combinator.combine(left, right):
      yield result

  def __str__(self):
    return ">%s%s" % (self._combinator, self._suffix)


class BackwardCombinator(DirectedBinaryCombinator):
  """
  The backward equivalent of the ForwardCombinator class.
  """

  def __init__(self, combinator, predicate, suffix=''):
    super().__init__()
    self._combinator = combinator
    self._predicate = predicate
    self._suffix = suffix

  def set_ontology(self, ontology):
    self._combinator.set_ontology(ontology)

  def can_combine(self, left, right):
    return self._combinator.can_combine(right, left) and self._predicate(
      left, right
    )

  def combine(self, left, right):
    for result in self._combinator.combine(right, left):
      yield result

  def __str__(self):
    return "<%s%s" % (self._combinator, self._suffix)


class UndirectedFunctionApplication(UndirectedBinaryCombinator):
  """
  Class representing function application.
  Implements rules of the form:
  X/Y Y -> X (>)
  And the corresponding backwards application rule
  """

  def _typecheck(self, function, argument):
    fsem, asem = function.semantics(), argument.semantics()
    if fsem is None or asem is None:
      return True

    ftype, atype = fsem.type, asem.type
    if ftype is None or atype is None:
      return True
    if not isinstance(ftype, l.ComplexType):
      return False
    if not atype.matches(ftype.first):
      return False

    return True

  def can_combine(self, function, argument):
    if not function.categ().is_function():
      return False

    if function.categ().arg().can_unify(argument.categ()) is None:
      return False

    if not self._typecheck(function, argument):
      return False

    return True

  def combine(self, function, argument):
    if not function.categ().is_function():
      return

    subs = function.categ().arg().can_unify(argument.categ())
    if subs is None:
      return

    if not self._typecheck(function, argument):
      return

    categ = function.categ().res().substitute(subs)
    fsem, asem = function.semantics(), argument.semantics()
    if fsem is not None and asem is not None:
      # if function's type is not set, do the best we can to infer.
      if self._ontology is not None \
          and isinstance(fsem, l.FunctionVariableExpression) \
          and (fsem.type is None or fsem.type == self._ontology.types.ANY_TYPE):
        fsem.variable.type = self._ontology.types[asem.type.flat + ("e",)]
        fsem.type = fsem.variable.type

      semantics = l.ApplicationExpression(function.semantics(), argument.semantics()).simplify()
    else:
      semantics = None

    yield categ, semantics

  def __str__(self):
    return ''


# Predicates for function application.

# Ensures the left functor takes an argument on the right
def forwardOnly(left, right):
  return left.categ().dir().is_forward()


# Ensures the right functor takes an argument on the left
def backwardOnly(left, right):
  return right.categ().dir().is_backward()


# Application combinator instances
ForwardApplication = ForwardCombinator(UndirectedFunctionApplication(), forwardOnly)
BackwardApplication = BackwardCombinator(UndirectedFunctionApplication(), backwardOnly)


class UndirectedComposition(UndirectedBinaryCombinator):
  """
  Functional composition (harmonic) combinator.
  Implements rules of the form
  X/Y Y/Z -> X/Z (B>)
  And the corresponding backwards and crossed variations.
  """

  def can_combine(self, function, argument):
    # Can only combine two functions, and both functions must
    # allow composition.
    if not (function.categ().is_function() and argument.categ().is_function()):
      return False
    if function.categ().dir().can_compose() and argument.categ().dir().can_compose():
      if function.categ().arg().can_unify(argument.categ().res()) is None:
        return False

    fsem, asem = function.semantics(), argument.semantics()
    if fsem is not None and asem is not None:
      if not isinstance(argument.semantics(), l.LambdaExpression):
        return False

      if not asem.term.type.matches(fsem.type.first):
        return False

    return True

  def combine(self, function, argument):
    if not (function.categ().is_function() and argument.categ().is_function()):
      return
    if function.categ().dir().can_compose() and argument.categ().dir().can_compose():
      subs = function.categ().arg().can_unify(argument.categ().res())
      if subs is not None:
        categ = FunctionalCategory(
            function.categ().res().substitute(subs),
            argument.categ().arg().substitute(subs),
            argument.categ().dir())

        fsem, asem = function.semantics(), argument.semantics()
        if fsem is not None and asem is not None:
          semantics = l.LambdaExpression(asem.variable, l.ApplicationExpression(fsem, asem.term).simplify())
        else:
          semantics = None

        yield categ, semantics

  def __str__(self):
    return 'B'


# Predicates for restricting application of straight composition.
def bothForward(left, right):
  return left.categ().dir().is_forward() and right.categ().dir().is_forward()


def bothBackward(left, right):
  return left.categ().dir().is_backward() and right.categ().dir().is_backward()


# Predicates for crossed composition
def crossedDirs(left, right):
  return left.categ().dir().is_forward() and right.categ().dir().is_backward()


def backwardBxConstraint(left, right):
  # The functors must be crossed inwards
  if not crossedDirs(left, right):
    return False
  # Permuting combinators must be allowed
  if not left.categ().dir().can_cross() and right.categ().dir().can_cross():
    return False
  # The resulting argument category is restricted to be primitive
  return left.categ().arg().is_primitive()


# Straight composition combinators
ForwardComposition = ForwardCombinator(UndirectedComposition(), forwardOnly)
BackwardComposition = BackwardCombinator(UndirectedComposition(), backwardOnly)

# Backward crossed composition
BackwardBx = BackwardCombinator(
  UndirectedComposition(), backwardBxConstraint, suffix='x'
)


class UndirectedSubstitution(UndirectedBinaryCombinator):
  """
  Substitution (permutation) combinator.
  Implements rules of the form
  Y/Z (X\Y)/Z -> X/Z (<Sx)
  And other variations.
  """

  def can_combine(self, function, argument):
    if function.categ().is_primitive() or argument.categ().is_primitive():
      return False

    # These could potentially be moved to the predicates, as the
    # constraints may not be general to all languages.
    if function.categ().res().is_primitive():
      return False
    if not function.categ().arg().is_primitive():
      return False

    if not (function.categ().dir().can_compose() and argument.categ().dir().can_compose()):
      return False

    fsem, asem = function.semantics(), argument.semantics()
    # F must be a lambda expression with 2 arguments
    if not (isinstance(fsem, l.LambdaExpression) and isinstance(fsem.term, l.LambdaExpression)):
      return False
    # A must be a lambda expression
    if not isinstance(asem, l.LambdaExpression):
      return False

    return (function.categ().res().arg() == argument.categ().res()) and (
      function.categ().arg() == argument.categ().arg()
    )

  def combine(self, function, argument):
    if self.can_combine(function, argument):
      categ = FunctionalCategory(
        function.categ().res().res(), argument.categ().arg(), argument.categ().dir()
      )

      # TODO type-inference
      fsem, asem = function.semantics(), argument.semantics()
      new_arg = l.ApplicationExpression(asem, l.VariableExpression(fsem.variable)).simplify()
      new_term = l.ApplicationExpression(fsem.term, new_arg).simplify()
      semantics = l.LambdaExpression(fsem.variable, new_term)

      yield categ, semantics

  def __str__(self):
    return 'S'


# Predicate for forward substitution
def forwardSConstraint(left, right):
  if not bothForward(left, right):
    return False
  return left.categ().res().dir().is_forward() and left.categ().arg().is_primitive()


# Predicate for backward crossed substitution
def backwardSxConstraint(left, right):
  if not left.categ().dir().can_cross() and right.categ().dir().can_cross():
    return False
  if not bothForward(left, right):
    return False
  return right.categ().res().dir().is_backward() and right.categ().arg().is_primitive()


# Instances of substitution combinators
ForwardSubstitution = ForwardCombinator(UndirectedSubstitution(), forwardSConstraint)
BackwardSx = BackwardCombinator(UndirectedSubstitution(), backwardSxConstraint, 'x')


# Retrieves the left-most functional category.
# ie, (N\N)/(S/NP) => N\N
def innermostFunction(categ):
  while categ.res().is_function():
    categ = categ.res()
  return categ


class UndirectedTypeRaise(UndirectedBinaryCombinator):
  """
  Undirected combinator for type raising.
  """

  def can_combine(self, function, arg):
    # The argument must be a function.
    # The restriction that arg.res() must be a function
    # merely reduces redundant type-raising; if arg.res() is
    # primitive, we have:
    # X Y\X =>(<T) Y/(Y\X) Y\X =>(>) Y
    # which is equivalent to
    # X Y\X =>(<) Y
    if not (arg.categ().is_function() and arg.categ().res().is_function()):
      return False

    arg = innermostFunction(arg.categ())

    # left, arg_categ are undefined!
    subs = left.categ().can_unify(arg_categ.arg())
    if subs is not None:
      return True
    return False

  def combine(self, function, arg):
    if not (
      function.categ().is_primitive() and arg.categ().is_function() and arg.categ().res().is_function()
    ):
      return

    # Type-raising matches only the innermost application.
    arg = innermostFunction(arg.categ())

    subs = function.categ().can_unify(arg.arg())
    if subs is not None:
      xcat = arg.res().substitute(subs)
      categ = FunctionalCategory(
        xcat, FunctionalCategory(xcat, function.categ(), arg.dir()), -(arg.dir())
      )

      # compute semantics
      semantics = function.semantics()
      if semantics is not None:
        core = deepcopy(function.semantics())
        parent = None
        while isinstance(core, l.LambdaExpression):
          parent = core
          core = core.term

        var = l.Variable("F")
        while var in core.free():
          var = l.unique_variable(pattern=var)
        core = l.ApplicationExpression(l.FunctionVariableExpression(var), core)

        if parent is not None:
          parent.term = core
        else:
          semantics = core

        semantics = l.LambdaExpression(var, semantics)

      yield categ, semantics

  def __str__(self):
    return 'T'


# Predicates for type-raising
# The direction of the innermost category must be towards
# the primary functor.
# The restriction that the variable must be primitive is not
# common to all versions of CCGs; some authors have other restrictions.
def forwardTConstraint(left, right):
  arg = innermostFunction(right.categ())
  return arg.dir().is_backward() and arg.res().is_primitive()


def backwardTConstraint(left, right):
  arg = innermostFunction(left.categ())
  return arg.dir().is_forward() and arg.res().is_primitive()


# Instances of type-raising combinators
ForwardT = ForwardCombinator(UndirectedTypeRaise(), forwardTConstraint)
BackwardT = BackwardCombinator(UndirectedTypeRaise(), backwardTConstraint)


def category_search_replace(expr, search, replace):
  """
  Return all reanalyses of the syntactic category expression `expr` which
  involve the (primitive) derived category. Effectively a search-replace
  operation.

  Args:
    expr: `AbstractCCGCategory` expression
    search: `AbstractCCGCategory` expression
    replace: `AbstractCCGCategory` expression

  Returns:
    Set of modified forms of `expr` where combinations of instances of
    `search` are replaced with `replace`.
  """
  # Find all reanalyses of the functional category expression `expr` which involve the
  # derived category.
  def traverse(node):
    if node == search:
      return [replace]
    elif isinstance(node, FunctionalCategory):
      left_subresults = [node.res()] + traverse(node.res())
      right_subresults = [node.arg()] + traverse(node.arg())

      results = [FunctionalCategory(left_subresult, right_subresult, node.dir())
                 for left_subresult, right_subresult
                 in itertools.product(left_subresults, right_subresults)
                 if not (left_subresult == node.res() and right_subresult == node.arg())]
      return results
    else:
      return []

  return set(traverse(expr))

