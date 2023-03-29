# -*- coding: utf-8
"""
An API for describing a typed lambda calculus, and methods for enumerating and
executing expressions in that language.
"""

from collections import defaultdict
from copy import copy, deepcopy
import functools
import inspect
import itertools
import logging
import operator
import re

from nltk.internals import Counter as nCounter
from nltk.sem import logic as l
from nltk.util import Trie
from six import string_types

L = logging.getLogger(__name__)


TypeException = l.TypeException
InconsistentTypeHierarchyException = l.InconsistentTypeHierarchyException
TypeResolutionException = l.TypeResolutionException
IllegalTypeException = l.IllegalTypeException


APP = "APP"
_counter = nCounter()

class Tokens(object):
    LAMBDA = '\\';     LAMBDA_LIST = ['\\']

    #Quantifiers
    EXISTS = 'exists'; EXISTS_LIST = ['some', 'exists', 'exist']
    ALL = 'all';       ALL_LIST = ['all', 'forall']

    #Punctuation
    DOT = '.'
    OPEN = '('
    CLOSE = ')'
    COMMA = ','

    #Operations
    NOT = '-';         NOT_LIST = ['not', '-', '!']
    AND = '&';         AND_LIST = ['and', '&', '^']
    OR = '|';          OR_LIST = ['or', '|']
    IMP = '->';        IMP_LIST = ['implies', '->', '=>']
    IFF = '<->';       IFF_LIST = ['iff', '<->', '<=>']
    EQ = '=';          EQ_LIST = ['=', '==']
    NEQ = '!=';        NEQ_LIST = ['!=']

    #Collections of tokens
    BINOPS = AND_LIST + OR_LIST + IMP_LIST + IFF_LIST
    QUANTS = EXISTS_LIST + ALL_LIST
    PUNCT = [DOT, OPEN, CLOSE, COMMA]

    TOKENS = BINOPS + EQ_LIST + NEQ_LIST + QUANTS + LAMBDA_LIST + PUNCT + NOT_LIST

    #Special
    SYMBOLS = [x for x in TOKENS if re.match(r'^[-\\.(),!&^|>=<]*$', x)]


def boolean_ops():
    """
    Boolean operators
    """
    names =  ["negation", "conjunction", "disjunction", "implication", "equivalence"]
    for pair in zip(names, [Tokens.NOT, Tokens.AND, Tokens.OR, Tokens.IMP, Tokens.IFF]):
        print("%-15s\t%s" %  pair)

def equality_preds():
    """
    Equality predicates
    """
    names =  ["equality", "inequality"]
    for pair in zip(names, [Tokens.EQ, Tokens.NEQ]):
        print("%-15s\t%s" %  pair)

def binding_ops():
    """
    Binding operators
    """
    names =  ["existential", "universal", "lambda"]
    for pair in zip(names, [Tokens.EXISTS, Tokens.ALL, Tokens.LAMBDA]):
        print("%-15s\t%s" %  pair)


class LogicParser(object):
    """A lambda calculus expression parser."""

    def __init__(self, type_check=False):
        """
        :param type_check: bool should type checking be performed?
        to their types.
        """
        assert isinstance(type_check, bool)

        self._currentIndex = 0
        self._buffer = []
        self.type_check = type_check

        """A list of tuples of quote characters.  The 4-tuple is comprised
        of the start character, the end character, the escape character, and
        a boolean indicating whether the quotes should be included in the
        result. Quotes are used to signify that a token should be treated as
        atomic, ignoring any special characters within the token.  The escape
        character allows the quote end character to be used within the quote.
        If True, the boolean indicates that the final token should contain the
        quote and escape characters.
        This method exists to be overridden"""
        self.quote_chars = []

        self.operator_precedence = dict(
                           [(x,1) for x in Tokens.LAMBDA_LIST]             + \
                           [(x,2) for x in Tokens.NOT_LIST]                + \
                           [(APP,3)]                                       + \
                           [(x,4) for x in Tokens.EQ_LIST+Tokens.NEQ_LIST] + \
                           [(x,5) for x in Tokens.QUANTS]                  + \
                           [(x,6) for x in Tokens.AND_LIST]                + \
                           [(x,7) for x in Tokens.OR_LIST]                 + \
                           [(x,8) for x in Tokens.IMP_LIST]                + \
                           [(x,9) for x in Tokens.IFF_LIST]                + \
                           [(None,10)])
        self.right_associated_operations = [APP]

    def parse(self, data, signature=None):
        """
        Parse the expression.

        :param data: str for the input to be parsed
        :param signature: ``dict<str, str>`` that maps variable names to type
        strings
        :returns: a parsed Expression
        """
        data = data.rstrip()

        self._currentIndex = 0
        self._buffer, mapping = self.process(data)

        try:
            result = self.process_next_expression(None)
            if self.inRange(0):
                raise UnexpectedTokenException(self._currentIndex+1, self.token(0))
        except LogicalExpressionException as e:
            msg = '%s\n%s\n%s^' % (e, data, ' '*mapping[e.index-1])
            raise LogicalExpressionException(None, msg)

        if self.type_check:
            result.typecheck(signature)

        return result

    def process(self, data):
        """Split the data into tokens"""
        out = []
        mapping = {}
        tokenTrie = Trie(self.get_all_symbols())
        token = ''
        data_idx = 0
        token_start_idx = data_idx
        while data_idx < len(data):
            cur_data_idx = data_idx
            quoted_token, data_idx = self.process_quoted_token(data_idx, data)
            if quoted_token:
                if not token:
                    token_start_idx = cur_data_idx
                token += quoted_token
                continue

            st = tokenTrie
            c = data[data_idx]
            symbol = ''
            while c in st:
                symbol += c
                st = st[c]
                if len(data)-data_idx > len(symbol):
                    c = data[data_idx+len(symbol)]
                else:
                    break
            if Trie.LEAF in st:
                #token is a complete symbol
                if token:
                    mapping[len(out)] = token_start_idx
                    out.append(token)
                    token = ''
                mapping[len(out)] = data_idx
                out.append(symbol)
                data_idx += len(symbol)
            else:
                if data[data_idx] in ' \t\n': #any whitespace
                    if token:
                        mapping[len(out)] = token_start_idx
                        out.append(token)
                        token = ''
                else:
                    if not token:
                        token_start_idx = data_idx
                    token += data[data_idx]
                data_idx += 1
        if token:
            mapping[len(out)] = token_start_idx
            out.append(token)
        mapping[len(out)] = len(data)
        mapping[len(out)+1] = len(data)+1
        return out, mapping

    def process_quoted_token(self, data_idx, data):
        token = ''
        c = data[data_idx]
        i = data_idx
        for start, end, escape, incl_quotes in self.quote_chars:
            if c == start:
                if incl_quotes:
                    token += c
                i += 1
                while data[i] != end:
                    if data[i] == escape:
                        if incl_quotes:
                            token += data[i]
                        i += 1
                        if len(data) == i: #if there are no more chars
                            raise LogicalExpressionException(None, "End of input reached.  "
                                    "Escape character [%s] found at end."
                                    % escape)
                        token += data[i]
                    else:
                        token += data[i]
                    i += 1
                    if len(data) == i:
                        raise LogicalExpressionException(None, "End of input reached.  "
                                             "Expected: [%s]" % end)
                if incl_quotes:
                    token += data[i]
                i += 1
                if not token:
                    raise LogicalExpressionException(None, 'Empty quoted token found')
                break
        return token, i

    def get_all_symbols(self):
        """This method exists to be overridden"""
        return Tokens.SYMBOLS

    def inRange(self, location):
        """Return TRUE if the given location is within the buffer"""
        return self._currentIndex+location < len(self._buffer)

    def token(self, location=None):
        """Get the next waiting token.  If a location is given, then
        return the token at currentIndex+location without advancing
        currentIndex; setting it gives lookahead/lookback capability."""
        try:
            if location is None:
                tok = self._buffer[self._currentIndex]
                self._currentIndex += 1
            else:
                tok = self._buffer[self._currentIndex+location]
            return tok
        except IndexError:
            raise ExpectedMoreTokensException(self._currentIndex+1)

    def isvariable(self, tok):
        return tok not in Tokens.TOKENS

    def process_next_expression(self, context):
        """Parse the next complete expression from the stream and return it."""
        try:
            tok = self.token()
        except ExpectedMoreTokensException:
            raise ExpectedMoreTokensException(self._currentIndex+1, message='Expression expected.')

        accum = self.handle(tok, context)

        if not accum:
            raise UnexpectedTokenException(self._currentIndex, tok, message='Expression expected.')

        return self.attempt_adjuncts(accum, context)

    def handle(self, tok, context):
        """This method is intended to be overridden for logics that
        use different operators or expressions"""
        if self.isvariable(tok):
            return self.handle_variable(tok, context)

        elif tok in Tokens.NOT_LIST:
            return self.handle_negation(tok, context)

        elif tok in Tokens.LAMBDA_LIST:
            return self.handle_lambda(tok, context)

        elif tok in Tokens.QUANTS:
            return self.handle_quant(tok, context)

        elif tok == Tokens.OPEN:
            return self.handle_open(tok, context)

    def attempt_adjuncts(self, expression, context):
        cur_idx = None
        while cur_idx != self._currentIndex: #while adjuncts are added
            cur_idx = self._currentIndex
            expression = self.attempt_EqualityExpression(expression, context)
            expression = self.attempt_ApplicationExpression(expression, context)
            expression = self.attempt_BooleanExpression(expression, context)
        return expression

    def handle_negation(self, tok, context):
        return self.make_NegatedExpression(self.process_next_expression(Tokens.NOT))

    def make_NegatedExpression(self, expression):
        return NegatedExpression(expression)

    def handle_variable(self, tok, context):
        #It's either: 1) a predicate expression: sees(x,y)
        #             2) an application expression: P(x)
        #             3) a solo variable: john OR x
        accum = self.make_VariableExpression(tok)
        if self.inRange(0) and self.token(0) == Tokens.OPEN:
            #The predicate has arguments
            if not isinstance(accum, FunctionVariableExpression) and \
               not isinstance(accum, ConstantExpression):
                raise LogicalExpressionException(self._currentIndex,
                                     "'%s' is an illegal predicate name.  "
                                     "Individual variables may not be used as "
                                     "predicates." % tok)
            self.token() #swallow the Open Paren

            #curry the arguments
            accum = self.make_ApplicationExpression(accum, self.process_next_expression(APP))
            while self.inRange(0) and self.token(0) == Tokens.COMMA:
                self.token() #swallow the comma
                accum = self.make_ApplicationExpression(accum, self.process_next_expression(APP))
            self.assertNextToken(Tokens.CLOSE)
        return accum

    def get_next_token_variable(self, description):
        try:
            tok = self.token()
        except ExpectedMoreTokensException as e:
            raise ExpectedMoreTokensException(e.index, 'Variable expected.')
        if isinstance(self.make_VariableExpression(tok), ConstantExpression):
            raise LogicalExpressionException(self._currentIndex,
                                 "'%s' is an illegal variable name.  "
                                 "Constants may not be %s." % (tok, description))
        return Variable(tok)

    def handle_lambda(self, tok, context):
        # Expression is a lambda expression
        if not self.inRange(0):
            raise ExpectedMoreTokensException(self._currentIndex+2,
                                              message="Variable and Expression expected following lambda operator.")
        vars = [self.get_next_token_variable('abstracted')]
        while True:
            if not self.inRange(0) or (self.token(0) == Tokens.DOT and not self.inRange(1)):
                raise ExpectedMoreTokensException(self._currentIndex+2, message="Expression expected.")
            if not self.isvariable(self.token(0)):
                break
            # Support expressions like: \x y.M == \x.\y.M
            vars.append(self.get_next_token_variable('abstracted'))
        if self.inRange(0) and self.token(0) == Tokens.DOT:
            self.token() #swallow the dot

        accum = self.process_next_expression(tok)
        while vars:
            accum = self.make_LambdaExpression(vars.pop(), accum)
        return accum

    def handle_quant(self, tok, context):
        # Expression is a quantified expression: some x.M
        factory = self.get_QuantifiedExpression_factory(tok)

        if not self.inRange(0):
            raise ExpectedMoreTokensException(self._currentIndex+2,
                                              message="Variable and Expression expected following quantifier '%s'." % tok)
        vars = [self.get_next_token_variable('quantified')]
        while True:
            if not self.inRange(0) or (self.token(0) == Tokens.DOT and not self.inRange(1)):
                raise ExpectedMoreTokensException(self._currentIndex+2, message="Expression expected.")
            if not self.isvariable(self.token(0)):
                break
            # Support expressions like: some x y.M == some x.some y.M
            vars.append(self.get_next_token_variable('quantified'))
        if self.inRange(0) and self.token(0) == Tokens.DOT:
            self.token() #swallow the dot

        accum = self.process_next_expression(tok)
        while vars:
            accum = self.make_QuanifiedExpression(factory, vars.pop(), accum)
        return accum

    def get_QuantifiedExpression_factory(self, tok):
        """This method serves as a hook for other logic parsers that
        have different quantifiers"""
        if tok in Tokens.EXISTS_LIST:
            return ExistsExpression
        elif tok in Tokens.ALL_LIST:
            return AllExpression
        else:
            self.assertToken(tok, Tokens.QUANTS)

    def make_QuanifiedExpression(self, factory, variable, term):
        return factory(variable, term)

    def handle_open(self, tok, context):
        #Expression is in parens
        accum = self.process_next_expression(None)
        self.assertNextToken(Tokens.CLOSE)
        return accum

    def attempt_EqualityExpression(self, expression, context):
        """Attempt to make an equality expression.  If the next token is an
        equality operator, then an EqualityExpression will be returned.
        Otherwise, the parameter will be returned."""
        if self.inRange(0):
            tok = self.token(0)
            if tok in Tokens.EQ_LIST + Tokens.NEQ_LIST and self.has_priority(tok, context):
                self.token() #swallow the "=" or "!="
                expression = self.make_EqualityExpression(expression, self.process_next_expression(tok))
                if tok in Tokens.NEQ_LIST:
                    expression = self.make_NegatedExpression(expression)
        return expression

    def make_EqualityExpression(self, first, second):
        """This method serves as a hook for other logic parsers that
        have different equality expression classes"""
        return EqualityExpression(first, second)

    def attempt_BooleanExpression(self, expression, context):
        """Attempt to make a boolean expression.  If the next token is a boolean
        operator, then a BooleanExpression will be returned.  Otherwise, the
        parameter will be returned."""
        while self.inRange(0):
            tok = self.token(0)
            factory = self.get_BooleanExpression_factory(tok)
            if factory and self.has_priority(tok, context):
                self.token() #swallow the operator
                expression = self.make_BooleanExpression(factory, expression,
                                                         self.process_next_expression(tok))
            else:
                break
        return expression

    def get_BooleanExpression_factory(self, tok):
        """This method serves as a hook for other logic parsers that
        have different boolean operators"""
        if tok in Tokens.AND_LIST:
            return AndExpression
        elif tok in Tokens.OR_LIST:
            return OrExpression
        elif tok in Tokens.IMP_LIST:
            return ImpExpression
        elif tok in Tokens.IFF_LIST:
            return IffExpression
        else:
            return None

    def make_BooleanExpression(self, factory, first, second):
        return factory(first, second)

    def attempt_ApplicationExpression(self, expression, context):
        """Attempt to make an application expression.  The next tokens are
        a list of arguments in parens, then the argument expression is a
        function being applied to the arguments.  Otherwise, return the
        argument expression."""
        if self.has_priority(APP, context):
            if self.inRange(0) and self.token(0) == Tokens.OPEN:
                if not isinstance(expression, LambdaExpression) and \
                   not isinstance(expression, ApplicationExpression) and \
                   not isinstance(expression, FunctionVariableExpression) and \
                   not isinstance(expression, ConstantExpression):
                    raise LogicalExpressionException(self._currentIndex,
                                         ("The function '%s" % expression) +
                                         "' is not a Lambda Expression, an "
                                         "Application Expression, or a "
                                         "functional predicate, so it may "
                                         "not take arguments.")
                self.token() #swallow then open paren
                #curry the arguments
                accum = self.make_ApplicationExpression(expression, self.process_next_expression(APP))
                while self.inRange(0) and self.token(0) == Tokens.COMMA:
                    self.token() #swallow the comma
                    accum = self.make_ApplicationExpression(accum, self.process_next_expression(APP))
                self.assertNextToken(Tokens.CLOSE)
                return accum
        return expression

    def make_ApplicationExpression(self, function, argument):
        return ApplicationExpression(function, argument)

    def make_VariableExpression(self, name):
        return VariableExpression(Variable(name))

    def make_LambdaExpression(self, variable, term):
        return LambdaExpression(variable, term)

    def has_priority(self, operation, context):
        return self.operator_precedence[operation] < self.operator_precedence[context] or \
               (operation in self.right_associated_operations and \
                self.operator_precedence[operation] == self.operator_precedence[context])

    def assertNextToken(self, expected):
        try:
            tok = self.token()
        except ExpectedMoreTokensException as e:
            raise ExpectedMoreTokensException(e.index, message="Expected token '%s'." % expected)

        if isinstance(expected, list):
            if tok not in expected:
                raise UnexpectedTokenException(self._currentIndex, tok, expected)
        else:
            if tok != expected:
                raise UnexpectedTokenException(self._currentIndex, tok, expected)

    def assertToken(self, tok, expected):
        if isinstance(expected, list):
            if tok not in expected:
                raise UnexpectedTokenException(self._currentIndex, tok, expected)
        else:
            if tok != expected:
                raise UnexpectedTokenException(self._currentIndex, tok, expected)

    def __repr__(self):
        if self.inRange(0):
            msg = 'Next token: ' + self.token(0)
        else:
            msg = 'No more tokens'
        return '<' + self.__class__.__name__ + ': ' + msg + '>'


def read_logic(s, logic_parser=None, encoding=None):
    """
    Convert a file of First Order Formulas into a list of {Expression}s.

    :param s: the contents of the file
    :type s: str
    :param logic_parser: The parser to be used to parse the logical expression
    :type logic_parser: LogicParser
    :param encoding: the encoding of the input string, if it is binary
    :type encoding: str
    :return: a list of parsed formulas.
    :rtype: list(Expression)
    """
    if encoding is not None:
        s = s.decode(encoding)
    if logic_parser is None:
        logic_parser = LogicParser()

    statements = []
    for linenum, line in enumerate(s.splitlines()):
        line = line.strip()
        if line.startswith('#') or line=='': continue
        try:
            statements.append(logic_parser.parse(line))
        except LogicalExpressionException:
            raise ValueError('Unable to parse line %s: %s' % (linenum, line))
    return statements


@functools.total_ordering
class Variable(object):
  def __init__(self, name, type=None):
    """
    :param name: the name of the variable
    """
    assert isinstance(name, string_types), "%s is not a string" % name
    self.name = name
    self.type = type

  def __eq__(self, other):
    return isinstance(other, Variable) and self.name == other.name

  def __ne__(self, other):
    return not self == other

  def __lt__(self, other):
    if not isinstance(other, Variable):
        raise TypeError
    return self.name < other.name

  def substitute_bindings(self, bindings):
    return bindings.get(self, self)

  def __hash__(self):
    return hash((self.name, self.type))

  def __str__(self):
    return self.name

  def __repr__(self):
    return "Variable('%s')" % self.name


def unique_variable(pattern=None, ignore=None, type=None):
    """
    Return a new, unique variable.

    :param pattern: ``Variable`` that is being replaced.  The new variable must
        be the same type.
    :param term: a set of ``Variable`` objects that should not be returned from
        this function.
    :rtype: Variable
    """
    if pattern is not None:
        if is_indvar(pattern.name):
            prefix = 'z'
        elif is_funcvar(pattern.name):
            prefix = 'F'
        elif is_eventvar(pattern.name):
            prefix = 'e0'
        else:
            assert False, "Cannot generate a unique constant"
    else:
        prefix = 'z'

    v = Variable("%s%s" % (prefix, _counter.get()), type=type)
    while ignore is not None and v in ignore:
        v = Variable("%s%s" % (prefix, _counter.get()), type=type)
    return v

def skolem_function(univ_scope=None):
    """
    Return a skolem function over the variables in univ_scope
    param univ_scope
    """
    skolem = VariableExpression(Variable('F%s' % _counter.get()))
    if univ_scope:
        for v in list(univ_scope):
            skolem = skolem(VariableExpression(v))
    return skolem


class Type(object):
  def __hash__(self):
    return hash(str(self))

  @classmethod
  def fromstring(cls, s):
    return read_type(s)


class BasicType(Type):
  def __init__(self, name=None, parent=None):
    self.name = name
    self.parent = parent

  @property
  def flat(self):
    return (self,)

  @property
  def parents(self):
    parents = []
    node = self
    while isinstance(node, BasicType) and node.parent is not None:
      parents.append(node.parent)
      node = node.parent

    return parents

  def __eq__(self, other):
    return isinstance(other, BasicType) and str(self) == str(other)

  def __ne__(self, other):
    return not self == other

  def __repr__(self):
    return "<%s>" % self.name

  __hash__ = Type.__hash__

  def matches(self, other):
    ret = other == ANY_TYPE or self == other \
        or any(my_parent == other for my_parent in self.parents)
    return ret

  def resolve(self, other):
    if self.matches(other):
      return self
    return None


class ComplexType(Type):
  def __init__(self, first, second):
    assert(isinstance(first, Type)), "%s is not a Type" % first
    assert(isinstance(second, Type)), "%s is not a Type" % second
    self.first = first
    self.second = second

  @property
  def flat(self):
    expr = [self.first]
    node = self.second
    while node.__class__ == ComplexType:
      expr.append(node.first)
      node = node.second
    expr.append(node)
    return tuple(expr)

  @property
  def parents(self):
    return []

  def __eq__(self, other):
    return isinstance(other, ComplexType) and \
            self.first == other.first and \
            self.second == other.second

  def __ne__(self, other):
    return not self == other

  def __repr__(self):
    return "<%s>" % ",".join(map(repr, self.flat))

  __hash__ = Type.__hash__

  def matches(self, other):
    if isinstance(other, ComplexType):
      return self.first.matches(other.first) and \
              self.second.matches(other.second)
    else:
        return self == ANY_TYPE

  def resolve(self, other):
    if other == ANY_TYPE:
      return self
    elif isinstance(other, ComplexType):
      f = self.first.resolve(other.first)
      s = self.second.resolve(other.second)
      if f and s:
        return ComplexType(f,s)
      else:
        return None
    elif self == ANY_TYPE:
      return other
    else:
      return None

  def __str__(self):
    if self == ANY_TYPE:
      return "%s" % ANY_TYPE
    else:
      return '<%s,%s>' % (self.first, self.second)

  def str(self):
    if self == ANY_TYPE:
      return ANY_TYPE.str()
    else:
      return '(%s -> %s)' % (self.first.str(), self.second.str())


class EntityType(BasicType):
  def __str__(self):
    return "e"

  def str(self):
    return "IND"

class TruthValueType(BasicType):
    def __str__(self):
        return 't'

    def str(self):
        return 'BOOL'

class EventType(BasicType):
    def __str__(self):
        return 'v'

    def str(self):
        return 'EVENT'


class AnyType(BasicType, ComplexType):
  @property
  def first(self): return self
  @property
  def second(self): return self

  def __eq__(self, other):
    return isinstance(other, AnyType) or other.__eq__(self)

  def __ne__(self, other):
    return not self == other

  __hash__ = Type.__hash__

  def matches(self, other):
    return True

  def resolve(self, other):
    return other

  def __str__(self):
    return "?"

  def str(self):
    return "ANY"


ENTITY_TYPE = EntityType()
TRUTH_TYPE = TruthValueType()
EVENT_TYPE = EventType()
ANY_TYPE = AnyType()


def read_type(type_string):
  assert isinstance(type_string, string_types)
  type_string = type_string.replace(' ', '') #remove spaces

  if type_string[0] == '<':
    assert type_string[-1] == '>'
    paren_count = 0
    for i,char in enumerate(type_string):
      if char == '<':
        paren_count += 1
      elif char == '>':
        paren_count -= 1
        assert paren_count > 0
      elif char == ',':
        if paren_count == 1:
          break
    return ComplexType(read_type(type_string[1  :i ]),
                       read_type(type_string[i+1:-1]))
  elif type_string[0] == "%s" % ENTITY_TYPE:
    return ENTITY_TYPE
  # elif type_string[0] == "%s" % TRUTH_TYPE:
  #     return TRUTH_TYPE
  elif type_string[0] == "%s" % ANY_TYPE:
      return ANY_TYPE
  else:
      raise LogicalExpressionException("Unexpected character: '%s'." % type_string[0])


class SubstituteBindingsI(object):
    """
    An interface for classes that can perform substitutions for
    variables.
    """
    def substitute_bindings(self, bindings):
        """
        :return: The object that is obtained by replacing
            each variable bound by ``bindings`` with its values.
            Aliases are already resolved. (maybe?)
        :rtype: (any)
        """
        raise NotImplementedError()

    def variables(self):
        """
        :return: A list of all variables in this object.
        """
        raise NotImplementedError()


class Expression(SubstituteBindingsI):
    """This is the base abstract object for all logical expressions"""

    _logic_parser = LogicParser()
    _type_checking_logic_parser = LogicParser(type_check=True)

    @classmethod
    def fromstring(cls, s, type_check=False, signature=None):
        if type_check:
            return cls._type_checking_logic_parser.parse(s, signature)
        else:
            return cls._logic_parser.parse(s, signature)

    def __call__(self, other, *additional):
        accum = self.applyto(other)
        for a in additional:
            accum = accum(a)
        return accum

    def applyto(self, other):
        assert isinstance(other, Expression), "%s is not an Expression" % other
        return ApplicationExpression(self, other)

    def __neg__(self):
        return NegatedExpression(self)

    def negate(self):
        """If this is a negated expression, remove the negation.
        Otherwise add a negation."""
        return -self

    def __and__(self, other):
        if not isinstance(other, Expression):
            raise TypeError("%s is not an Expression" % other)
        return AndExpression(self, other)

    def __or__(self, other):
        if not isinstance(other, Expression):
            raise TypeError("%s is not an Expression" % other)
        return OrExpression(self, other)

    def __gt__(self, other):
        if not isinstance(other, Expression):
            raise TypeError("%s is not an Expression" % other)
        return ImpExpression(self, other)

    def __lt__(self, other):
        if not isinstance(other, Expression):
            raise TypeError("%s is not an Expression" % other)
        return IffExpression(self, other)

    def __eq__(self, other):
        raise NotImplementedError()

    def __ne__(self, other):
        return not self == other

    def equiv(self, other, prover=None):
        """
        Check for logical equivalence.
        Pass the expression (self <-> other) to the theorem prover.
        If the prover says it is valid, then the self and other are equal.

        :param other: an ``Expression`` to check equality against
        :param prover: a ``nltk.inference.api.Prover``
        """
        assert isinstance(other, Expression), "%s is not an Expression" % other

        if prover is None:
            from nltk.inference import Prover9
            prover = Prover9()
        bicond = IffExpression(self.simplify(), other.simplify())
        return prover.prove(bicond)

    def __hash__(self):
        return hash(repr(self))

    def substitute_bindings(self, bindings):
        expr = self
        for var in expr.variables():
            if var in bindings:
                val = bindings[var]
                if isinstance(val, Variable):
                    val = self.make_VariableExpression(val)
                elif not isinstance(val, Expression):
                    raise ValueError('Can not substitute a non-expression '
                                     'value into an expression: %r' % (val,))
                # Substitute bindings in the target value.
                val = val.substitute_bindings(bindings)
                # Replace var w/ the target value.
                expr = expr.replace(var, val)
        return expr.simplify()

    def typecheck(self, signature=None):
        """
        Infer and check types.  Raise exceptions if necessary.

        :param signature: dict that maps variable names to types (or string
            representations of types)
        :return: the signature, plus any additional type mappings
        """
        sig = defaultdict(list)
        if signature:
            for key in signature:
                val = signature[key]
                varEx = VariableExpression(Variable(key))

                if isinstance(val, Type):
                    destType = val
                else:
                    destType = read_type(val)

                varEx.variable.type = destType
                varEx.type = destType
                # print(key, val, type(varEx), varEx, varEx.type)
                sig[key].append(varEx)

        self._set_type(signature=sig)

        return dict((key, sig[key][0].type) for key in sig)

    def findtype(self, variable):
        """
        Find the type of the given variable as it is used in this expression.
        For example, finding the type of "P" in "P(x) & Q(x,y)" yields "<e,t>"

        :param variable: Variable
        """
        raise NotImplementedError()

    def _set_type(self, other_type=ANY_TYPE, signature=None):
        """
        Set the type of this expression to be the given type.  Raise type
        exceptions where applicable.

        :param other_type: Type
        :param signature: dict(str -> list(AbstractVariableExpression))
        """
        raise NotImplementedError()

    def replace(self, variable, expression, replace_bound=False, alpha_convert=True):
        """
        Replace every instance of 'variable' with 'expression'
        :param variable: ``Variable`` The variable to replace
        :param expression: ``Expression`` The expression with which to replace it
        :param replace_bound: bool Should bound variables be replaced?
        :param alpha_convert: bool Alpha convert automatically to avoid name clashes?
        """
        assert isinstance(variable, Variable), "%s is not a Variable" % variable
        assert isinstance(expression, Expression), "%s is not an Expression" % expression

        return self.visit_structured(lambda e: e.replace(variable, expression,
                                                         replace_bound, alpha_convert),
                                     self.__class__)

    def normalize(self, newvars=None):
        """Rename auto-generated unique variables"""
        def get_indiv_vars(e):
            if isinstance(e, IndividualVariableExpression):
                return set([e])
            elif isinstance(e, AbstractVariableExpression):
                return set()
            else:
                return e.visit(get_indiv_vars,
                               lambda parts: functools.reduce(operator.or_, parts, set()))

        result = self
        for i,e in enumerate(sorted(get_indiv_vars(self), key=lambda e: e.variable)):
            if isinstance(e,EventVariableExpression):
                newVar = e.__class__(Variable('e0%s' % (i+1)))
            elif isinstance(e,IndividualVariableExpression):
                newVar = e.__class__(Variable('z%s' % (i+1)))
            else:
                newVar = e
            result = result.replace(e.variable, newVar, True)
        return result

    def visit(self, function, combinator):
        """
        Recursively visit subexpressions.  Apply 'function' to each
        subexpression and pass the result of each function application
        to the 'combinator' for aggregation:

            return combinator(map(function, self.subexpressions))

        Bound variables are neither applied upon by the function nor given to
        the combinator.
        :param function: ``Function<Expression,T>`` to call on each subexpression
        :param combinator: ``Function<list<T>,R>`` to combine the results of the
        function calls
        :return: result of combination ``R``
        """
        raise NotImplementedError()

    def visit_structured(self, function, combinator):
        """
        Recursively visit subexpressions.  Apply 'function' to each
        subexpression and pass the result of each function application
        to the 'combinator' for aggregation.  The combinator must have
        the same signature as the constructor.  The function is not
        applied to bound variables, but they are passed to the
        combinator.
        :param function: ``Function`` to call on each subexpression
        :param combinator: ``Function`` with the same signature as the
        constructor, to combine the results of the function calls
        :return: result of combination
        """
        return self.visit(function, lambda parts: combinator(*parts))

    def __repr__(self):
        return '<%s %s>' % (self.__class__.__name__, self)

    def __str__(self):
        return self.str()

    SPECIAL_VAR_RE = re.compile('^[?@]')
    def variables(self):
        """
        Return a set of all the variables for binding substitution.
        The variables returned include all free (non-bound) individual
        variables and any variable starting with '?' or '@'.
        :return: set of ``Variable`` objects
        """
        return self.free() | set(p for p in self.predicates()|self.constants()
                                 if self.SPECIAL_VAR_RE.match(p.name))

    def free(self):
        """
        Return a set of all the free (non-bound) variables.  This includes
        both individual and predicate variables, but not constants.
        :return: set of ``Variable`` objects
        """
        return self.visit(lambda e: e.free(),
                          lambda parts: functools.reduce(operator.or_, parts, set()))

    def bound(self):
        """
        Return a set of all the bound variables.
        """
        return self.visit(lambda e: e.bound(),
                          lambda parts: functools.reduce(operator.concat, parts, []))

    def constants(self):
        """
        Return a set of individual constants (non-predicates).
        :return: set of ``Variable`` objects
        """
        return self.visit(lambda e: e.constants(),
                          lambda parts: functools.reduce(operator.or_, parts, set()))

    def predicates(self):
        """
        Return a set of predicates (constants, not variables).
        :return: set of ``Variable`` objects
        """
        return self.visit(lambda e: e.predicates(),
                          lambda parts: functools.reduce(operator.or_, parts, set()))

    def simplify(self):
        """
        :return: beta-converted version of this expression
        """
        return self.visit_structured(lambda e: e.simplify(), self.__class__)

    def make_VariableExpression(self, variable):
        return VariableExpression(variable)


class ApplicationExpression(Expression):
    r"""
    This class is used to represent two related types of logical expressions.

    The first is a Predicate Expression, such as "P(x,y)".  A predicate
    expression is comprised of a ``FunctionVariableExpression`` or
    ``ConstantExpression`` as the predicate and a list of Expressions as the
    arguments.

    The second is a an application of one expression to another, such as
    "(\x.dog(x))(fido)".

    The reason Predicate Expressions are treated as Application Expressions is
    that the Variable Expression predicate of the expression may be replaced
    with another Expression, such as a LambdaExpression, which would mean that
    the Predicate should be thought of as being applied to the arguments.

    The logical expression reader will always curry arguments in a application expression.
    So, "\x y.see(x,y)(john,mary)" will be represented internally as
    "((\x y.(see(x))(y))(john))(mary)".  This simplifies the internals since
    there will always be exactly one argument in an application.

    The str() method will usually print the curried forms of application
    expressions.  The one exception is when the the application expression is
    really a predicate expression (ie, underlying function is an
    ``AbstractVariableExpression``).  This means that the example from above
    will be returned as "(\x y.see(x,y)(john))(mary)".
    """
    def __init__(self, function, argument):
        """
        :param function: ``Expression``, for the function expression
        :param argument: ``Expression``, for the argument
        """
        assert isinstance(function, Expression), "%s is not an Expression" % function
        assert isinstance(argument, Expression), "%s is not an Expression" % argument
        self.function = function
        self.argument = argument

    def simplify(self):
        function = self.function.simplify()
        argument = self.argument.simplify()
        if isinstance(function, LambdaExpression):
            return function.term.replace(function.variable, argument).simplify()
        else:
            return self.__class__(function, argument)

    @property
    def type(self):
        if isinstance(self.function.type, ComplexType):
            return self.function.type.second
        else:
            return ANY_TYPE

    def _set_type(self, other_type=ANY_TYPE, signature=None):
        """:see Expression._set_type()"""
        assert isinstance(other_type, Type)

        if signature is None:
            signature = defaultdict(list)

        self.argument._set_type(ANY_TYPE, signature)
        try:
            # print("====", self.function, self.argument, self.argument.type)
            self.function._set_type(ComplexType(self.argument.type, other_type), signature)
        except TypeResolutionException:
            raise TypeException(
                    "The function '%s' is of type '%s' and cannot be applied "
                    "to '%s' of type '%s'.  Its argument must match type '%s'."
                    % (self.function, self.function.type, self.argument,
                       self.argument.type, self.function.type.first))

    def findtype(self, variable):
        """:see Expression.findtype()"""
        assert isinstance(variable, Variable), "%s is not a Variable" % variable
        if self.is_atom():
            function, args = self.uncurry()
        else:
            #It's not a predicate expression ("P(x,y)"), so leave args curried
            function = self.function
            args = [self.argument]

        found = [arg.findtype(variable) for arg in [function]+args]

        unique = []
        for f in found:
            if f != ANY_TYPE:
                if unique:
                    for u in unique:
                        if f.matches(u):
                            break
                else:
                    unique.append(f)

        if len(unique) == 1:
            return list(unique)[0]
        else:
            return ANY_TYPE

    def constants(self):
        """:see: Expression.constants()"""
        if isinstance(self.function, AbstractVariableExpression):
            function_constants = set()
        else:
            function_constants = self.function.constants()
        return function_constants | self.argument.constants()

    def predicates(self):
        """:see: Expression.predicates()"""
        if isinstance(self.function, ConstantExpression):
            function_preds = set([self.function.variable])
        else:
            function_preds = self.function.predicates()
        return function_preds | self.argument.predicates()

    def visit(self, function, combinator):
        """:see: Expression.visit()"""
        return combinator([function(self.function), function(self.argument)])

    def __eq__(self, other):
        return isinstance(other, ApplicationExpression) and \
                self.function == other.function and \
                self.argument == other.argument

    def __ne__(self, other):
        return not self == other

    __hash__ = Expression.__hash__

    def __str__(self):
        # uncurry the arguments and find the base function
        if self.is_atom():
            function, args = self.uncurry()
            arg_str = ','.join("%s" % arg for arg in args)
        else:
            #Leave arguments curried
            function = self.function
            arg_str = "%s" % self.argument

        function_str = "%s" % function
        parenthesize_function = False
        if isinstance(function, LambdaExpression):
            if isinstance(function.term, ApplicationExpression):
                if not isinstance(function.term.function,
                                  AbstractVariableExpression):
                    parenthesize_function = True
            elif not isinstance(function.term, BooleanExpression):
                parenthesize_function = True
        elif isinstance(function, ApplicationExpression):
            parenthesize_function = True

        if parenthesize_function:
            function_str = Tokens.OPEN + function_str + Tokens.CLOSE

        return function_str + Tokens.OPEN + arg_str + Tokens.CLOSE

    def uncurry(self):
        """
        Uncurry this application expression

        return: A tuple (base-function, arg-list)
        """
        function = self.function
        args = [self.argument]
        while isinstance(function, ApplicationExpression):
            #(\x.\y.sees(x,y)(john))(mary)
            args.insert(0, function.argument)
            function = function.function
        return (function, args)

    @property
    def pred(self):
        """
        Return uncurried base-function.
        If this is an atom, then the result will be a variable expression.
        Otherwise, it will be a lambda expression.
        """
        return self.uncurry()[0]

    @property
    def args(self):
        """
        Return uncurried arg-list
        """
        return self.uncurry()[1]

    def is_atom(self):
        """
        Is this expression an atom (as opposed to a lambda expression applied
        to a term)?
        """
        return isinstance(self.pred, AbstractVariableExpression)


@functools.total_ordering
class AbstractVariableExpression(Expression):
    """This class represents a variable to be used as a predicate or entity"""
    def __init__(self, variable):
        """
        :param variable: ``Variable``, for the variable
        """
        assert isinstance(variable, Variable), "%s is not a Variable" % variable
        self.variable = variable

    def simplify(self):
        return self

    def replace(self, variable, expression, replace_bound=False, alpha_convert=True):
        """:see: Expression.replace()"""
        assert isinstance(variable, Variable), "%s is not an Variable" % variable
        assert isinstance(expression, Expression), "%s is not an Expression" % expression
        if self.variable == variable:
            return expression
        else:
            return self

    def _set_type(self, other_type=ANY_TYPE, signature=None):
        """:see Expression._set_type()"""
        assert isinstance(other_type, Type)

        if signature is None:
            signature = defaultdict(list)

        resolution = other_type
        # print("in var", self, resolution)
        for varEx in signature[self.variable.name]:
            resolution = varEx.type.resolve(resolution)
            if not resolution:
                raise InconsistentTypeHierarchyException(self)

        signature[self.variable.name].append(self)
        for varEx in signature[self.variable.name]:
            varEx.type = resolution

    def findtype(self, variable):
        """:see Expression.findtype()"""
        assert isinstance(variable, Variable), "%s is not a Variable" % variable
        if self.variable == variable:
            return self.type
        else:
            return ANY_TYPE

    def predicates(self):
        """:see: Expression.predicates()"""
        return set()

    def bound(self):
        return []

    def __eq__(self, other):
        """Allow equality between instances of ``AbstractVariableExpression``
        subtypes."""
        return isinstance(other, AbstractVariableExpression) and \
               self.variable == other.variable

    def __ne__(self, other):
        return not self == other

    def __lt__(self, other):
        if not isinstance(other, AbstractVariableExpression):
            raise TypeError
        return self.variable < other.variable

    __hash__ = Expression.__hash__

    def __str__(self):
        return "%s" % self.variable

class IndividualVariableExpression(AbstractVariableExpression):
    """This class represents variables that take the form of a single lowercase
    character (other than 'e') followed by zero or more digits."""
    def _set_type(self, other_type=ANY_TYPE, signature=None):
        """:see Expression._set_type()"""
        assert isinstance(other_type, Type)

        if signature is None:
            signature = defaultdict(list)

        if self.variable.type is None or self.variable.type == ANY_TYPE:
            # Missing a variable type! Look elsewhere in the expression for
            # other uses of the same variable.
            try:
                self.variable.type = signature[self.variable.name][0].variable.type
            except IndexError:
                if self.variable.type is None:
                    # We're genuinely missing a type. Uh oh.
                    raise RuntimeError("Missing declared variable type for %s" % self.variable)

        if not other_type.matches(self.variable.type):
            raise TypeResolutionException(self.variable.type, other_type)
        # if not other_type.matches(ENTITY_TYPE):
        #     raise IllegalTypeException(self, other_type, ENTITY_TYPE)

        signature[self.variable.name].append(self)

    def _get_type(self): return self.variable.type
    type = property(_get_type, _set_type)

    def free(self):
        """:see: Expression.free()"""
        return set([self.variable])

    def constants(self):
        """:see: Expression.constants()"""
        return set()

class FunctionVariableExpression(AbstractVariableExpression):
    """This class represents variables that take the form of a single uppercase
    character followed by zero or more digits."""
    type = ANY_TYPE

    def free(self):
        """:see: Expression.free()"""
        return set([self.variable])

    def constants(self):
        """:see: Expression.constants()"""
        return set()

class EventVariableExpression(IndividualVariableExpression):
    """This class represents variables that take the form of a single lowercase
    'e' character followed by zero or more digits."""
    type = EVENT_TYPE

class ConstantExpression(AbstractVariableExpression):
    """This class represents variables that do not take the form of a single
    character followed by zero or more digits."""
    type = ENTITY_TYPE

    def _set_type(self, other_type=ANY_TYPE, signature=None):
        """:see Expression._set_type()"""
        assert isinstance(other_type, Type)

        if signature is None:
            signature = defaultdict(list)

        resolution = ANY_TYPE
        if other_type != ANY_TYPE:
            resolution = other_type
            if self.type != ENTITY_TYPE:
                resolution = resolution.resolve(self.type)

        for varEx in signature[self.variable.name]:
            # print("\t", varEx, varEx.type, resolution)
            resolution = varEx.type.resolve(resolution)
            # print("\t\t", resolution)
            if not resolution:
                raise InconsistentTypeHierarchyException(self)

        signature[self.variable.name].append(self)
        for varEx in signature[self.variable.name]:
            varEx.type = resolution

    def free(self):
        """:see: Expression.free()"""
        return set()

    def bound(self):
        return []

    def constants(self):
        """:see: Expression.constants()"""
        return set([self.variable])


def VariableExpression(variable):
    """
    This is a factory method that instantiates and returns a subtype of
    ``AbstractVariableExpression`` appropriate for the given variable.
    """
    assert isinstance(variable, Variable), "%s is not a Variable" % variable
    if is_indvar(variable.name):
        return IndividualVariableExpression(variable)
    elif is_funcvar(variable.name):
        return FunctionVariableExpression(variable)
    elif is_eventvar(variable.name):
        return EventVariableExpression(variable)
    else:
        return ConstantExpression(variable)


class VariableBinderExpression(Expression):
    """This an abstract class for any Expression that binds a variable in an
    Expression.  This includes LambdaExpressions and Quantified Expressions"""
    def __init__(self, variable, term):
        """
        :param variable: ``Variable``, for the variable
        :param term: ``Expression``, for the term
        """
        assert isinstance(variable, Variable), "%s is not a Variable" % variable
        assert isinstance(term, Expression), "%s is not an Expression" % term
        self.variable = variable
        self.term = term

    def replace(self, variable, expression, replace_bound=False, alpha_convert=True):
        """:see: Expression.replace()"""
        assert isinstance(variable, Variable), "%s is not a Variable" % variable
        assert isinstance(expression, Expression), "%s is not an Expression" % expression
        #if the bound variable is the thing being replaced
        if self.variable == variable:
            if replace_bound:
                assert isinstance(expression, AbstractVariableExpression),\
                       "%s is not a AbstractVariableExpression" % expression
                return self.__class__(expression.variable,
                                      self.term.replace(variable, expression, True, alpha_convert))
            else:
                return self
        else:
            # if the bound variable appears in the expression, then it must
            # be alpha converted to avoid a conflict
            if alpha_convert and self.variable in expression.free():
                self = self.alpha_convert(unique_variable(pattern=self.variable))

            #replace in the term
            return self.__class__(self.variable,
                                  self.term.replace(variable, expression, replace_bound, alpha_convert))

    def alpha_convert(self, newvar):
        """Rename all occurrences of the variable introduced by this variable
        binder in the expression to ``newvar``.
        :param newvar: ``Variable``, for the new variable
        """
        assert isinstance(newvar, Variable), "%s is not a Variable" % newvar
        return self.__class__(newvar,
                              self.term.replace(self.variable,
                                                VariableExpression(newvar),
                                                True))

    def free(self):
        """:see: Expression.free()"""
        return self.term.free() - set([self.variable])

    def bound(self):
        return self.term.bound() + [self.variable]

    def findtype(self, variable):
        """:see Expression.findtype()"""
        assert isinstance(variable, Variable), "%s is not a Variable" % variable
        if variable == self.variable:
            return ANY_TYPE
        else:
            return self.term.findtype(variable)

    def visit(self, function, combinator):
        """:see: Expression.visit()"""
        return combinator([function(self.term)])

    def visit_structured(self, function, combinator):
        """:see: Expression.visit_structured()"""
        return combinator(self.variable, function(self.term))

    def __eq__(self, other):
        r"""Defines equality modulo alphabetic variance.  If we are comparing
        \x.M  and \y.N, then check equality of M and N[x/y]."""
        if isinstance(self, other.__class__) or \
           isinstance(other, self.__class__):
            if self.variable == other.variable:
                return self.term == other.term
            else:
                # Comparing \x.M  and \y.N.  Relabel y in N with x and continue.
                varex = VariableExpression(self.variable)
                return self.term == other.term.replace(other.variable, varex)
        else:
            return False

    def __ne__(self, other):
        return not self == other

    __hash__ = Expression.__hash__


class LambdaExpression(VariableBinderExpression):
    @property
    def type(self):
        return ComplexType(self.variable.type or ANY_TYPE,
                           self.term.type or ANY_TYPE)

    def _set_type(self, other_type=ANY_TYPE, signature=None):
        """:see Expression._set_type()"""
        assert isinstance(other_type, Type)

        if signature is None:
            signature = defaultdict(list)

        if self.variable.type is None:
            # Missing a variable type! Look elsewhere in the expression for
            # other uses of the same variable.
            try:
                self.variable.type = signature[self.variable.name][0].variable.type
            except IndexError:
                self.variable.type = other_type.first

            if self.variable.name not in signature or not signature[self.variable.name]:
              signature[self.variable.name] = [self]

        self.term._set_type(other_type.second, signature)
        if not self.type.resolve(other_type):
            raise TypeResolutionException(self, other_type)

    def __str__(self):
        variables = [self.variable]
        term = self.term
        while term.__class__ == self.__class__:
            variables.append(term.variable)
            term = term.term
        return Tokens.LAMBDA + ' '.join("%s" % v for v in variables) + \
               Tokens.DOT + "%s" % term


class QuantifiedExpression(VariableBinderExpression):
    @property
    def type(self): return TRUTH_TYPE

    def _set_type(self, other_type=ANY_TYPE, signature=None):
        """:see Expression._set_type()"""
        assert isinstance(other_type, Type)

        if signature is None:
            signature = defaultdict(list)

        if not other_type.matches(TRUTH_TYPE):
            raise IllegalTypeException(self, other_type, TRUTH_TYPE)
        self.term._set_type(TRUTH_TYPE, signature)

    def __str__(self):
        variables = [self.variable]
        term = self.term
        while term.__class__ == self.__class__:
            variables.append(term.variable)
            term = term.term
        return self.getQuantifier() + ' ' + ' '.join("%s" % v for v in variables) + \
               Tokens.DOT + "%s" % term

class ExistsExpression(QuantifiedExpression):
    def getQuantifier(self):
        return Tokens.EXISTS

class AllExpression(QuantifiedExpression):
    def getQuantifier(self):
        return Tokens.ALL


class NegatedExpression(Expression):
    def __init__(self, term):
        assert isinstance(term, Expression), "%s is not an Expression" % term
        self.term = term

    @property
    def type(self): return TRUTH_TYPE

    def _set_type(self, other_type=ANY_TYPE, signature=None):
        """:see Expression._set_type()"""
        assert isinstance(other_type, Type)

        if signature is None:
            signature = defaultdict(list)

        if not other_type.matches(TRUTH_TYPE):
            raise IllegalTypeException(self, other_type, TRUTH_TYPE)
        self.term._set_type(TRUTH_TYPE, signature)

    def findtype(self, variable):
        assert isinstance(variable, Variable), "%s is not a Variable" % variable
        return self.term.findtype(variable)

    def visit(self, function, combinator):
        """:see: Expression.visit()"""
        return combinator([function(self.term)])

    def negate(self):
        """:see: Expression.negate()"""
        return self.term

    def __eq__(self, other):
        return isinstance(other, NegatedExpression) and self.term == other.term

    def __ne__(self, other):
        return not self == other

    __hash__ = Expression.__hash__

    def __str__(self):
        return Tokens.NOT + "%s" % self.term


class BinaryExpression(Expression):
    def __init__(self, first, second):
        assert isinstance(first, Expression), "%s is not an Expression" % first
        assert isinstance(second, Expression), "%s is not an Expression" % second
        self.first = first
        self.second = second

    @property
    def type(self): return TRUTH_TYPE

    def findtype(self, variable):
        """:see Expression.findtype()"""
        assert isinstance(variable, Variable), "%s is not a Variable" % variable
        f = self.first.findtype(variable)
        s = self.second.findtype(variable)
        if f == s or s == ANY_TYPE:
            return f
        elif f == ANY_TYPE:
            return s
        else:
            return ANY_TYPE

    def visit(self, function, combinator):
        """:see: Expression.visit()"""
        return combinator([function(self.first), function(self.second)])

    def __eq__(self, other):
        return (isinstance(self, other.__class__) or \
                isinstance(other, self.__class__)) and \
               self.first == other.first and self.second == other.second

    def __ne__(self, other):
        return not self == other

    __hash__ = Expression.__hash__

    def __str__(self):
        first = self._str_subex(self.first)
        second = self._str_subex(self.second)
        return Tokens.OPEN + first + ' ' + self.getOp() \
                + ' ' + second + Tokens.CLOSE

    def _str_subex(self, subex):
        return "%s" % subex


class BooleanExpression(BinaryExpression):
    def _set_type(self, other_type=ANY_TYPE, signature=None):
        """:see Expression._set_type()"""
        assert isinstance(other_type, Type)

        if signature is None:
            signature = defaultdict(list)

        if not other_type.matches(TRUTH_TYPE):
            raise IllegalTypeException(self, other_type, TRUTH_TYPE)
        self.first._set_type(TRUTH_TYPE, signature)
        self.second._set_type(TRUTH_TYPE, signature)

class AndExpression(BooleanExpression):
    """This class represents conjunctions"""
    def getOp(self):
        return Tokens.AND

    def _str_subex(self, subex):
        s = "%s" % subex
        if isinstance(subex, AndExpression):
            return s[1:-1]
        return s

class OrExpression(BooleanExpression):
    """This class represents disjunctions"""
    def getOp(self):
        return Tokens.OR

    def _str_subex(self, subex):
        s = "%s" % subex
        if isinstance(subex, OrExpression):
            return s[1:-1]
        return s

class ImpExpression(BooleanExpression):
    """This class represents implications"""
    def getOp(self):
        return Tokens.IMP

class IffExpression(BooleanExpression):
    """This class represents biconditionals"""
    def getOp(self):
        return Tokens.IFF


class EqualityExpression(BinaryExpression):
    """This class represents equality expressions like "(x = y)"."""
    def _set_type(self, other_type=ANY_TYPE, signature=None):
        """:see Expression._set_type()"""
        assert isinstance(other_type, Type)

        if signature is None:
            signature = defaultdict(list)

        if not other_type.matches(TRUTH_TYPE):
            raise IllegalTypeException(self, other_type, TRUTH_TYPE)
        self.first._set_type(ENTITY_TYPE, signature)
        self.second._set_type(ENTITY_TYPE, signature)

    def getOp(self):
        return Tokens.EQ


class LogicalExpressionException(Exception):
    def __init__(self, index, message):
        self.index = index
        Exception.__init__(self, message)

class UnexpectedTokenException(LogicalExpressionException):
    def __init__(self, index, unexpected=None, expected=None, message=None):
        if unexpected and expected:
            msg = "Unexpected token: '%s'.  " \
                  "Expected token '%s'." % (unexpected, expected)
        elif unexpected:
            msg = "Unexpected token: '%s'." % unexpected
            if message:
                msg += '  '+message
        else:
            msg = "Expected token '%s'." % expected
        LogicalExpressionException.__init__(self, index, msg)

class ExpectedMoreTokensException(LogicalExpressionException):
    def __init__(self, index, message=None):
        if not message:
            message = 'More tokens expected.'
        LogicalExpressionException.__init__(self, index, 'End of input found.  ' + message)


INDVAR_RE = re.compile(r'^[a-df-z]\d*$')
FUNCVAR_RE = re.compile(r'^[A-Z]\d*$')
EVENTVAR_RE = re.compile(r'^e\d*$')
def is_indvar(expr):
    """
    An individual variable must be a single lowercase character other than 'e',
    followed by zero or more digits.

    :param expr: str
    :return: bool True if expr is of the correct form
    """
    assert isinstance(expr, string_types), "%s is not a string" % expr
    return INDVAR_RE.match(expr) is not None

def is_funcvar(expr):
    """
    A function variable must be a single uppercase character followed by
    zero or more digits.

    :param expr: str
    :return: bool True if expr is of the correct form
    """
    assert isinstance(expr, string_types), "%s is not a string" % expr
    return FUNCVAR_RE.match(expr) is not None

def is_eventvar(expr):
    """
    An event variable must be a single lowercase 'e' character followed by
    zero or more digits.

    :param expr: str
    :return: bool True if expr is of the correct form
    """
    assert isinstance(expr, string_types), "%s is not a string" % expr
    return EVENTVAR_RE.match(expr) is not None


class TypeSystem(object):

  ANY_TYPE = ANY_TYPE
  ENTITY_TYPE = ENTITY_TYPE
  EVENT_TYPE = EVENT_TYPE

  def __init__(self, primitive_types):
    assert "?" not in primitive_types, "Cannot override ANY_TYPE name"
    assert "v" not in primitive_types, "Cannot override EVENT_TYPE name"
    self._types = {}

    for primitive_type in primitive_types:
        if isinstance(primitive_type, str):
            self._types[primitive_type] = BasicType(name=primitive_type,
                                                    parent=self.ENTITY_TYPE)
        else:
            assert isinstance(primitive_type, BasicType)
            self._types[primitive_type.name] = primitive_type

    self._types["?"] = self.ANY_TYPE
    self._types["v"] = self.EVENT_TYPE
    self._types["e"] = self.ENTITY_TYPE

  def __getitem__(self, type_expr):
    if isinstance(type_expr, Type):
      return type_expr
    if isinstance(type_expr, str):
      return self._types[type_expr]
    if isinstance(type_expr, (tuple, list)):
      if len(type_expr) == 1:
        return self[type_expr[0]]
      elif len(type_expr) > 1:
        return self.make_function_type(type_expr)
      else:
        raise ValueError("Invalid empty type expr %s" % (type_expr,))
    raise ValueError("Invalid type expr %s" % (type_expr,))

  def __contains__(self, type_expr):
    if isinstance(type_expr, str):
      return type_expr in self._types
    try:
      return self[type_expr] and True
    except:
      return False

  def __iter__(self):
    return iter(self._types.values())

  def make_function_type(self, type_expr):
    ret = self[type_expr[-1]]
    for type_expr_i in type_expr[:-1][::-1]:
      ret = ComplexType(self[type_expr_i], ret)
    return ret

  def new_function(self, name, type, defn, **kwargs):
    type = self[type]
    return Function(name, type, defn, **kwargs)

  def new_constant(self, name, type, **kwargs):
    type = self[type]
    return Variable(name, type)


class ConstantSystem(object):
  def __init__(self, constants):
    self.constants = constants
    self.constants_dict = {const.name: const for const in self.constants}
    self._used = {const.name: False for const in self.constants}
    self._iter_new_constants_buffer = None

  def mark_used(self, constant):
    assert constant.name in self._used, 'unregistered constant: {}'.format(constant)
    self._used[constant.name] = True

  def mark_used_expressions(self, expressions):
    for expr in expressions:
      for c in expr.constants():
        self.mark_used(c)

  def override_used_expressions(self, expressions):
    # clean the "used constants" buffer.
    self._used = {const.name: False for const in self.constants}
    self.mark_used_expressions(expressions)

  def make_new_constant(self, type_request=None, newly_used_constants_expr=None):
    unused = [
      name for name, v in self._used.items() if (
        (not v) and
        (newly_used_constants_expr is None or name not in newly_used_constants_expr) and
        (type_request is None or self.constants_dict[name].type.matches(type_request))
      )
    ]
    if len(unused) == 0:
      raise ValueError('cannot find unused constants of type: {}'.format(str(type_request)))
    return self.constants_dict[unused[0]]

  def iter_new_constants(self, type_request=None, newly_used_constants_expr=None):
    if newly_used_constants_expr is not None:
      for name in newly_used_constants_expr:
        if type_request is None or self.constants_dict[name].type.matches(type_request):
          yield self.constants_dict[name]
    yield self.make_new_constant(type_request=type_request, newly_used_constants_expr=newly_used_constants_expr)


# Wrapper for a typed function.
class Function(object):
  """
  Wrapper for a typed function.
  """

  def __init__(self, name, type, defn, weight=0.0):
    self.name = name
    self.type = type
    self.defn = defn
    self.weight = weight

  @property
  def arity(self):
    return len(self.type.flat) - 1

  @property
  def arg_types(self):
    return self.type.flat[:-1]

  @property
  def return_type(self):
    return self.type.flat[-1]

  def __hash__(self):
    return hash((self.name, self.type, self.defn))

  def __eq__(self, other):
    return hash(self) == hash(other)

  def __str__(self):
    return "function %s : %s" % (self.name, self.type)

  __repr__ = __str__


def make_application(pred, args):
  pred = ConstantExpression(Variable(pred)) if isinstance(pred, str) else pred
  expr = ApplicationExpression(pred, args[0])
  return functools.reduce(lambda x, y: ApplicationExpression(x, y),
                          args[1:], expr)


def next_bound_var(bound_vars, type):
  """
  Helper function: generate the next bound variable in a context where there
  are currently the bound variables `bound_vars` (assumed to be generated by
  this function).
  """
  name_length = 1 + len(bound_vars) // 26
  name_id = len(bound_vars) % 26
  name = chr(97 + name_id)
  return Variable(name * name_length, type)


def listify(fn=None, wrapper=list):
  """
  A decorator which wraps a function's return value in ``list(...)``.

  Useful when an algorithm can be expressed more cleanly as a generator but
  the function should return an list.
  """
  def listify_return(fn):
    @functools.wraps(fn)
    def listify_helper(*args, **kw):
      return wrapper(fn(*args, **kw))
    return listify_helper
  if fn is None:
    return listify_return
  return listify_return(fn)


def extract_lambda(expr):
  """
  Extract `LambdaExpression` arguments to the top of a semantic form.
  This makes them compatible with the CCG parsing setup, which needs top-level
  lambdas in order to perform function application during parsing.
  """
  variables = []

  def process_lambda(lambda_expr):
    # Create a new unique variable and substitute.
    unique = unique_variable()
    unique.type = lambda_expr.variable.type
    new_expr = lambda_expr.term.replace(lambda_expr.variable, IndividualVariableExpression(unique))
    return unique, new_expr

  # Traverse the LF and replace lambda expressions wherever necessary.
  def inner(node):
    if isinstance(node, ApplicationExpression):
      new_args = []

      for arg in node.args:
        if isinstance(arg, LambdaExpression):
          new_var, new_arg = process_lambda(arg)

          variables.append(new_var)
          new_args.append(new_arg)
        else:
          new_args.append(inner(arg))

      return make_application(node.pred.variable.name, new_args)
    else:
      return node

  expr = inner(expr)
  wrappings = []

  for variable_ordering in itertools.permutations(variables):
    wrapping = expr
    for variable in variable_ordering:
      wrapping = LambdaExpression(variable, wrapping)

    wrappings.append(wrapping.normalize())

  return wrappings


def get_arity(expr):
  """
  Get the arity of a lambda-extracted expression.
  """
  if isinstance(expr, LambdaExpression):
    return 1 + get_arity(expr.term)
  else:
    return 0


def read_ec_sexpr(sexpr):
  """
  Parse an EC-style S-expression into an untyped NLTK representation.
  """
  tokens = re.split(r"([()\s])", sexpr)

  bound_vars = set()
  bound_var_stack = []

  is_call = False
  stack = [(None, None, [])]
  for token in tokens:
    token = token.strip()
    if not token:
      continue

    if token == "(":
      if is_call:
        # Second consecutive left-paren -- this means we have a complex
        # function expression.
        stack.append((ApplicationExpression, None, []))
      is_call = True
    elif token == "lambda":
      is_call = False
      variable = next_bound_var(bound_vars, ANY_TYPE)
      bound_vars.add(variable)
      bound_var_stack.append(variable)

      stack.append((LambdaExpression, None, []))
    elif is_call:
      head = token
      if head.startswith("$"):
        bruijn_index = int(head[1:])
        # Bound variable is the head of an application expression.
        # First replace with a function-looking variable, then update parser
        # state.
        var_idx = -1 - bruijn_index
        var = bound_var_stack[var_idx]
        bound_vars.remove(var)

        new_var = Variable(var.name.upper())
        bound_var_stack[var_idx] = new_var
        bound_vars.add(new_var)

        head = FunctionVariableExpression(new_var)

      stack.append((ApplicationExpression, head, []))
      is_call = False
    elif token == ")":
      stack_top = stack.pop()
      if stack_top[0] == ApplicationExpression:
        _, pred, args = stack_top
        result = make_application(pred, args)
      elif stack_top[0] == LambdaExpression:
        _, _, term = stack_top
        variable = bound_var_stack.pop()
        result = LambdaExpression(variable, term[0])
      else:
        raise RuntimeError("unknown element on stack", stack_top)

      stack_parent = stack[-1]
      if stack_parent[0] == ApplicationExpression and stack_parent[1] is None:
        # We have just finished reading the head of an application expression.
        expr, _, args = stack_parent
        stack[-1] = (expr, result, args)
      else:
        # Add to children of parent node.
        stack_parent[2].append(result)
    elif token.startswith("$"):
      bruijn_index = int(token[1:])
      stack[-1][2].append(IndividualVariableExpression(bound_var_stack[-1 - bruijn_index]))
    else:
      stack[-1][2].append(ConstantExpression(Variable(token)))

  assert len(stack) == 1
  assert len(stack[0][2]) == 1
  return stack[0][2][0], bound_vars


class Ontology(object):
  """
  Defines an ontology for expressing and evaluating logical forms.
  """

  def __init__(self, types, functions, constants, variable_weight=0.1):
    """
    Arguments:
      types: TypeSystem
      functions: List of `k` `Function` instances
      constants: List of constants as (optionally typed) `Variable` instances
      variable_weight: log-probability of observing any variable
    """
    self.types = types

    self.functions = []
    self.functions_dict = {}
    self.variable_weight = variable_weight

    self.add_functions(functions)
    self.constant_system = ConstantSystem(constants)

    self._prepare()

  @property
  def constants(self):
    return self.constant_system.constants

  @property
  def constants_dict(self):
    return self.constant_system.constants_dict

  EXPR_TYPES = [ApplicationExpression, ConstantExpression,
                IndividualVariableExpression, LambdaExpression,
                FunctionVariableExpression]

  def add_functions(self, functions):
    self._clear_expression_cache()

    # Ignore functions which already exist.
    new_functions = []
    for function in functions:
      if function.name in self.functions_dict:
        existing_function = self.functions_dict[function.name]
        assert existing_function == function, \
            "Function name clash: existing %r, inserting %r" % (existing_function, function)
      else:
        new_functions.append(function)

    self.functions.extend(new_functions)
    self.functions_dict.update({fn.name: fn for fn in functions})

    for function in functions:
      # We can't statically verify the type of the definition, but we can at
      # least verify the arity.
      if function.defn is not None:
        L.debug("verifying arity: %s stated %i actual %i (%s)",
                function.name, function.arity, self.get_expr_arity(function.defn),
                function.defn)
        assert function.arity == self.get_expr_arity(function.defn), function.name

  def add_constants(self, constants):
    self._iter_expressions_inner.clear_cache()

    self.constants = constants
    self.constants_dict = {c.name: c for c in constants}

  def _prepare(self):
    self._nltk_type_signature = self._make_nltk_type_signature()

  def iter_expressions(self, max_depth=3, type_request=None, **kwargs):
    if type_request is not None and isinstance(type_request, (list, tuple)):
      type_request = self.types[type_request]
    ret = self._iter_expressions_inner(max_depth, bound_vars=(),
                                       type_request=type_request,
                                       **kwargs)
    ret = [x.normalize() for x in ret]

    return ret

  def _clear_expression_cache(self):
    self._iter_expressions_inner.cache_clear()

  @functools.lru_cache(maxsize=None)
  @listify
  def _iter_expressions_inner(self, max_depth, bound_vars,
                              type_request=None, function_weights=None,
                              use_unused_constants=False, newly_used_constants_expr=None):
    """
    Enumerate all legal expressions.

    Arguments:
      max_depth: Maximum tree depth to traverse.
      bound_vars: Bound variables (and their types) in the parent context. The
        returned expressions may reference these variables. List of `(name,
        type)` tuples.
      type_request: Optional requested type of the expression. This helps
        greatly restrict the space of enumerations when the type system is
        strong.
      function_weights: Override for function weights to determine the order in
        which we consider proposing function application expressions.
      use_unused_constants: If true, always use unused constants.
      newly_used_constants_expr: If not None, a set of constants (by name),
        all newly used constants for the current expression.
    """
    if max_depth == 0:
      return
    elif max_depth == 1 and not bound_vars:
      # require some bound variables to generate a valid lexical entry
      # semantics
      return

    newly_used_constants_expr = frozenset(newly_used_constants_expr or [])

    for expr_type in self.EXPR_TYPES:
      if expr_type == ApplicationExpression:
        # Loop over functions according to their weights.
        fn_weight_key = (lambda fn: function_weights[fn.name]) if function_weights is not None \
                        else (lambda fn: fn.weight)
        fns_sorted = sorted(self.functions_dict.values(), key=fn_weight_key,
                            reverse=True)

        if max_depth > 1:
          for fn in fns_sorted:
            # If there is a present type request, only consider functions with
            # the correct return type.
            # print("\t" * (6 - max_depth), fn.name, fn.return_type, " // request: ", type_request, bound_vars)
            if type_request is not None and not fn.return_type.matches(type_request):
              continue

            # Special case: yield fast event queries without recursion.
            if fn.arity == 1 and fn.arg_types[0] == self.types.EVENT_TYPE:
              yield make_application(fn.name, (ConstantExpression(Variable("e")),))
            elif fn.arity == 0:
              # 0-arity functions are represented in the logic as
              # `ConstantExpression`s.
              # print("\t" * (6 - max_depth + 1), "yielding const ", fn.name)
              yield ConstantExpression(Variable(fn.name))
            else:
              # print("\t" * (6 - max_depth), fn, fn.arg_types)
              sub_args = []

              all_arg_type_requests = list(fn.arg_types)

              def product_sub_args(i, ret, nuce):
                if i >= len(all_arg_type_requests):
                  yield ret
                  return

                arg_type_request = all_arg_type_requests[i]
                results = self._iter_expressions_inner(max_depth=max_depth - 1,
                                                       bound_vars=bound_vars,
                                                       type_request=arg_type_request,
                                                       function_weights=function_weights,
                                                       use_unused_constants=use_unused_constants,
                                                       newly_used_constants_expr=frozenset(nuce))
                for expr in results:
                  new_nuce = nuce | {c.name for c in expr.constants()}
                  yield from product_sub_args(i + 1, ret + (expr, ), new_nuce)

              for arg_combs in product_sub_args(0, tuple(), newly_used_constants_expr):
                candidate = make_application(fn.name, arg_combs)
                valid = self._valid_application_expr(candidate)
                # print("\t" * (6 - max_depth + 1), "valid %s? %s" % (candidate, valid))
                if valid:
                  yield candidate
      elif expr_type == LambdaExpression and max_depth > 1:
        if type_request is None or not isinstance(type_request, ComplexType):
          continue

        for num_args in range(1, len(type_request.flat)):
          for bound_var_types in itertools.product(self.observed_argument_types, repeat=num_args):
            # TODO typecheck with type request

            bound_vars = list(bound_vars)
            subexpr_bound_vars = []
            for new_type in bound_var_types:
              subexpr_bound_vars.append(next_bound_var(bound_vars + subexpr_bound_vars, new_type))
            all_bound_vars = tuple(bound_vars + subexpr_bound_vars)

            if type_request is not None:
              # TODO strong assumption -- assumes that lambda variables are used first
              subexpr_type_request_flat = type_request.flat[num_args:]
              subexpr_type_request = self.types[subexpr_type_request_flat]
            else:
              subexpr_type_request = None

            results = self._iter_expressions_inner(max_depth=max_depth - 1,
                                                   bound_vars=all_bound_vars,
                                                   type_request=subexpr_type_request,
                                                   function_weights=function_weights,
                                                   use_unused_constants=use_unused_constants,
                                                   newly_used_constants_expr=newly_used_constants_expr)

            for expr in results:
              candidate = expr
              for var in subexpr_bound_vars:
                candidate = LambdaExpression(var, candidate)
              valid = self._valid_lambda_expr(candidate, bound_vars)
              # print("\t" * (6 - max_depth), "valid lambda %s? %s" % (candidate, valid))
              if valid:
                # Assign variable types before returning.
                extra_types = {bound_var.name: bound_var.type
                               for bound_var in subexpr_bound_vars}

                try:
                  # TODO make sure variable names are unique before this happens
                  self.typecheck(candidate, extra_types)
                except InconsistentTypeHierarchyException:
                  pass
                else:
                  yield candidate
      elif expr_type == IndividualVariableExpression:
        for bound_var in bound_vars:
          if type_request and not bound_var.type.matches(type_request):
            continue

          # print("\t" * (6-max_depth), "var %s" % bound_var)

          yield IndividualVariableExpression(bound_var)
      elif expr_type == ConstantExpression:
        if use_unused_constants:
          try:
            for constant in self.constant_system.iter_new_constants(
                type_request=type_request,
                newly_used_constants_expr=newly_used_constants_expr
            ):

              yield ConstantExpression(constant)
          except ValueError:
            pass
        else:
          for constant in self.constants:
            if type_request is not None and not constant.type.matches(type_request):
              continue

            yield ConstantExpression(constant)
      elif expr_type == FunctionVariableExpression:
        # NB we don't support enumerating bound variables with function types
        # right now -- the following only considers yielding fixed functions
        # from the ontology.
        for function in self.functions:
          # Be a little strict here to avoid excessive enumeration -- only
          # consider emitting functions when the type request specifically
          # demands a function, not e.g. AnyType
          if type_request is None or type_request == self.types.ANY_TYPE \
              or not function.type.matches(type_request):
            continue

          yield FunctionVariableExpression(Variable(function.name, function.type))

  def typecheck(self, expr, extra_type_signature=None):
    type_signature = self._nltk_type_signature
    if extra_type_signature is not None:
      type_signature = copy(type_signature)
      type_signature.update(extra_type_signature)

    # First infer types of bound variables.
    # TODO assumes variable names are unique
    for variable in expr.bound():
      var_type = self.infer_type(expr, variable.name)
      variable.type = var_type
      type_signature[variable.name] = var_type

    expr.typecheck(signature=type_signature)

  def register_expressions(self, expressions):
    self._clear_expression_cache()
    self.constant_system.mark_used_expressions(expressions)

  def override_registered_expressions(self, expressions):
    self._clear_expression_cache()
    self.constant_system.override_used_expressions(expressions)

  def infer_type(self, expr, variable_name, extra_types=None):
    """
    Infer the type of a bound variable with name `variable_name` used in `expr`.

    Args:
      expr:
      variable_name:
      extra_types: Optional dictionary of provisional function types, mapping
        from function name to a type expression. Useful for doing type
        inference on elements of new functions before adding them to an
        ontology instance.
    """
    apparent_types = set()
    extra_types = extra_types or {}

    def visitor(node):
      if isinstance(node, ApplicationExpression):
        fn_name = node.pred.variable.name

        try:
          function_type = self.functions_dict[fn_name].type
        except KeyError:
          try:
            function_type = extra_types[fn_name]
          except KeyError:
            # No function information available. Ditch.
            return ANY_TYPE

        args = list(node.args)
        if len(args) != len(function_type.flat) - 1:
          raise InconsistentTypeHierarchyException("Function %s appears with the wrong arity" % fn_name)

        for i, arg in enumerate(node.args):
          visitor(arg)

          if isinstance(arg, IndividualVariableExpression) and arg.variable.name == variable_name:
            # We've found a use of the value as a function argument -- extract
            # the apparent type.
            apparent_types.add(function_type.flat[i])
          elif isinstance(arg, ApplicationExpression) and isinstance(arg.pred, FunctionVariableExpression) \
              and arg.pred.variable.name == variable_name:
            arg_types = ((ANY_TYPE,) * len(arg.args))
            apparent_types.add(arg_types + (function_type.flat[i],))
      elif isinstance(node, LambdaExpression):
        visitor(node.term)

    visitor(expr)
    if len(apparent_types) > 1:
      if len(apparent_types) == 2 and ANY_TYPE in apparent_types:
        # Good, just remove the AnyType.
        apparent_types.remove(ANY_TYPE)
      else:
        # TODO check type compatibility
        raise InconsistentTypeHierarchyException(variable_name, expr)
    elif len(apparent_types) == 0:
      return ANY_TYPE

    type_ret = next(iter(apparent_types))
    return self.types[type_ret]

  @property
  def observed_argument_types(self):
    """
    Collection of types (primitive or functional) which have been observed as
    arguments in this ontology's function.
    """
    obs_types = set(itertools.chain.from_iterable(
      fn.arg_types for fn in self.functions))
    return obs_types - set([self.types.ANY_TYPE])

  def get_expr_arity(self, expr):
    """
    Get the arity (number of bound variables) of a function definition.
    """
    if isinstance(expr, LambdaExpression):
      return 1 + self.get_expr_arity(expr.term)
    elif isinstance(expr, ApplicationExpression):
      function = self.functions_dict[expr.pred.variable.name]
      return function.arity - len(expr.args)
    elif isinstance(expr, (FunctionVariableExpression, ConstantExpression)) \
        and expr.variable.name in self.functions_dict:
      return self.functions_dict[expr.variable.name].arity
    elif isinstance(expr, ConstantExpression) \
        and expr.variable.name in self.constants_dict:
      return 0
    elif isinstance(expr, IndividualVariableExpression):
      return 0
    elif callable(expr):
      return len(inspect.signature(expr).parameters)
    else:
      raise ValueError("non-callable object: %r" % expr)

  def _valid_application_expr(self, application_expr):
    """
    Check whether this `ApplicationExpression` should be considered when
    enumerating programs.
    """
    # TODO check type consistency
    return True

  def _valid_lambda_expr(self, lambda_expr, ctx_bound_vars):
    """
    Check whether this `LambdaExpression` should be considered when enumerating
    programs.

    Arguments:
      lambda_expr: `LambdaExpression`
      ctx_bound_vars: Bound variables from the containing context
    """

    # Collect bound arguments and the body expression.
    bound_args = []
    expr = lambda_expr
    while isinstance(expr, LambdaExpression):
      bound_args.append(expr.variable)
      expr = expr.term
    body = expr

    if isinstance(body, IndividualVariableExpression):
      # Skip pointless identity function \z1.z1, etc.
      return False

    # Find bound variables in body.
    bound_variables = set(body.variables())
    # Remove ontology members.
    bound_variables = set(var for var in bound_variables
                          if var.name not in self.functions_dict)

    # Exclude exprs which do not use all of their bound arguments.
    available_vars = set(bound_args) | set(ctx_bound_vars)
    if available_vars != bound_variables:
      return False

    # Exclude unnecessarily curried expressions, e.g.
    # `\x.exists(x)` (vs. `exists`)
    if isinstance(body, ApplicationExpression) \
        and all(isinstance(a, IndividualVariableExpression) for a in body.args):
      arg_variables = [a.variable for a in body.args]
      if arg_variables == bound_args:
        return False

    # # Exclude exprs with simplistic bodies.
    # if isinstance(body, IndividualVariableExpression):
    #   return False

    return True

  def _make_nltk_type_expr(self, type_expr):
    if isinstance(type_expr, tuple) and len(type_expr) == 1:
      type_expr = type_expr[0]

    if type_expr in self.nltk_types:
      return self.nltk_types[type_expr]
    elif len(type_expr) > 1:
      return ComplexType(self._make_nltk_type_expr(type_expr[0]),
                         self._make_nltk_type_expr(type_expr[1:]))
    else:
      raise RuntimeError("unknown basic type %s" % (type_expr,))

  def _make_nltk_type_signature(self):
    signature = {fn.name: fn.type for fn in self.functions}
    signature.update({const.name: const.type for const in self.constants})
    return signature

  def as_ec_sexpr(self, expr):
    """
    Convert an `nltk.sem.logic` `Expression` to an S-expr string.
    """

    # Expressions which might contain a function reference
    func_exprs = (ConstantExpression, EventVariableExpression,
                  FunctionVariableExpression)

    def inner(expr, var_stack):
      if isinstance(expr, LambdaExpression):
        # Add lambda variable to var map.
        return "(lambda %s)" % inner(expr.term, var_stack + [expr.variable.name])
      elif isinstance(expr, ApplicationExpression):
        args = [inner(arg, var_stack) for arg in expr.args]
        return "(%s %s)" % (expr.pred.variable.name, " ".join(args))
      # elif isinstance(expr, AndExpression):
      #   return "(and %s %s)" % (inner(expr.first), inner(expr.second))
      elif isinstance(expr, func_exprs) and expr.variable.name in self.functions_dict:
        # EC requires S-expressions in normal form -- i.e. functions need to
        # appear in their applied form. We'll need a valid function type here
        # to get anything done.
        arity = self.functions_dict[expr.variable.name].arity

        if arity > 0:
          return ("(lambda " * arity) + \
              ("(%s %s)" % (expr.variable.name,
                            " ".join("$%i" % (idx - 1) for idx in range(arity, 0, -1)))) + \
              (")" * arity)
        else:
          return expr.variable.name
      elif isinstance(expr, IndividualVariableExpression):
        bruijn_index = len(var_stack) - var_stack.index(expr.variable.name) - 1
        return "$%i" % bruijn_index
      elif isinstance(expr, FunctionVariableExpression):
        raise ValueError("unknown function %s" % expr)
      elif isinstance(expr, ConstantExpression):
        return expr.variable.name
      else:
        raise ValueError("un-handled expression component %r" % expr)

    return inner(expr, [])

  def unwrap_function(self, function):
    """
    Given a function of this ontology, return an "unwrapped" `LambdaExpression`
    referencing that function. e.g.

        unwrap_function("sphere") => \\x.sphere(x)
    """
    fn = self.functions_dict[function]
    variables = [unique_variable(type=type) for type in fn.arg_types]

    if len(variables) > 0:
      # TODO make sure applicationexpression is properly typed
      core = make_application(function, [IndividualVariableExpression(v) for v in variables])
      ret = core
      for variable in variables[::-1]:
        ret = LambdaExpression(variable, ret)
      return ret.normalize()

    return ConstantExpression(Variable(function))

  def unwrap_base_functions(self, expr):
    """
    Given an Expression, unwrap all functions in base form.
    """
    # TODO extract the more general replacement logic here
    if isinstance(expr, (ConstantExpression, FunctionVariableExpression)) \
        and expr.variable.name in self.functions_dict:
      return self.unwrap_function(expr.variable.name)
    elif isinstance(expr, LambdaExpression):
      new = self.unwrap_base_functions(expr.term)
      if new != expr.term:
        expr = LambdaExpression(expr.variable, expr.term)
    elif isinstance(expr, ApplicationExpression):
      expr.argument = self.unwrap_base_functions(expr.argument)
      if isinstance(expr.function, ApplicationExpression):
        expr.function = self.unwrap_base_functions(expr.function)

    return expr


def compute_type_raised_semantics(semantics):
  core = deepcopy(semantics)
  parent = None
  while isinstance(core, LambdaExpression):
    parent = core
    core = core.term

  var = Variable("F")
  while var in core.free():
    var = unique_variable(pattern=var)
  core = ApplicationExpression(FunctionVariableExpression(var), core)

  if parent is not None:
    parent.term = core
  else:
    semantics = core

  return LambdaExpression(var, semantics)
