class Token(object):

  def __init__(self, token, categ, semantics=None, weight=0.001):
    self._token = token
    self._categ = categ
    self._weight = weight
    self._semantics = semantics

  def categ(self):
    return self._categ

  def weight(self):
    return self._weight

  def semantics(self):
    return self._semantics

  def clone(self):
    return Token(self._token, self._categ, self._semantics, self._weight)

  def __str__(self):
    return "Token(%s => %s%s)" % (self._token, self._categ,
                                  " {%s}" % self._semantics if self._semantics else "")

  __repr__ = __str__

  def __eq__(self, other):
    return isinstance(other, Token) and self.categ() == other.categ() \
        and self.weight() == other.weight() \
        and self.semantics() == other.semantics()

  def __hash__(self):
    return hash((self._token, self._categ, self._weight, self._semantics))


from pyccg import *
