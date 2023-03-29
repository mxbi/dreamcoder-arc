from collections import Counter, defaultdict
from copy import copy
import heapq
import itertools
from queue import PriorityQueue

import numpy as np


class NoParsesError(Exception):
  """
  No parses were computed for the given sentence.
  """
  def __init__(self, message, sentence):
    super().__init__(message)
    self.sentence = sentence


class Distribution(Counter):
  """
  Weight distribution with discrete support.
  """

  @classmethod
  def uniform(cls, support):
    ret = cls()
    for key in support:
      ret[key] = 1 / len(support)
    return ret

  @property
  def support(self):
    return self.keys()

  def ensure_support(self, keys):
    for key in keys:
      if key not in self:
        self[key] = 0.

    return self

  def __mul__(self, scale):
    assert isinstance(scale, (int, float))

    ret = Distribution()
    for key in self:
      ret[key] = self[key] * scale
    return ret

  def __add__(self, other):
    ret = copy(self)

    if isinstance(other, dict):
      for key, val in other.items():
        ret[key] += val
    else:
      for key in self:
        ret[key] += other

    return ret

  def __iadd__(self, other):
    return self + other

  def normalize(self):
    Z = sum(self.values())
    if Z != 0:
      return self * (1 / Z)
    elif Z == 0:
      return Distribution.uniform(self.keys())

  def mix(self, other, alpha=0.5):
    assert alpha >= 0 and alpha <= 1
    return self * alpha + other * (1 - alpha)

  def argmax(self):
    return max(self, key=lambda k: self[k])

  def plot(self, name, out_dir, k=5, xlabel=None, title=None, save_csv=True):
    """
    Save a bar plot of the distribution.
    """
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    support = sorted(self.keys(), key=lambda k: distribution[k], reverse=True)

    if save_csv:
      with (out_dir / ("%s.csv" % name)).open("w") as csv_f:
        for key in support:
          csv_f.write("%s,%f\n" % (key, self[key]))

    # Trim support for plot.
    if k is not None:
      support = support[:k]

    xs = np.arange(len(support))
    fig = plt.figure(figsize=(10, 8))
    plt.bar(xs, [self[support] for support in support])
    plt.xticks(xs, list(map(str, support)), rotation="vertical")
    plt.ylabel("Probability mass")
    if xlabel is not None:
      plt.xlabel(xlabel)
    if title is not None:
      plt.title(title)

    plt.tight_layout()

    path = out_dir / ("%s.png" % name)
    fig.savefig(path)


class ConditionalDistribution(object):

  def __init__(self):
    self.dists = defaultdict(Distribution)

  def __getitem__(self, key):
    return self.dists[key]

  def __setitem__(self, key, val):
    self.dists[key] = val

  def __iter__(self):
    return iter(self.dists)

  def __str__(self):
    return "{" + ", ".join("%s: %s" % (key, dist) for key, dist in self.dists.items()) + "}"

  __repr__ = __str__

  @property
  def support(self):
    return set(itertools.chain.from_iterable(
      dist.keys() for dist in self.dists.values()))

  @property
  def cond_support(self):
    return set(self.dists.keys())

  def ensure_cond_support(self, key):
    """
    Ensure that `key` is in the support of the conditioning factor.
    """
    return self.dists[key]

  def mix(self, other, alpha=0.5):
    # TODO assert that distributions are normalized
    assert 0 <= alpha and 1 >= alpha
    support = self.support
    assert support == other.support

    mixed = ConditionalDistribution()
    cond_support = self.cond_support
    other_cond_support = other.cond_support
    for key in cond_support | other_cond_support:
      if key in cond_support:
        self_dist = self[key]
      else:
        self_dist = Distribution.uniform(support)

      if key in other_cond_support:
        other_dist = other[key]
      else:
        other_dist = Distribution.uniform(support)

      mixed[key] = self_dist.mix(other_dist, alpha)

    return mixed

  def normalize_all(self):
    for key in self.dists.keys():
      self.dists[key] = self.dists[key].normalize()


class UniquePriorityQueue(PriorityQueue):
  def _init(self, maxsize):
    PriorityQueue._init(self, maxsize)
    self.values = set()

  def _put(self, item, heappush=heapq.heappush):
    if item[1] not in self.values:
      self.values.add(item[1])
      heappush(self.queue, item)

  def _get(self, heappop=heapq.heappop):
    item = heappop(self.queue)
    self.values.remove(item[1])
    return item

  def as_distribution(self):
    # NB critical region
    # NB assumes priorities are log-probabilities
    ret = Distribution()
    for priority, item in self.queue:
      val = np.exp(priority)
      ret[item] = val

    ret = ret.normalize()
    return ret


def softmax(arr, axis=-1):
    assert axis == -1
    arr = arr - arr.max(axis=axis)
    arr = np.exp(arr)
    arr /= arr.sum(axis=axis, keepdims=True)
    return arr


class tuple_unordered(tuple):
  """
  tuple which blocks ordering ops. good for tuples whose contents are
  type-strict w.r.t. ordering, but need to be put into a priority queue :)
  """
  def __lt__(self, other):
    return False
  def __gt__(self, other):
    return False
  def __eq__(self, other):
    return False
  def __hash__(self):
    return super().__hash__()
