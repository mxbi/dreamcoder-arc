[![Build Status](https://travis-ci.org/insperatum/pregex.svg?branch=master)](https://travis-ci.org/insperatum/pregex)
# pregex
Probabilistic regular expressions

- Kleene star (and +) use geometric distributions on length. 
- Currently, score returns likelihood of *most probable* execution trace, rather than marginalising.
- Add new primitives with Wrapper class
- Primitives may be stateful (so log p(/AA/->xx) is not necessarily 2 * log p(/A/->x))

Usage:

```
import pregex as pre
r = pre.create("\\d+\\l+\\u+\\s")
samp = r.sample() //'3gclxbZ\t'
score = r.match("123abcABC ") //-34.486418601378595
```

# Todo:
- [ ] use separate bracket types for each function?
- [ ] 'sample' and 'marginalise' modes.
Note -- for this, KleeneStar needs to be adapted to get correct score for fo?* -> foo.
First calculate probability q=P(o?->ε), then multiply all partialmatches by 1/[1-q(1-p))]
- [ ] Should still be able to do dynamic programming to combine partialMatches that have different states, so long as the difference in state doesn't affect the continuation
- [ ] Replace namedtuples with attrs
