"""
Contains CLEVR metadata and minor utilities for working with CLEVR.
"""

from functools import reduce

from pyccg.logic import *


# CLEVR constants
# https://github.com/facebookresearch/clevr-dataset-gen/blob/master/question_generation/metadata.json
ENUMS = {
  "shape": ["cube", "sphere", "cylinder"],
  "color": ["gray", "red", "blue", "green", "brown", "purple", "cyan", "yellow"],
  "relation": ["left", "right", "behind", "front"],
  "size": ["small", "large"],
  "material": ["rubber", "metal"],
}

# Maps object properties back to enumeration types.
# Assumes no overlap in enum values (true for now).
VAL_TO_ENUM = {val: enum for enum, vals in ENUMS.items()
               for val in vals}


ENUM_LF_TEMPLATES = {
  "shape": [r"\a.filter_shape(scene,a)"],
  "color": [r"\a.\x.filter_color(x,a)"],
  "size": [r"\a.\x.filter_size(x,a)"],
  "material": [r"\a.\x.filter_material(x,a)"],
}
ENUM_LF_TEMPLATES = {enum: [Expression.fromstring(expr)
                            for expr in exprs]
                     for enum, exprs in ENUM_LF_TEMPLATES.items()}


def scene_candidate_referents(scene):
  candidates = set()

  # for now, just enumerate object properties
  for obj in scene['objects']:
    for k, v in obj.items():
      if isinstance(v, str):
        enum = VAL_TO_ENUM[v]
        templates = ENUM_LF_TEMPLATES[enum]
        v_expr = Expression.fromstring(v)

        for template in templates:
          candidates.add(template.applyto(v_expr).simplify())

  return candidates


## CLEVR LF processing functions.

def make_application(fn_name, args):
  expr = ApplicationExpression(ConstantExpression(Variable(fn_name)),
                               args[0])
  return reduce(lambda x, y: ApplicationExpression(x, y), args[1:], expr)


def lf_merge_filters(lf):
  # this closure variable is a list so that it can be written to within the
  # recursive function
  lambda_counter = [0]

  def inner(lf, lambda_context=None):
    filter_name = "filter_and"
    def is_merged_filter(lf):
      return isinstance(lf, ApplicationExpression) \
          and lf.pred.variable.name == filter_name

    # TODO handle lambda expression?
    if not isinstance(lf, ApplicationExpression):
      return lf

    if not lf.pred.variable.name.startswith("filter_"):
      args = [inner(arg) for arg in lf.args]
      return make_application(lf.pred.variable.name, args)
    else:
      assert len(lf.args) == 2
      child, feature_val = lf.args
      assert isinstance(feature_val, ConstantExpression)
      feature_val = feature_val.variable.name.replace("'", "")

      fn = lf.pred.variable.name
      filter_type = fn[fn.index("_") + 1:]

      spawned_lambda = False
      if lambda_context is None:
        lambda_context = [Variable(chr(lambda_counter[0] + 97))]
        lambda_counter[0] += 1
        spawned_lambda = True

      # reduced form: green(obj)
      reduced_form = make_application(feature_val,
          [VariableExpression(lambda_context[-1])])

      child = inner(child, lambda_context)
      if is_merged_filter(child):
        # child is already merged; add reduced form and return
        new_args = child.args
        new_args.append(reduced_form)
        ret = make_application(child.pred.variable.name, new_args)
      else:
        # create a new function call
        args = []
        # TODO
        if not isinstance(child, ConstantExpression):
          args.append(child)
        args.append(reduced_form)

        ret = make_application(filter_name, args)

      if spawned_lambda:
        return LambdaExpression(lambda_context[-1], ret)
      else:
        return ret

  return inner(lf)


def functionalize_program(program, merge_filters=True):
  """
  Convert a CLEVR question program into a sexpr format,
  amenable to semantic parsing.
  """

  def inner(p):
    if p['function'] == 'scene':
      return 'scene'
    ret = '%s(%s' % ('exist_' if p['function'] == 'exist' else p['function'],
                     ','.join(inner(program[x]) for x in p['inputs']))
    if p['value_inputs']:
      ret += ',' + ','.join(map(repr, p['value_inputs']))
    ret += ')'
    return ret
  program_str = inner(program[-1])

  expr = Expression.fromstring(program_str)
  if merge_filters:
    expr = lf_merge_filters(expr)

  return str(expr)


if __name__ == '__main__':
  question = "Are there any other things that are the same shape as the big metallic object?"
  program = [{'inputs': [], 'function': 'scene', 'value_inputs': []}, {'inputs': [0], 'function': 'filter_size', 'value_inputs': ['large']}, {'inputs': [1], 'function': 'filter_material', 'value_inputs': ['metal']}, {'inputs': [2], 'function': 'unique', 'value_inputs': []}, {'inputs': [3], 'function': 'same_shape', 'value_inputs': []}, {'inputs': [4], 'function': 'exist', 'value_inputs': []}]
  print(functionalize_program(program, merge_filters=False))
  print(functionalize_program(program, merge_filters=True))
