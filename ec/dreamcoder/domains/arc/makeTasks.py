from dreamcoder.domains.arc.arcPrimitives import *
import dreamcoder.domains.arc.arcPrimitives as p
from dreamcoder.domains.arc.arcInput import load_task, num_to_id, get_all_tasks

# def train_examples(task_dict):
#     # examples = [((Input(ex["input"], task_dict["train"]),),
#     #     Grid(ex["output"])) for ex in task_dict["train"]]
#     # examples += [((Input(ex["input"], task_dict["train"]),),
#     #     Grid(ex["output"])) for ex in [task_dict["test"][0]]]
#     examples = [((Grid(ex["input"]),),
#         Grid(ex["output"])) for ex in task_dict["train"]]
#     # examples += [((Grid(ex["input"]),),
#     #     Grid(ex["output"])) for ex in [task_dict["test"][0]]]
#     examples += [((Grid(ex["input"]),),
#         Grid(ex["output"])) for ex in task_dict["test"]]
#     return examples

def train_examples(task):
    examples = [((Grid(x),), Grid(y)) for x, y in task.train]
    examples += [((Grid(x),), Grid(y)) for x, y in task.test]
    return examples

def test_examples(task_dict):
    # you don't get the test input/output, so you can only check for the
    # training examples given. So mask the output grid for each train example,
    # so that it's impossible to copy the output grid for each solution.

    # still need to debug/test this code.
    def mask_output(examples, ix):
        e = [e for example in examples]
        e[ix] = {"input": e["input"], "output": e["input"]}
        return e

    examples = [((Input(ex["input"], mask_output(task_dict["train"], i)),),
        Grid(ex["output"])) for i, ex in enumerate(task_dict["train"])]
    return examples

def make_arc_task(task_id, task_num=None, test=False, use_toutput=False,
        from_eval=False):
    # task_num is an optional argument, if you want to import the task by the
    # number alone, use get_arc_task() below, which calls this function.

    # I kept this one so as not to break old code, but I'm guessing
    # doing things by number with get_arc_task() is easier.
    d = load_task(task_id)

    if test:
        raise NotImplementedError
    examples = train_examples(d)

    # examples = test_examples(d) if test else train_examples(d)

    if task_num is None:
        name = task_id
    else:
        name = str(task_num)

    if use_toutput:
        task = Task(name,
                arrow(tinput, toutput),
                examples)
    else:
        task = Task(name,
                arrow(tgrid, tgrid),
                examples)

    return task

def convert_arc_task(task, use_toutput=False):
    examples = train_examples(task)

    # examples = test_examples(d) if test else train_examples(d)

    name = task.id

    if use_toutput:
        task = Task(name,
                arrow(tinput, toutput),
                examples)
    else:
        task = Task(name,
                arrow(tgrid, tgrid),
                examples)

    return task

def get_arc_train_tasks(n=None):
    trainset = get_all_tasks()
    if n:
        trainset = trainset[:n]
    return [convert_arc_task(task) for task in trainset]

def get_arc_task(task_num, use_toutput=False):
    task_id = num_to_id(task_num)
    return make_arc_task(task_id, task_num=task_num, use_toutput=use_toutput)


# def get_eval_tasks():
#     evaluation_dir = 'data/ARC/data/evaluation/'
#     task_ids = [t[:-5] for t in os.listdir(evaluation_dir)]
#     return [make_arc_task(task_id, from_eval=True) for task_id in task_ids]
