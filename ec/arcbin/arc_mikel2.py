import binutil
import dill
import numpy as np
import os
import time

from dreamcoder.dreamcoder import commandlineArguments, ecIterator
from dreamcoder.grammar import Grammar
from dreamcoder.domains.arc.makeTasks import get_arc_task, get_arc_tasks
from dreamcoder.domains.arc.main import MikelArcNet
from dreamcoder.domains.arc import arcPrimitivesIC2

import wandb

primitives = arcPrimitivesIC2.dsl.primitives.values()
arcPrimitivesIC2.dsl.generate_ocaml_primitives()

# make a starting grammar to enumerate over
grammar = Grammar.uniform(primitives)

def extra_args(parser):
    parser.add_argument('--evalset', action='store_true', default=False, help='Use the eval set instead of the train set')
    parser.add_argument('--bothset', action='store_true', default=False, help='Use both datasets (800 tasks)')
    parser.add_argument('--task-isolation', action='store_true', default=False, help='Isolate tasks from each other')

# generic command line options
args = commandlineArguments(
    enumerationTimeout=120, 
    aic=0.1,
    iterations=1, 
    recognitionTimeout=360,
    featureExtractor=MikelArcNet,
    useRecognitionModel=True,#True,
    # contextual=True,
    a=3, 
    maximumFrontier=10, 
    topK=5, 
    pseudoCounts=30.0,
    structurePenalty=0.1,
    solver='python',
    compressor='ocaml',
    CPUs=48,
    
    extras=extra_args,
    )

wandb_config = args.copy()
wandb_config['hostname'] = os.uname()[1]
run = wandb.init(
    # set the wandb project where this run will be logged
    project="arc",
    # track hyperparameters and run metadata
    config=wandb_config,
    save_code=True,
    # magic=True,
)

run_id = run.id#int(time.time())
print(f'Run ID: {run_id}')

# run.define_metric('recog-loss', summary='min', goal='minimise', step_metric='recog-iter')
# symmetry_tasks = [30, 38, 52, 56, 66, 70, 82, 86, 105, 108, 112, 115, 116, 139, 141, 149, 151, 154, 163, 171, 176, 178, 179, 209, 210, 240, 241, 243, 248, 310, 345, 359, 360, 379, 371, 384]
# training = [get_arc_task(i) for i in symmetry_tasks]
run.define_metric('iteration')
if args['evalset']:
    print('Running on eval-set')
    training = get_arc_tasks(n=400, eval=True)
    run.define_metric('test-hit1-eval', summary='max', goal='maximize', step_metric='iteration')
    run.define_metric('test-hit3-eval', summary='max', goal='maximize', step_metric='iteration')
elif args['bothset']:
    print('Running on both sets')
    training = get_arc_tasks(n=400, eval=False) + get_arc_tasks(n=400, eval=True)
    run.define_metric('test-hit1-both', summary='max', goal='maximize', step_metric='iteration')
    run.define_metric('test-hit3-both', summary='max', goal='maximize', step_metric='iteration')
else:
    print('Running on train-set')
    training = get_arc_tasks(n=400, eval=False)
    run.define_metric('test-hit1', summary='max', goal='maximize', step_metric='iteration')
    run.define_metric('test-hit3', summary='max', goal='maximize', step_metric='iteration')

run.define_metric('batch')
run.define_metric('recog-loss', summary='min', goal='minimise', step_metric='batch')
run.define_metric('recog-mdl', summary='min', goal='minimise', step_metric='batch')
run.define_metric('recog-class-loss', summary='min', goal='minimise', step_metric='batch')

# iterate over wake and sleep cycles for our task
os.makedirs('./experimentOutputs/arc/', exist_ok=True)
generator = ecIterator(grammar,
                       training,
                       testingTasks=[],
                       outputPrefix='./experimentOutputs/arc/',
                       **args)

def test_evaluate(task, soln):
    corrects_list = []
    n_test = len(task.test_examples)
    for i, frontier in enumerate(soln.entries):
        f = frontier.program.evaluate([])
        
        corrects = 0
        for (input_grid, ), output_grid in task.test_examples:
            try:
                corrects += np.array_equal(output_grid.grid, f(input_grid).grid)
            except Exception as e:
                print(f'Exception {e} for {task.name} with {soln.entries[i].program.body}')
            
        corrects_list.append(corrects)
        
    if n_test in corrects_list:
        correct_index = corrects_list.index(n_test)
        print(f'HIT @ {correct_index+1} for {task.name} with {soln.entries[correct_index].program.body}')
        return correct_index == 0, correct_index < 3
    else:
        print(f'FAIL: Evaluated {len(soln)} solns for task {task.name}, no successes.')
        return False, False

# run the DreamCoder learning process for the set number of iterations
for i, result in enumerate(generator):
    # print(result)

    # Evaluate on test set
    print('Test set evaluation')
    hit1, hit3 = 0, 0
    for task, soln in result.taskSolutions.items():
        if len(soln.entries) == 0:
            continue
        try:
            h1, h3 = test_evaluate(task, soln)
        except Exception as e:
            print(f'Exception {e} while evaluating {task.name}')
            h1, h3 = False, False
        hit1 += h1
        hit3 += h3

    print(f'Test summary: {hit1} ({hit1/len(result.taskSolutions):.1%}) acc@1, {hit3} ({hit3/len(result.taskSolutions):.1%}) acc@3')

    os.makedirs('results/', exist_ok=True)
    dill.dump(result, open(f'results/result_{run_id}_{i}.pkl', 'wb'))
    print('ecIterator count {}'.format(i))

    if args['evalset']:
        wandb.log({'test-hit1-eval': hit1, 'test-hit3-eval': hit3, 'iteration': i})
    elif args['bothset']:
        wandb.log({'test-hit1-both': hit1, 'test-hit3-both': hit3, 'iteration': i})
    else:
        wandb.log({'test-hit1': hit1, 'test-hit3': hit3, 'iteration': i})
