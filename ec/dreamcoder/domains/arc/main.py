import numpy as np
import random 
import torch
import torch.nn as nn
import torch.nn.functional as F

from dreamcoder.task import Task
from dreamcoder.domains.arc.modules import AllConv

def check_test_accuracy(ec_result):
    '''
        A task is 'hit' if we solve all of the training examples. Now we check
        the top 3 programs, and see how many successfully solve the test
        example.
    '''

    print('hitsAtEachWake: {}'.format(ec_result.hitsAtEachWake))
    total_attempted = 0
    total_solved = 0
    failed = []
    for frontiers in ec_result.frontiersOverTime.values():
        frontier = frontiers[-1]
        task = frontier.task
        if len(frontier.entries) == 0:
            continue
        total_attempted += 1
        # TODO should check that this is correct criteria
        top_entries = sorted(frontier.entries, key=lambda e: -e.logPosterior)
        top_3 = top_entries[0:3]
        solved = False
        for entry in top_3:
            # try solving
            program = entry.program
            test_examples = task.test_examples
            failed_example = False
            for xs, y in test_examples:
                f = program.evaluate([])
                for a in xs:
                    f = f(a)
                out = f
                if out != y:
                    failed_example = True

            if not failed_example:
                solved = True
                break

        if solved:
            # print('Solved test example(s) for task {}!'.format(str(task)))
            total_solved += 1
        else:
            failed.append(str(task))
            # print('Failed test example(s) for task{}'.format(str(task)))

    print('Solved {}/{} test examples within 3 tries'.format(
            total_solved, total_attempted))
    if total_attempted > total_solved:
        print('Failed on tasks {}'.format(', '.join(tasks)))

                    
            



class ArcNet(nn.Module):
    special = "ARC" # needed if we ever try bias optimal learning

    def __init__(self, tasks, testingTasks=[], cuda=False):
        super().__init__()

        # for sampling input grids during dreaming
        self.grids = []
        for t in tasks:
            self.grids += [ex[0][0] for ex in t.examples]

        # maybe this should be True, See recognition.py line 908
        self.recomputeTasks = False

        self.num_examples_list = [len(t.examples) for t in tasks]

        # need to keep this named this for some other part of dreamcoder.
        self.outputDimensionality = 64

        self.all_conv = AllConv(residual_blocks=2,
                residual_filters=32,
                conv_1x1s=2,
                output_dim=self.outputDimensionality,
                conv_1x1_filters=64,
                pooling='max')


    def forward(self, x):
        # (num_examples, num_colors, h, w) to (num_examples, intermediate_dim)
        x = x.to(torch.float32)
        x = self.all_conv(x)

        # sum features over examples
        # (num_examples, intermediate_dim) to (intermediate_dim)
        x = torch.sum(x, 0)

        # test if this is actually helping.
        # return torch.rand(x.shape)

        return x

    def make_features(self, examples):
        # zero pad, concatenate, one-hot encode
        def pad(i):
            a = np.zeros((30, 30))
            if i.size == 0:
                return a

            # if input is larger than 30x30, crop it. Must be a created grid
            i = i[:min(30, len(i)),:min(30, len(i[0]))]
            a[:len(i), :len(i[0])] = i
            return a

        examples = [(pad(ex[0][0].input_grid.grid), pad(ex[1].grid))
                for ex in examples]
        examples = [torch.from_numpy(np.concatenate(ex)).to(torch.int64)
                for ex in examples]
        input_tensor = F.one_hot(torch.stack(examples), num_classes=10)
        input_tensor = input_tensor.permute(0, 3, 1, 2)
        # (num_examples, num_colors, h, w)
        return input_tensor


    #  we subclass nn.module, so this calls __call__, which calls forward()
    #  above
    def featuresOfTask(self, t):
        # t.features is created when the task is made in makeArcTasks.py
        return self(self.make_features(t.examples))


    # make tasks out of the program, to learn how the program operates (dreaming)
    def taskOfProgram(self, p, t):
        num_examples = random.choice(self.num_examples_list)

        def generate_sample():
            grid = random.choice(self.grids)
            try: 
                #print('p: {}'.format(p))
                out = p.evaluate([])(grid)
                example = (grid,), out
                return example
            except ValueError:
            # except:
                return None

        examples = [generate_sample() for _ in range(num_examples)]
        if None in examples:
            return None
        t = Task("Helm", t, examples)
        return t
