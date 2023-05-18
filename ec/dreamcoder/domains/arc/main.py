import numpy as np
import random 
import torch
import torch.nn as nn
import torch.nn.functional as F

from dreamcoder.task import Task

from typing import List

class MikelArcNet(nn.Module):
    special = "ARC" # needed if we ever try bias optimal learning

    def __init__(self, tasks: List[Task], testingTasks=[], cuda=False):
        super().__init__()

        # for sampling input grids during dreaming
        self.input_grids = []
        for t in tasks:
            self.input_grids += [ex[0][0] for ex in t.examples]
        # maybe this should be True, See recognition.py line 908
        self.recomputeTasks = False

        self.num_examples_list = [len(t.examples) for t in tasks]

        # need to keep this named this for some other part of dreamcoder.
        self.outputDimensionality = 64

        # self.all_conv = AllConv(residual_blocks=2,
        #         residual_filters=32,
        #         conv_1x1s=2,
        #         output_dim=self.outputDimensionality,
        #         conv_1x1_filters=64,
        #         pooling='max')

        self.convblock0 = nn.Sequential(
            nn.Conv2d(10, 32, 3, padding=1),
            # nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 32, 3, padding=1),
            # nn.BatchNorm2d(32),
            nn.ReLU(),

            # nn.Conv2d(32, 64, 3, padding=1),
            # # nn.BatchNorm2d(64),
            # nn.ReLU(),

            # nn.AdaptiveAvgPool2d((1, 1)),
            # nn.Flatten(),
        )

        self.res1 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=2, dilation=2),
            nn.ReLU(),
        )

        self.res2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=3, dilation=3),
            nn.ReLU(),
        )

        self.output = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((3, 3)),
            nn.Flatten(),
            # nn.Linear(2*2*64, 64),
        )

        self.linear = nn.Linear(3*3*64, 64)

    def forward(self, x, y):
        # (num_examples, num_colors, h, w) to (num_examples, intermediate_dim)
        x = x.to(torch.float32)
        y = y.to(torch.float32)
        # print(x.shape, y.shape)
        try:
            # x = self.model(x)
            x = self.convblock0(x)
            x = self.res1(x) + x
            x = self.res2(x) + x
            x = self.output(x)

            y = self.convblock0(y)
            y = self.res1(y) + y
            y = self.res2(y) + y
            y = self.output(y)
        except:
            # print(x, y)
            raise

        diff = y - x
        diff = self.linear(diff)

        # sum features over examples
        # (num_examples, intermediate_dim) to (intermediate_dim)
        # d = torch.sum(x, 0)

        # test if this is actually helping.
        # return torch.rand(x.shape)

        return diff

    def make_features(self, examples):
        """
        Returns either:
        - True, Tensor, Tensor. In this case, all grids were the same size, and so can be processed as one batch (Bx9xWxH)
        - False, List[Tensor], List[Tensor]. In this case, grids differed in size, so must be processed by the network separately.
        """

        # Check if all input grids are the same size
        try:
            inputs = [ex[0][0].grid for ex in examples]
            outputs = [ex[1].grid for ex in examples]
        except:
            print(examples)
            print("FAILED ONE")
            outputs = inputs
        if all([i.shape == inputs[0].shape for i in inputs]) and all([o.shape == outputs[0].shape for o in outputs]):
            # Single batch for each
            inputs = np.stack(inputs)
            outputs = np.stack(outputs)

            inputs = torch.from_numpy(inputs).to(torch.int64)
            outputs = torch.from_numpy(outputs).to(torch.int64)

            # One-hot, we keep the first slice in case of padding later
            inputs = F.one_hot(inputs, num_classes=10).permute(0, 3, 1, 2)
            outputs = F.one_hot(outputs, num_classes=10).permute(0, 3, 1, 2)

            return True, inputs, outputs
        else:
            # Process separately
            inputs = [torch.from_numpy(i.astype(np.int64)).to(torch.int64)[np.newaxis, :, :] for i in inputs]
            outputs = [torch.from_numpy(o.astype(np.int64)).to(torch.int64)[np.newaxis, :, :] for o in outputs]

            # One-hot
            inputs = [F.one_hot(i, num_classes=10).permute(0, 3, 1, 2) for i in inputs]
            outputs = [F.one_hot(o, num_classes=10).permute(0, 3, 1, 2) for o in outputs]

            return False, inputs, outputs



    # def make_features(self, examples):
    #     # zero pad, concatenate, one-hot encode
    #     def pad(i):
    #         a = np.zeros((30, 30))
    #         if i.size == 0:
    #             return a

    #         # if input is larger than 30x30, crop it. Must be a created grid
    #         i = i[:min(30, len(i)),:min(30, len(i[0]))]
    #         a[:len(i), :len(i[0])] = i
    #         return a

    #     examples = [(pad(ex[0][0].input_grid.grid), pad(ex[1].grid))
    #             for ex in examples]
    #     examples = [torch.from_numpy(np.concatenate(ex)).to(torch.int64)
    #             for ex in examples]
    #     input_tensor = F.one_hot(torch.stack(examples), num_classes=10)
    #     input_tensor = input_tensor.permute(0, 3, 1, 2)
    #     # (num_examples, num_colors, h, w)
    #     return input_tensor


    #  we subclass nn.module, so this calls __call__, which calls forward()
    #  above
    def featuresOfTask(self, t):
        # t.features is created when the task is made in makeArcTasks.py
        try:
            single_batch, x1, x2 = self.make_features(t.examples)
        except:
            print(t.examples)
            raise
        if single_batch:
            # print(x1.shape, x2.shape)
            # ret = self(x2)-self(x1) # difference between output and input
            ret = torch.sum(self(x1, x2), axis=0)
        else:
            ret = []
            for x, y in zip(x1, x2):
                ret.append(self(x, y)[0])

            ret = torch.mean(torch.stack(ret), axis=0)
            # x1 = [self(x) for x in x1]
            # x2 = [self(x) for x in x2]
            # print(len(x1), x1[0].shape, x1[1].shape)
            # print(len(x2), x2[0].shape, x2[2].shape)
            # ret = torch.sum(torch.stack(x1), axis=0) - torch.sum(torch.stack(x2), axis=0)
            # return torch.mean(torch.stack([self(x2[i])-self(x1[i]) for i in range(len(x1))]), axis=0)

        # print(ret.shape)
        return ret

    # make tasks out of the program, to learn how the program operates (dreaming)
    def taskOfProgram(self, p, t):
        num_examples = random.choice(self.num_examples_list)

        def generate_sample():
            grid = random.choice(self.input_grids)
            try: 
                #print('p: {}'.format(p))
                out = p.evaluate([])(grid)
                example = (grid,), out

                if out.grid.size == 0:
                    # print("no elements in this one", p)
                    return None

                return example
            except Exception as e:
                # print(e)
                # import traceback
                # traceback.print_exc()
            # except:
                return None

        examples = [generate_sample() for _ in range(num_examples)]
        if None in examples:
            return None
        t = Task("Helm", t, examples)
        return t
