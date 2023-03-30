from dreamcoder.domains.arc.arcInput import num_to_id, load_task
from dreamcoder.domains.arc.arcPrimitives import Grid, Input
from dreamcoder.domains.arc.arcPrimitives import primitive_dict as p
from dreamcoder.domains.arc.main import ArcNet
from dreamcoder.task import Task
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import random
import time


def get_grids():
    grids = []
    for i in range(400):
        task_id = num_to_id(i)

        task_dict = load_task(task_id, task_path="data/ARC/data/training/")
        for x in task_dict["train"] + task_dict["test"]:
            grids.append(x["input"])
            grids.append(x["output"])

    return grids

def get_operators():
    names = ['rotate_cw', 'rotate_ccw', 'y_mirror', 'x_mirror', 'top_half', 'right_half', 'bottom_half', 'left_half', 'shell', 'hollow']

    # the lambda function for each of them.
    operators = [p[name].value for name in names]

    d = dict(zip(names, operators))
    d['identity'] = lambda x: x

    return d

def export_data(samples, path):
    # samples is (grid, grid, name)
    def to_str(grid):
        return str(grid.shape) + ', ' + ' '.join(map(str, list(grid.flatten())))

    def line_of_example(example):
        inp, outp, op = example
        return 'IN: {}, OUT: {}, OP: {}'.format(to_str(inp.grid),
                                                to_str(outp.grid),
                                                op)

    lines = '\n'.join(line_of_example(s) for s in samples)
    with open(path, 'w+') as f:
        f.write(lines)


def generate_dataset():
    grids = get_grids()
    grids = [g for g in grids if min(g.shape) > 2]
    grids = [Grid(g) for g in grids]
    print('{} grids'.format(len(grids)))

    operators = get_operators()
    print('{} operators'.format(len(operators)))


    num_unique = 0
    num_total = len(operators)*len(grids)
    print('num_total: {}'.format(num_total))
    samples = [(grid, f(grid), name) for grid in grids for name, f in
            operators.items()]
    # for grid in grids:
    #     outputs = []
    #     for f in operators.values():
    #         match = np.any([np.all(o == f(grid)) for o in outputs])
    #         if not match:
    #             num_unique += 1
    #             outputs.append(f(grid))

    # print('num_unique: {}'.format(num_unique))

    # export_data(samples, 'arcnet_data.txt')
    data = import_data('arcnet_data.txt')
    export_data(data, 'arcnet_data2.txt')



def import_data(path):
    def parse_example(line):
        def parse_grid(grid_txt):
            i1 = grid_txt.index('(')
            assert i1 == 0
            i2 = grid_txt.index(')')
            shape = grid_txt[i1 + 1:i2]
            shape = shape.split(', ')
            w, h = list(map(int, shape))

            rest = grid_txt[grid_txt.index(')') + 3:]
            one_d = np.fromstring(rest, dtype=int, sep=' ')
            out = one_d.reshape(w, h)
            return out

        i1 = line.index('IN: ')
        i2 = line.index('OUT: ')
        i3 = line.index('OP: ')
        grid1 = line[i1 + 4:i2 - 2]
        grid2 = line[i2 + 5:i3 - 2]
        op = line[i3 + 4:-1]
        grid1 = parse_grid(grid1)
        grid2 = parse_grid(grid2)
        examples = [((Input(grid1, []),), Grid(grid2))]

        return examples, op

    with open(path, 'r') as f:
        lines = f.readlines()
        return [parse_example(l) for l in lines]


class FullNet(nn.Module):
    """Puts a FC layer onto ArcNet to match the output dim."""
    def __init__(self, out_dim):
        super().__init__()
        self.arc_net = ArcNet(tasks=[])
        self.fc = nn.Linear(self.arc_net.outputDimensionality, out_dim)

    def forward(self, examples):
        # examples is (N, num_examples, C, H, W)
        # input to arc_net should be (num_examples, C, H, W)
        # so flatten N into num_examples range
        # (N*num_examples, C, H, W)
        examples = torch.flatten(examples, start_dim=0, end_dim=1)
        return self.fc(self.arc_net(examples))


class FCNet(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.fc = nn.Linear(2*30*30*10, out_dim)

    def forward(self, examples):
        x = self.make_features(examples)
        x = torch.flatten(x)
        x = x.to(torch.float32)
        return self.fc(x)

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


def make_features(examples):
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


def train():
    torch.set_num_threads(10)
    # list of (examples, op_str)
    data = import_data('arcnet_data.txt')

    ops = sorted(list(set([d[1] for d in data])))
    op_dict = dict(zip(ops, range(len(ops))))

    # convert (examples, op_str) to (input_tensor, target_tensor)
    def tensorize(datum):
        (examples, op_str) = datum
        # (1, num_colors, h, w)
        input_tensor = make_features(examples)
        # (1)
        target_tensor = torch.tensor(op_dict[op_str])
        return input_tensor, target_tensor

    data = [tensorize(d) for d in data]

    net = FullNet(len(op_dict))
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    batch_size = 16
    epochs = 3100

    def make_batches(data, batch_size):
        return [data[batch_size * i: min((i + 1) * batch_size, len(data))]
                for i in range(math.ceil(len(data) / batch_size))]

    print('Starting training...')
    for epoch in range(1, epochs+1):
        random.shuffle(data)
        batches = make_batches(data, batch_size)
        start = time.time()

        total_loss = 0
        total_correct = 0
        for i, batch in enumerate(batches):
            # batch is list of (in_grid, out_grid, op_str) tuples
            examples, targets = zip(*batch)

            print(examples[0].shape)
            examples = torch.cat(examples)
            print('examples: {}'.format(examples))
            assert False

            targets_tensor = torch.tensor(targets)

            optimizer.zero_grad()

            out = net(examples)
            out = [t.unsqueeze(0) for t in out]
            out = torch.cat(out)
            predictions = torch.argmax(out, dim=1)
            # print('predictions: {}'.format(predictions))

            loss = criterion(out, targets_tensor)
            loss.backward()
            optimizer.step()

            total_loss += loss.sum().item()

            num_correct = sum(t == p for t, p in zip(targets,
                              predictions))
            # print('num_correct: {}'.format(num_correct))

            total_correct += num_correct

        accuracy = 100*total_correct / len(data)
        duration = time.time() - start
        m = math.floor(duration / 60)
        s = duration - m * 60
        duration = f'{m}m {int(s)}s'

        print(f'Epoch {epoch} completed ({duration}) accuracy: {accuracy:.2f} loss: {loss:.2f}')


    print('Finished Training')



