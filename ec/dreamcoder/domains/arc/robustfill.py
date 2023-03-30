import random
import numpy as np
from dreamcoder.domains.arc.arcPrimitives import ArcExample
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def make_map_task2(num_colors=7):
    def sample(i, j, k):
        return np.array([[i,j,k],[i,j,k],[i,j,k]])

    def sample_easy(i, j, k):
        return np.array([i,j,k])

    inp = 0
    def next_input():
        nonlocal inp
        inp += 1
        if inp == num_colors + 1:
            inp = 1
        return inp

    num_examples = math.ceil(num_colors/3)
    examples = []
    for i in range(num_examples):
        a, b, c = next_input(), next_input(), next_input()
        input_grid = ArcExample(sample_easy(a, b, c))
        output_grid = input_grid.transform(d).grid
        input_grid = input_grid.grid
        examples.append((input_grid, output_grid))

    return examples

def random_grid(easy=True):
    grid = ArcExample(np.random.randint(low=0,high=10,size=(1,3)))
    if not easy:
        grid = ArcExample(np.repeat(np.random.randint(
                        low=0, high=10, size=(1,3)), 3, axis=0))
    return grid

def random_grid2():
    # 1D grid, with numbers wrapped.
    i = list(range(1, 10))

def random_grid_and_program():
    num_mappings = random.randint(1, 9)
    inp = list(range(1, num_mappings+1))
    out = list(range(1, num_mappings+1))
    random.shuffle(inp)
    random.shuffle(out)
    transformation = zip(inp, out)
    inp = (inp*10)[:10]
    out = (out*10)[:10]
    program = ''
    for i, o in transformation:
        program += 'm' + str(i) + str(o)

    input_grid = np.array(inp)
    output_grid = np.array(out)

    ran = run_program(input_grid, program)
    assert np.array_equal(output_grid, ran)

    tensor_inp = torch.from_numpy(np.concatenate((input_grid, output_grid)))
    return tensor_inp, program, input_grid, output_grid

def correct_type(p):
    if len(p) % 3 != 0:
        return False

    a = np.array(list(p)).reshape(-1, 3).transpose()
    m_s = ''.join(a[0])
    if m_s != 'm'*len(m_s):
        # print('m_s bad')
        return False

    if 'm' in a[1:]:
        # print('bad m')
        return False

    return True

def run_program(i, p):
    if 'x' in p:
        p = p[:p.index('x')]

    # print('p: {}'.format(p))

    if not correct_type(p):
        return None

    o = np.copy(i)
    for m, k, v in np.array(list(p)).reshape(-1, 3):
        o[i == int(k)] = int(v)

    return o

def get_training_examples(batch_size=500):
    return [random_grid_and_program() for i in range(batch_size)]

def format_batch(data):
    tensors, programs, input_grids, output_grids = zip(*data)
    inp = torch.stack(tensors)
    inp = F.one_hot(inp, num_classes=11).to(torch.float32)
    return inp, programs, input_grids, output_grids


def convert(t):
    s = 'm123456789x'
    return ''.join(s[i.item()] for i in t)

def train():
    net = Net()
    l = format_batch(get_training_examples(batch_size=10000))
    test_inp, test_programs, test_input_grids, test_target_grids = l
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    batch_size = 16
    test_every = 100
    save_every = 100000

    total_loss = 0

    for epoch in range(10000000):

        if epoch % save_every == 0:
            print('saving')
            torch.save(net.state_dict(), 'robustfill_7_17.pt')
        if epoch % test_every == 0:
            print('epoch: {}'.format(epoch))
            net.eval()
            num_correct = 0
            probabilities, programs = net(test_inp)
            # print('programs: {}'.format(programs))
            programs = torch.stack(programs).permute(1, 0)
            programs = [convert(t) for t in programs]
            print('programs: {}'.format(programs[0:10]))
            print('target programs: {}'.format(test_programs[0:10]))
            predicted_grids = [run_program(i, p) for i,p in zip(test_input_grids, programs)]
            # print('predicted_grids: {}'.format(predicted_grids))
            # print('test_target_grids: {}'.format(test_target_grids))
            num_correct = sum([np.array_equal(predicted, target) for predicted, target in zip(predicted_grids, test_target_grids)])
            # print('num_correct: {}'.format(num_correct))
            
            percent_correct = num_correct / len(test_programs)
            print('accuracy: {}'.format(percent_correct))
            print('loss: {}'.format(total_loss))
            total_loss = 0

            net.train()

        l = get_training_examples(batch_size)
        inp, programs, input_grids, output_grids = format_batch(l)

        optimizer.zero_grad()
        
        probabilities, program_predictions = net(inp)
        programs = torch.stack([torch.tensor(['m123456789x'.index(i) for i in (m + 'x'*30)[:30]]) for m in programs])
        # print('programs: {}'.format(programs.shape))

        probabilities = torch.stack(probabilities).permute(1, 2, 0)
        # print('probabilities: {}'.format(probabilities.shape))

        loss = criterion(probabilities, programs)
        loss.backward()
        optimizer.step()

        total_loss += loss.sum().item()


    print('Finished Training')

def run():
    train()


def one_hot_grid(input_grid):
    # from
    # https://stackoverflow.com/questions/29831489/convert-array-of-indices-to-1-hot-encoded-numpy-array
    b = np.zeros((input_grid.size, input_grid.max()))
    b[np.arange(input_grid.size), input_grid-1] = 1
    return b

def undo_one_hot_grid(input_grid):
    return np.argmax(input_grid, axis=1)+1


class FC(nn.Module):
    def __init__(self):
        self.input_size = 220
        self.hidden_dim = 64
        self.output_size = 100
        self.net = nn.Sequential(
                nn.Linear(self.input_size, self.hidden_dim), nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU(),
                nn.Linear(self.hidden_dim, self.output_size))

    def forward(self, x):
        return self.net(x)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_size = 220
        self.hidden_dim = 128
        self.num_tokens = 11
        self.max_length = 30
        self.embedder = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_dim), nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU())
        # takes in hidden_dim, as well as the previous token, which is a one-hot vector of dimension 11
        self.lstm = nn.LSTM(self.hidden_dim + self.num_tokens, 
            self.hidden_dim,
            batch_first=True,
            bidirectional=False)

        self.to_tokens = nn.Linear(self.hidden_dim, self.num_tokens)

    def forward(self, x):
        """
            x: a (batch, 10,18) matrix of one-hot encodings
            returns:
                probabilities, tokens:
                where probabilities is a list of probabilities for each token at each step
                tokens is the token from each step (arg max of probabilities)
        """
        # print('x: {}'.format(x.shape))
        batch = x.shape[0]
        x = x.flatten(start_dim=1)  # (batch, 180)
        # print('x: {}'.format(x.shape))
        x = self.embedder(x) # (batch, hidden_dim)
        # print('x embedder: {}'.format(x.shape))

        # (batch, hidden_dim + num_tokens)
        x = torch.cat((x, torch.zeros((batch, self.num_tokens))), dim=1)
        # print('x: {}'.format(x.shape))
        hidden = None
        program = []
        probabilities = []
        x = x.unsqueeze(1) # (batch, 1, hidden)
        # print('x: {}'.format(x.shape))
        for i in range(self.max_length):
            out, hidden = self.lstm(x, hidden)
            # out is (batch, 1, hidden_dim)
            # hidden is tuple with (1, batch, hidden) for each
            # print('out: {}'.format(out.shape))
            # print('hidden: {}'.format(hidden[0].shape))
            # print('hidden1: {}'.format(hidden[1].shape))
            # (batch, 10)
            out = out.squeeze() # (batch, hidden_dim)
            # print('out: {}'.format(out.shape))
            token_dist = self.to_tokens(out)
            # print('token_dist: {}'.format(token_dist.shape))
            probabilities.append(token_dist)
            # print('probabilities: {}'.format(probabilities))
            top_token = token_dist.argmax(dim=1) # (batch, 1)
            # print('program: {}'.format(program))
            token_input = F.one_hot(top_token, num_classes=self.num_tokens).to(torch.float32)
            x = torch.cat((torch.zeros(batch, self.hidden_dim), token_input), dim=1)
            # (batch, 1, hidden_dim)
            x = x.unsqueeze(1)

        return probabilities, program

def make_map(program):
    i = list(range(1, 10))
    o = run_program(i, program)
    return torch.tensor(o)

def train2():
    net = FC()
    l = format_batch(get_training_examples(batch_size=10000))
    test_inp, test_programs, test_input_grids, test_target_grids = l
    optimizer = optim.Adam(net.parameters(), lr=0.0001)
    criterion = torch.nn.CrossEntropyLoss()
    batch_size = 16
    test_every = 100

    total_loss = 0

    for epoch in range(100000):

        if epoch % test_every == 1:
            print('epoch: {}'.format(epoch))
            net.eval()
            num_correct = 0
            predictions = net(test_inp)
            # print('programs: {}'.format(programs))
            programs = torch.stack(programs).permute(1, 0)
            programs = [convert(t) for t in programs]
            print('programs: {}'.format(programs[0:10]))
            print('target programs: {}'.format(test_programs[0:10]))
            predicted_grids = [run_program(i, p) for i,p in zip(test_input_grids, programs)]
            # print('predicted_grids: {}'.format(predicted_grids))
            # print('test_target_grids: {}'.format(test_target_grids))
            num_correct = sum([np.array_equal(predicted, target) for predicted, target in zip(predicted_grids, test_target_grids)])
            # print('num_correct: {}'.format(num_correct))
            
            percent_correct = num_correct / len(test_programs)
            print('accuracy: {}'.format(percent_correct))
            print('loss: {}'.format(total_loss))
            total_loss = 0

            net.train()

        l = get_training_examples(batch_size)
        inp, programs, input_grids, output_grids = format_batch(l)

        optimizer.zero_grad()
        
        print('inp: {}'.format(inp.shape))
        predictions = net(inp)
        print('predictions: {}'.format(predictions.shape))
        maps = torch.stack([make_map(p) for p in programs])
        print('maps: {}'.format(maps.shape))


        loss = criterion(predictions, maps)
        loss.backward()
        optimizer.step()

        total_loss += loss.sum().item()


    print('Finished Training')


def one_hot_grid(input_grid):
    # from
    # https://stackoverflow.com/questions/29831489/convert-array-of-indices-to-1-hot-encoded-numpy-array
    b = np.zeros((input_grid.size, input_grid.max()))
    b[np.arange(input_grid.size), input_grid-1] = 1
    return b

def undo_one_hot_grid(input_grid):
    return np.argmax(input_grid, axis=1)+1


class FC(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_size = 220
        self.hidden_dim = 64
        self.output_size = 90
        self.net = nn.Sequential(
                nn.Linear(self.input_size, self.hidden_dim), nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU(),
                nn.Linear(self.hidden_dim, self.output_size))

    def forward(self, x):
        return self.net(x.flatten(start_dim=1)).view(-1, 10, 9)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_size = 220
        self.hidden_dim = 64
        self.num_tokens = 11
        self.max_length = 30
        self.embedder = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_dim), nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU())
        # takes in hidden_dim, as well as the previous token, which is a one-hot vector of dimension 11
        self.lstm = nn.LSTM(self.hidden_dim + self.num_tokens, 
            self.hidden_dim,
            batch_first=True,
            bidirectional=False)

        self.to_tokens = nn.Linear(self.hidden_dim, self.num_tokens)

    def forward(self, x):
        """
            x: a (batch, 10,18) matrix of one-hot encodings
            returns:
                probabilities, tokens:
                where probabilities is a list of probabilities for each token at each step
                tokens is the token from each step (arg max of probabilities)
        """
        # print('x: {}'.format(x.shape))
        batch = x.shape[0]
        x = x.flatten(start_dim=1)  # (batch, 180)
        # print('x: {}'.format(x.shape))
        x = self.embedder(x) # (batch, hidden_dim)
        # print('x embedder: {}'.format(x.shape))

        # (batch, hidden_dim + num_tokens)
        x = torch.cat((x, torch.zeros((batch, self.num_tokens))), dim=1)
        # print('x: {}'.format(x.shape))
        hidden = None
        program = []
        probabilities = []
        x = x.unsqueeze(1) # (batch, 1, hidden)
        # print('x: {}'.format(x.shape))
        for i in range(self.max_length):
            out, hidden = self.lstm(x, hidden)
            # out is (batch, 1, hidden_dim)
            # hidden is tuple with (1, batch, hidden) for each
            # print('out: {}'.format(out.shape))
            # print('hidden: {}'.format(hidden[0].shape))
            # print('hidden1: {}'.format(hidden[1].shape))
            # (batch, 10)
            out = out.squeeze() # (batch, hidden_dim)
            # print('out: {}'.format(out.shape))
            token_dist = self.to_tokens(out)
            # print('token_dist: {}'.format(token_dist.shape))
            probabilities.append(token_dist)
            # print('probabilities: {}'.format(probabilities))
            top_token = token_dist.argmax(dim=1) # (batch, 1)
            # print('top_token: {}'.format(top_token.shape))
            program.append(top_token)
            # print('program: {}'.format(program))
            token_input = F.one_hot(top_token, num_classes=self.num_tokens).to(torch.float32)
            # print('token_input: {}'.format(token_input.shape))
            # (batch, hidden_dim + num_tokens)
            x = torch.cat((torch.zeros(batch, self.hidden_dim), token_input), dim=1)
            # (batch, 1, hidden_dim)
            x = x.unsqueeze(1)

        return probabilities, program
