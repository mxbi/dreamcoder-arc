import json
from dreamcoder.program import Primitive
import os
import numpy as np
import re

import arckit

def load_task(task_id):
    return arckit.load_single(task_id)#.dreamcoder_format()

# def load_task(task_id, task_path='data/ARC/data/training/'):
#     filename = task_path + task_id + '.json'

#     with open(filename, 'r') as f:
#         task_dict = json.load(f)

#     task_dict['name'] = task_id

#     # turn to np arrays
#     train = task_dict["train"]
#     for ex in train:
#         for key in ex:
#             ex[key] = np.array(ex[key])

#     test=task_dict["test"]
#     for ex in test:
#         for key in ex:
#             ex[key] = np.array(ex[key])

#     # print(task_dict)

#     return task_dict

def num_to_id(task_num):
    with open('dreamcoder/domains/arc/task_number_ids.txt', 'r') as f:
        lines = [l[:-1] for l in f]
    lines = [(l.split(' ')[0], l.split(' ')[-1]) for l in lines]
    lines = [(int(a), b[:-5]) for (a, b) in lines]
    d = dict(lines)
    return d[task_num]
   

def run_stuff():
    d = load_task('0d3d703e')
    print(d)
    print(d['train'][0])
    print(d['train'][1])
    print(d['train'])
    print(d['test'])

def find_example(grid, tasks):
    for d in tasks:
        present = np.any([np.array_equal(grid, i) for i in d["grids"]])
        if present:
            return d["name"]
    return None


# def get_all_tasks():
#     "Gets all the training tasks"
#     trainset, testset = arckit.load_data()

#     # return [task.dreamcoder_format() for task in trainset]
#     return trainset

# def get_all_tasks():
#     training_dir = 'data/ARC/data/training/'
#     # take off last five chars of name to get rid of '.json'
#     task_ids = [t[:-5] for t in os.listdir(training_dir)]

    # def grids(task):
    #     grids = []
    #     for ex in task['train']:
    #         grids.append(np.array(ex['input']))
    #         grids.append(np.array(ex['output']))
    #     for ex in task['test']:
    #         grids.append(np.array(ex['input']))
    #         grids.append(np.array(ex['output']))

    #     return {"name": task["name"], "grids": grids}

#     tasks = [load_task(task_id) for task_id in task_ids]
#     tasks = [grids(task) for task in tasks]
#     return tasks

def export_tasks(path, tasks):
    # makes the json file which can be previewed through the arc interface.
    # useful for debugging!
    # puts the test tasks with the training examples
    for task in tasks:
        with open(path + '/' + str(task.name) + '.json', 'w+') as f:
            examples = task.examples
            examples = [(ex[0][0].input_grid.tolist(), ex[1].grid.tolist()) for ex in examples]
            examples = [{'input': i, 'output': o} for i, o in examples]
            d = {'train': examples, 'test': [{'input': [[0]], 'output':
                [[0]]}]}
            s = str(d)
            s = re.sub('\n', '', s)
            s = re.sub(' +', ' ', s)
            s = re.sub('\'', '"', s)
            # print('s: {}'.format(s))
            f.write(s)


def export_dc_demo(path, tasks, consolidation_dict={}):
    with open(path, 'w+') as f:
        d = {}
        frames = set()
        for task in tasks:
            task_d = {}
            task_d["json_name"] = task.arc_json
            task_d["full_task"] = task.arc_task_dict
            task_d["input_grid"] = task.arc_task_dict["test"][0]["input"]
            task_d["output_grid"] = task.arc_task_dict["test"][0]["output"]
            task_d["solved_number"] = task.arc_solved_number
            task_d["solved_iteration"] = task.arc_solved_iteration
            task_d["solved_program"] = task.arc_solved_program
            # task_d["consolidated_primitives_used"] = task.arc_consolidated_primitives_used
            frames.update(task.arc_grids.keys())
            for key in task.arc_grids.keys():
                task_d["grid_" + str(key)] = task.arc_grids[key] if 0 not in np.array(task.arc_grids[key]).shape not in [[], [[]]] else [0]
            d["task_" + str(task.name)]  = task_d

        d["consolidation"] = consolidation_dict
        frames = list(frames)
        frames = sorted(frames)
        d["frames"] = frames

        s = str(d)
        s = re.sub('\n', '', s)
        s = re.sub(' +', ' ', s)
        s = re.sub('\'', '"', s)
        s = re.sub('{', '{\n', s)
        s = re.sub('"task_', '\n\t"task_', s)
        s = re.sub('"json_name', '\t\t"json_name', s)
        s = re.sub('"full_task', '\n\t\t"full_task', s)
        s = re.sub('"input_grid', '\n\t\t"input_grid', s)
        s = re.sub('"output_grid', '\n\t\t"output_grid', s)
        s = re.sub('"train"', '\t\t\t"train"', s)
        s = re.sub('"input"', '\t\t\t\t\t"input"', s)
        s = re.sub('"output"', '\n\t\t\t\t\t"output"', s)
        s = re.sub('"test"', '\n\t\t\t"test"', s)
        s = re.sub('"solved_number', '\n\t\t"solved_number', s)
        s = re.sub('"solved_program', '\n\t\t"solved_program', s)
        s = re.sub('"consolidated_primitives', '\n\t\t"consolidated_primitives', s)
        s = re.sub('"grid_', '\n\t\t"grid_', s)
        s = re.sub('}', '\n}', s)
        s = re.sub('"consolidation', '\n\t"consolidation', s)
        s = re.sub('"new_primitives', '\t\t"new_primitives', s)
        s = re.sub('"made_in_iteration', '\n\t\t"made_in_iteration', s)
        # print('s: {}'.format(s))
        f.write(s)


def make_consolidation_dict(ec_result):
    consolidation_dict = {}

    def productionKey(xxx_todo_changeme):
        (l, t, p) = xxx_todo_changeme
        return not isinstance(p, Primitive), l is not None and -l

    new_primitives = [(l, t, p) for (l, t, p) in sorted(ec_result.grammars[-1].productions, key=productionKey) if not isinstance(p, Primitive)]

    primitives_each_round = [[p for (l, t, p) in sorted(g.productions, key=productionKey) if not isinstance(p, Primitive)] for g in ec_result.grammars]
    primitives_each_round = [[p for p in productions if np.all([p not in l for l in
        primitives_each_round[0:i]])] for i, productions in
        enumerate(primitives_each_round)]


    p_to_i_map = {p: i for i, (l, t, p) in enumerate(new_primitives)}

    primitives_each_round = primitives_each_round[1:]
    for r, prims in enumerate(primitives_each_round):
        consolidation_dict['made_in_iteration_{}'.format(r)] = [p_to_i_map[p] for p in prims]


    consolidation_dict['new_primitives'] = {str(i): [str(t), str(p)] for i, (l, t, p) in
            enumerate(new_primitives)}

    return consolidation_dict



if __name__ == '__main__':
    tasks = get_all_tasks()
    grid = np.array([[8, 5, 0], [8, 5, 3], [0, 3, 2]])
    print(find_example(grid, tasks))
