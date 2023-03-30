# ARC Tools

## Building container for DreamCoder

```bash
cd ec/
sudo singularity build container.img singularity
cd ..
sudo singularity build container.img singularity
```

## Using arctools

```
pip install -e .
```

```bash
arctask train2

                     <Task-train 017c7c7b | 3 train | 1 test>                     
┏━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━┳━━┳━━━━━━━━┓
┃ A-in 6x3 ┃ A-out 9x3 ┃ B-in 6x3 ┃ B-out 9x3 ┃ C-in 6x3 ┃ C-out 9x3 ┃  ┃ TA-in  ┃
┡━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━╇━━╇━━━━━━━━┩
│  0 1 0   │   0 2 0   │  0 1 0   │   0 2 0   │  0 1 0   │   0 2 0   │  │ 1 1 1  │
│  1 1 0   │   2 2 0   │  1 0 1   │   2 0 2   │  1 1 0   │   2 2 0   │  │ 0 1 0  │
│  0 1 0   │   0 2 0   │  0 1 0   │   0 2 0   │  0 1 0   │   0 2 0   │  │ 0 1 0  │
│  0 1 1   │   0 2 2   │  1 0 1   │   2 0 2   │  0 1 0   │   0 2 0   │  │ 1 1 1  │
│  0 1 0   │   0 2 0   │  0 1 0   │   0 2 0   │  1 1 0   │   2 2 0   │  │ 0 1 0  │
│  1 1 0   │   2 2 0   │  1 0 1   │   2 0 2   │  0 1 0   │   0 2 0   │  │ 0 1 0  │
│          │   0 2 0   │          │   0 2 0   │          │   0 2 0   │  │        │
│          │   0 2 2   │          │   2 0 2   │          │   2 2 0   │  │        │
│          │   0 2 0   │          │   0 2 0   │          │   0 2 0   │  │        │
└──────────┴───────────┴──────────┴───────────┴──────────┴───────────┴──┴────────┘
```

```python
>>> import arc
>>> train, test = arc.load_data()
>>> train
<TaskSet: 400 tasks>
>>> train[0]
<Task-train 007bbfb7 | 5 train | 1 test>
>>> train['017c7c7b']
<Task-train 017c7c7b | 3 train | 1 test>
```