# ARC DreamCoder

> **Neural networks for abstraction and reasoning: Towards broad generalization in machines**  
> *Mikel Bober-Irizar & Soumya Banerjee*

https://arxiv.org/abs/2402.03507

## Repo overview

Most of this repo follows the primary DreamCoder repo: https://github.com/ellisk42/ec.

Some helpful ARC-specific files:
- `ec/arcbin/arc_mikel2.py`: The main entry-point for DreamCoder on ARC
- `ec/dreamcoder/domains/arc/arcPrimitivesIC2.py`: PeARL definitions (domain-specific language).
- `ec/dreamcoder/domains/arc/main.py`: Recognition model
- `ec/arcbin/test_primitives_mikel2.py`: Very rough test harness to check that primitives aren't broken
- `arckit/`: Vendored early version of the [arckit](https://github.com/mxbi/arckit) library.
- `solved_tasks.md` shows a list of tasks solved by DreamCoder with corresponding programs.

## Building the DreamCoder environment

Since DreamCoder requires a complex set of dependencies, we follow the original repo in using [Singularity](https://docs.sylabs.io/guides/3.5/user-guide/introduction.html) containers. If you're familiar with Docker, this is quite similar.

The build is a 2-stage process. To use wandb, add a key in `singularity_mod` and create an `arc` project in your repo (or modify the entrypoint script to disable wandb).

```bash
cd ec/
# Build original DreamCoder (with fixes)
sudo singularity build container.img singularity

# Build additional packages and environment variables.
cd ..
sudo singularity build container_mod.img singularity_mod
```

Now, you have a `container.img` in the root of the repo which can be used to run the DreamCoder environment.

## Running experiments

```bash
# See all command-line arguments
../container_mod.img python -u arcbin/arc_mikel.py --help

# Getting 70/400 on training set
../container_mod.img python -u arcbin/arc_mikel2.py -c 76 -t 3600 -R 2400 -i 1
# -c 76: Run on 76 cores
# -t 3600: 3600 core-seconds per task
# -R 2400: Train recognition model for 2400s per iteration (all cores)
# -i 1: Run for one iteration

# 18/400 on evaluation set:
../container_mod.img python -u arcbin/arc_mikel2.py -c 76 -t 3600 -R 2400 -i 1 --evalset
# --evalset: Run on ARC-Hard

# Ablation without recognition model (1min per task)
../container_mod.img python -u arcbin/arc_mikel2.py -c 76 -t 60 -g -i 5 --task-isolation
# -g: disable recognition model
# --task-isolation: Don't share programs across multiple tasks
```

## Acknowledgements

The codebase in this repo is primarily based on the original [DreamCoder](https://github.com/ellisk42/ec) repository, licensed under MIT.

Additionally, I brought in some changes from Simon Alford's [bidir-synth](https://github.com/simonalford42/bidir-synth) repository as a starting point ([https://github.com/mxbi/arc/commit/a04da2471d327c7e39352048fed2fcd63408c3fd](commit)). The starting point was a combination of these two repos with some additional patches to get it compiling again after a couple years of changes in dependencies.

## License

The code in this repository is licensed under the MIT license. The original DreamCoder and bidir-synth repos are licened under the same license from their respective authors.

The ARC dataset (arckit/arc1.json) is licensed instead under the Apache license.