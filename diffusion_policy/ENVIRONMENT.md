# Environment notes: why this fork no longer uses the upstream pin set

*Last updated: 2026-09-17*

This document records a dependency conflict that makes the upstream Diffusion Policy
environment unusable for RoboCasa365 experiments, and the resolution this fork adopts.
Read it before changing any version pin in [setup.py](setup.py) or
[conda_environment.yaml](conda_environment.yaml).

## The conflict

The four repos that make up a RoboCasa365 training run declare mutually exclusive
dependencies:

| package | `robomimic/setup.py` wants | `robocasa/setup.py` + `lerobot==0.3.3` want |
|---|---|---|
| numpy | `==1.23.2` | `==2.2.5` (hard `assert` in `robocasa/__init__.py`) |
| torch | `==2.0.1` | `>=2.2.1,<2.8` |
| torchvision | `==0.15.2` | `>=0.21,<0.23` |
| diffusers | `==0.11.1` | `>=0.27.2` |
| huggingface_hub | (via diffusers 0.11.1) `<0.26` | `>=0.34.2` |

`robomimic` encodes the original 2022-era Diffusion Policy stack. `robocasa` encodes the
2025 LeRobot stack. Whichever is `pip install -e .`'d last wins, and the other breaks.

Both sides are on the training path and neither can simply be dropped:

- The task configs (e.g.
  [`config/task/robocasa/target_atomic_seen.yaml`](diffusion_policy/config/task/robocasa/target_atomic_seen.yaml))
  load data through `LerobotCotrainingDataset` and run rollouts through
  `RobomimicImageRunner`, which imports `robocasa.utils.lerobot_utils` →
  `lerobot.datasets.*`.
- `import robocasa` asserts `numpy == 2.2.5` at import time
  (`robocasa/__init__.py:1013`), so the old numpy pin fails immediately.
- lerobot decodes the `video.*` observation keys with `torchcodec`, which requires a
  recent torch.

## Symptoms

Installing the old pins into a modern environment produces a chain of errors that look
unrelated but all trace back to this conflict:

```
ImportError: cannot import name 'HfFolder' from 'huggingface_hub'
```
diffusers 0.11.1 uses `HfFolder`, removed in huggingface_hub 1.x.

```
ImportError: cannot import name 'is_offline_mode' from 'huggingface_hub'
```
Downgrading huggingface_hub to satisfy diffusers 0.11.1 then breaks transformers 5.x,
which requires `huggingface_hub>=1.5`. **Downgrading hub is not a fix** — the two
constraints cannot both be met.

```
AssertionError: numpy version must be 2.2.5. Please install this version.
```
`import robocasa` against robomimic's numpy pin.

```
ImportError: cannot import name 'get_ctx' from 'torch.library'
```
`import torchcodec` against torch 2.0.1.

## Resolution

Standardize on the modern stack and make this fork work against it. This direction was
chosen because robocasa's numpy assert and lerobot are hard requirements, whereas this
repo's diffusers surface is small and version-stable: only `DDPMScheduler`/`DDIMScheduler`
and `diffusers.optimization`.

Verified working version set (CUDA 12.6, driver 595.91, RTX 4070 Ti):

| package | version |
|---|---|
| python | 3.11 |
| torch | 2.7.1+cu126 |
| torchvision | 0.22.1+cu126 |
| torchcodec | 0.4.0 |
| numpy | 2.2.5 |
| numba | 0.61.2 |
| diffusers | 0.40.0 |
| huggingface_hub | 1.32.0 |
| hydra-core | 1.3.2 |
| lerobot | 0.3.3 |
| robocasa / robosuite / robomimic | 1.0.1 / 1.5.2 / 0.3.0 (editable) |

### Changes made

1. **[setup.py](setup.py)** — added `install_requires`. It was previously
   `setup(name=..., packages=...)` with no dependencies at all, which is why
   `pip install -e .` produced an environment with no `hydra` or `omegaconf`. Bounds are
   chosen to be co-installable with robocasa and lerobot.

2. **[conda_environment.yaml](conda_environment.yaml)** — rewritten for the modern
   stack. Also adds `ffmpeg=7`: torchcodec links FFmpeg's shared libraries rather than
   vendoring them, and without it `import torchcodec` fails with
   `Could not load libtorchcodec`.

3. **[README.md](README.md)** — install instructions now cover all four repos and the
   required `--no-deps` for robomimic.

4. **[`diffusion_policy/model/common/lr_scheduler.py`](diffusion_policy/model/common/lr_scheduler.py)**
   — modern diffusers no longer re-exports `Union`/`Optional`/`Optimizer` from
   `diffusers.optimization`. Now imported from `typing` and `torch.optim` directly;
   `SchedulerType` and `TYPE_TO_SCHEDULER_FUNCTION` still come from diffusers.

5. **`robomimic/robomimic/utils/torch_utils.py:142`** (sibling repo, not this one) — the
   same stale `diffusers.optimization` import, fixed the same way. This one is hit at
   runtime during optimizer construction, not at import, so it surfaces only once
   training actually starts.

6. **`config/task/robocasa/*.yaml`** — renamed the `dataset_soup` values from
   `posttrain_*` to `target_*` (7 files). `DATASET_SOUP_REGISTRY` in robocasa 1.0.1
   defines `target_atomic_seen`, `target_composite_seen`, `target_composite_unseen` and
   `target50`; the configs still referenced the pre-rename `posttrain_*` names and
   raised `KeyError` at dataset instantiation. Each config's own filename already used
   the `target_*` form, and the `pretrain_*` configs were unaffected, which is what
   identifies this as an incomplete rename rather than a missing registry entry.

### Installing robomimic

Always use `--no-deps`:

```
pip install --no-deps -e ../robomimic
```

Without it, pip re-applies robomimic's pins and silently reintroduces the entire
conflict. If training suddenly fails with `HfFolder` or numpy assertion errors, check
whether robomimic was reinstalled.

## Verification

A short training run completes on the fixed environment:

```
python train.py --config-name=train_diffusion_transformer_bs192 \
  task=robocasa/target_atomic_seen \
  training.num_epochs=2 training.max_train_steps=3 training.max_val_steps=3 \
  training.val_every=1 training.sample_every=1 training.checkpoint_every=1 \
  training.rollout_every=null \
  dataloader.batch_size=8 val_dataloader.batch_size=8 \
  dataloader.num_workers=4 val_dataloader.num_workers=4 \
  logging.project=debug
```

This exercises dataset construction over all 18 lerobot datasets, the forward/backward
pass, diffusion sampling and checkpoint saving. Exit code 0, `train_loss ≈ 1.18-1.22`,
`train_action_mse_error ≈ 0.77`, and two checkpoints written under
`data/outputs/<date>/<time>_.../checkpoints/`.

Two things to know about this invocation:

- **Do not use `training.debug=True` with the `robocasa` task configs.** It force-sets
  `rollout_every=1` (workspace line ~223), and the task configs' `env_kwargs` block has
  no `env_name` or `split` — those are supplied by `eval_robocasa.py` at eval time, not
  during training. The result is `KeyError('env_name')` in `RobomimicImageRunner.__init__`
  at the end of the second epoch. The explicit overrides above give the same short run
  without enabling rollouts. (`train_diffusion_transformer_bs192.yaml` ships
  `rollout_every: null`, i.e. rollouts are off during normal training anyway.)
- `val_loss` is never logged: the validation block in the workspace (lines 332-347) is
  commented out upstream. Only `train_loss` and `train_action_mse_error` appear.

Also note `env_runner.dataset_path` in the task configs points at
`/mnt/amlfs-01/shared/robocasa_benchmark/...`, a cluster path that does not exist
locally. It is unused on the training path but will need fixing for evaluation.

## Known issues not related to the environment

- The `numpy 1.23 → 2.2.5` and `torch 2.0 → 2.7` jumps have only been validated as far
  as dataset construction. Numerical code paths in this repo and in robomimic have not
  been exhaustively exercised against numpy 2.x semantics.
- The `numpy 1.23 → 2.2.5` and `torch 2.0 → 2.7` jumps mean numerical code paths in this
  repo and in robomimic have not been exhaustively re-validated against numpy 2.x
  semantics. Watch for silent dtype/casting differences rather than hard failures.

## Datasets

Datasets are expected under
`<robocasa>/datasets/v1.0/<split>/<category>/<Task>/<date>/lerobot/`, which the entries
in `DATASET_SOUP_REGISTRY` point at as absolute paths. Fetch them with:

```
python ../robocasa/robocasa/scripts/download_datasets.py --split target --source human
```

Add `--dryrun` to see what would be fetched, and `--tasks <Task> [...]` to limit the set.
The full `target_atomic_seen` soup (18 tasks) is roughly 58 GB on disk.

Note that `DATASET_BASE_PATH` in `robocasa.macros` is `None` unless a private macro file
has been set up; it is only consulted for registry entries with relative paths, so the
default absolute-path entries work without it.
