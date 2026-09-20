# Code fixes: making the RoboCasa365 train and rollout paths run

*Last updated: 2026-09-20*

Companion to [ENVIRONMENT.md](ENVIRONMENT.md). That document covers the dependency
conflict between the sibling repos and how the environment was rebuilt. This one covers
the **source changes** that were needed on top of it, and why each was necessary.

Nearly all of these are version rot: the repo was written against gym ~0.21, diffusers
0.11 and torch 2.0, and is now being run against gym 0.26, diffusers 0.40 and torch 2.7.
The failures surfaced one at a time because each one is hit at a different point in the
run — import, optimizer construction, dataset build, epoch 1, rollout reset.

Files touched in this repo: 15. Plus one file in the sibling `robomimic` repo.

The two most consequential entries are **7** and **8** — both silent, both affecting
training dynamics rather than raising an error. Read those first if you are auditing a
run's hyperparameters.

---

## 1. Packaging: `pip install -e .` installed nothing importable

**File:** [setup.py](setup.py)

`diffusion_policy/` contains no `__init__.py` files, so `find_packages()` returned `[]`.
The editable install therefore shipped a finder with `MAPPING = {}`:

```python
>>> import diffusion_policy; diffusion_policy.__file__
None                      # implicit namespace package, resolved from cwd
$ cd /tmp && python -c "import diffusion_policy"
ModuleNotFoundError: No module named 'diffusion_policy'
```

Every command had only ever worked because it was run from the repo root, where Python's
implicit namespace package resolution picks up `./diffusion_policy`. Switched to
`find_namespace_packages(include=['diffusion_policy*'])` (59 packages).

Also added `install_requires`, which was previously absent entirely — the original
`setup.py` was just `setup(name=..., packages=...)`. That is the direct reason a fresh
environment ends up with no `hydra`, no `omegaconf`, no `zarr`.

## 2. `diffusers.optimization` no longer re-exports typing names

**Files:** [diffusion_policy/model/common/lr_scheduler.py](diffusion_policy/model/common/lr_scheduler.py),
`robomimic/robomimic/utils/torch_utils.py:142` (sibling repo)

Both files did:

```python
from diffusers.optimization import (
    Union, SchedulerType, Optional, Optimizer, TYPE_TO_SCHEDULER_FUNCTION
)
```

diffusers ≥0.27 stopped re-exporting `Union`, `Optional` and `Optimizer` from that
module. `SchedulerType` and `TYPE_TO_SCHEDULER_FUNCTION` still live there. Now imported
from `typing` and `torch.optim` directly.

The robomimic copy is inside a function body, so it is hit during
`_create_optimizers()` rather than at import — it only appears once training actually
starts, after the dataset has loaded.

> This change is in a different git repo and is **not** captured by this repo's history.
> Re-cloning or reinstalling robomimic will reintroduce it.

## 3. Multi-GPU corrupted `logs.json.txt`

**File:** [diffusion_policy/workspace/train_diffusion_transformer_hybrid_workspace.py](diffusion_policy/workspace/train_diffusion_transformer_hybrid_workspace.py)

`json_logger.log(step_log)` was called on every rank, unguarded, while the neighbouring
wandb and checkpoint calls were all wrapped in `if accelerator.is_main_process:`. Under
`accelerate launch --num_processes=2` both ranks wrote to the same file and records
interleaved mid-token:

```
{"train_loss": 1.2667, "global_step": 0, ...}{"train_loss": 1.2100, "global_step": 1, ...}
{"train_loss": 1.2212, ..., "{"train_loss": 1.2281, "global_ste{"train_loss": 1.1561, "global_st...
```

Six training steps produced four unparseable lines (`json.loads` → `Extra data: line 1
column 95`). Both call sites are now rank-guarded. Single-GPU runs were never affected.

## 4. Rollouts: `AsyncVectorEnv` could not be constructed

**File:** [diffusion_policy/env_runner/robomimic_image_runner.py](diffusion_policy/env_runner/robomimic_image_runner.py)

Constructed with the default `shared_memory=True`. robocasa envs expose an
`OrderedDict` observation space, which gym classifies as a custom space and refuses to
back with shared memory:

```
ValueError: Using `shared_memory=True` in `AsyncVectorEnv` is incompatible with
non-standard Gym observation spaces
```

Now passes `shared_memory=False`; observations are pickled over the worker pipes.
Negligible at small `n_envs`.

## 5. Rollouts: vendored `AsyncVectorEnv` vs the gym 0.26 API

**File:** [diffusion_policy/gym_util/async_vector_env.py](diffusion_policy/gym_util/async_vector_env.py)

This file is a vendored copy of gym's `AsyncVectorEnv` from around gym 0.21, and its
header notes that `call`/`set_attr` were back-ported from 0.26. Two things were missed.

**Signatures.** gym ≥0.26's `VectorEnv.reset()` forwards `seed`/`options` to both hooks:

```python
self.reset_async(seed=seed, options=options)
return self.reset_wait(seed=seed, options=options)
```

The overrides were `reset_async(self)` and `reset_wait(self, timeout=None)`, so a bare
`env.reset()` raised `TypeError: reset_async() got an unexpected keyword argument
'seed'`. Both now accept and ignore them — seeding is handled separately through
`env_init_fn_dills`, and callers use a bare `reset()`, so both are always `None`.

**Argument order.** gym ≥0.26 takes `concatenate(space, items, out)`; older releases took
`concatenate(items, out, space)`. Two call sites still used the old order and failed with
a `functools.singledispatch` `ValueError` (it dispatches on `args[0]`, which was a
`tuple`). Note `create_empty_array` in the same file was *already* in the new order,
which is what makes this a partial-update bug rather than a wholesale version mismatch.

This path is only reachable with `shared_memory=False`, so fix 4 is what exposed it.

## 6. `AsyncVectorEnv` was never explicitly closed

**File:** [diffusion_policy/env_runner/robomimic_image_runner.py](diffusion_policy/env_runner/robomimic_image_runner.py)

`close()` began with:

```python
def close(self):
    if not isinstance(self.env, SyncVectorEnv):
        return          # <- AsyncVectorEnv: no-op
```

so the worker processes were never asked to shut down and cleanup fell entirely to
`VectorEnv.__del__`. In the training loop that usually still works, because
`del env_runner` drops the last reference and `__del__` runs immediately while the
workers are alive. In a script that exits straight afterwards, `__del__` instead runs
during interpreter shutdown, when the pipes are already gone:

```
BrokenPipeError: [Errno 32] Broken pipe
UserWarning: resource_tracker: There appear to be 6 leaked semaphore objects
```

`close()` now closes the async env explicitly, so workers are reaped deterministically.

---

## Config fixes

### `dataset_soup` names were pre-rename

**Files:** `diffusion_policy/config/task/robocasa/*.yaml` (7 files)

`DATASET_SOUP_REGISTRY` in robocasa 1.0.1 defines `target_atomic_seen`,
`target_composite_seen`, `target_composite_unseen` and `target50`. The configs still
referenced `posttrain_*`, raising `KeyError` at dataset instantiation. Each config's own
filename already used the `target_*` form and the `pretrain_*` configs were unaffected,
which identifies it as an incomplete rename rather than a missing registry entry.

### `checkpoint.topk` was inert — but the key was not the reason

**File:** [diffusion_policy/config/train_diffusion_transformer_bs192.yaml](diffusion_policy/config/train_diffusion_transformer_bs192.yaml)

Top-k checkpoint selection did nothing: every saved checkpoint was named
`test_mean_score=-1.000`, the default `TopKCheckpointManager` assigns when the monitor
key is absent, so all candidates tied.

The cause is simply that `rollout_every` was `null`. `test_mean_score` comes from the
rollout, so with rollouts off it is never present. Enabling `rollout_every` fixes it.

**`monitor_key: test_mean_score` is correct and must not be "fixed" to
`test/mean_score`.** `RobomimicImageRunner` does emit `test/mean_score` with a slash, but
the workspace sanitizes metric names before consulting the manager:

```python
for key, value in step_log.items():
    new_key = key.replace('/', '_')       # test/mean_score -> test_mean_score
    metric_dict[new_key] = value
topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)
```

Changing the config to the slashed form makes the key miss again and silently reverts
every checkpoint to `-1.000`. This was tried during this work and reverted.

### `env_kwargs` was missing `env_name` and `split`

**File:** [diffusion_policy/config/task/robocasa/target_atomic_seen.yaml](diffusion_policy/config/task/robocasa/target_atomic_seen.yaml)

`create_env()` needs `env_name` and `split`, and the task config supplied neither, so any
rollout died with `KeyError('env_name')` in `RobomimicImageRunner.__init__`. They were
absent because `eval_robocasa.py` *overwrites the entire `env_kwargs` block* with
`{split, seed, env_name}` at eval time — so the path was never exercised from the config.

A consequence worth knowing: the rest of that block (`layout_and_style_ids`,
`camera_names`, `obj_instance_split`, …) is **vestigial**. `create_env()` only consumes
`env_name`, `split` and `seed`.

Also note `env_runner.dataset_path` still points at
`/mnt/amlfs-01/shared/robocasa_benchmark/...`, a cluster path that does not exist
locally. It is unused on both the train and rollout paths, but it is dead config.

### `max_steps` did not match the task horizon

`max_steps: 600`, but `get_task_horizon('PickPlaceCounterToCabinet')` is `750`, so every
episode would have been silently truncated 150 steps early. Now 750. **This must be kept
in sync with `env_kwargs.env_name`** — `eval_robocasa.py` sets it per task, the training
config cannot.

## 7. `split_batches=True` was silently ignored by accelerate

**File:** [diffusion_policy/workspace/train_diffusion_transformer_hybrid_workspace.py](diffusion_policy/workspace/train_diffusion_transformer_hybrid_workspace.py)

The single highest-impact bug found, and the hardest to see: it is silent, and it
corrupts both the batch size and the LR schedule at once.

The workspace built `Accelerator(..., split_batches=True)`. In accelerate ≥1.x that bare
kwarg is **superseded by `DataLoaderConfiguration` and ignored without any warning** —
the parameter is still accepted, so nothing complains:

```python
acc = Accelerator(split_batches=True)
acc.split_batches      # -> False        (accelerate 1.15.0)
```

Two consequences, both invisible in the logs:

1. **Effective batch was double the intended value.** With `split_batches=False`,
   `dataloader.batch_size` is per-process, so `batch_size: 32` on 2 GPUs meant 64 per
   micro-batch → `× gradient_accumulate_every: 6` = **384 effective, not 192**, against
   a `learning_rate: 1e-4` tuned for 192.

2. **The cosine LR schedule ran at double speed.** `AcceleratedScheduler.step()` advances
   the underlying scheduler `num_processes` times per call when `split_batches` is false.
   The scheduler therefore consumed 166,666 steps over a run calibrated to
   `num_training_steps = 83,333`. Measured directly from the logs:
   `global_step / scheduler_step = 3.00` where the code assumes 6.

   The LR reaches 0 at **epoch ~500 of 1000**, after which `progress > 1` and
   `cos(π·progress)` turns back upward — the LR climbs toward peak instead of annealing.

The fix is to pass it the way accelerate now requires:

```python
from accelerate.utils import DataLoaderConfiguration
accelerator = Accelerator(
    log_with='wandb',
    kwargs_handlers=[ddp_kwargs, timeout_handler],
    dataloader_config=DataLoaderConfiguration(split_batches=True),
)
```

Verified after the change: `acc.split_batches` and `AcceleratedScheduler.split_batches`
are both `True`, and the observed rate becomes one scheduler step per 6 global steps.

**Do not "simplify" this back to the bare kwarg.** It will appear to work and silently
double your batch size and halve your LR schedule length.

## 8. `last_epoch` was in the wrong units, breaking resume

**File:** [diffusion_policy/workspace/train_diffusion_transformer_hybrid_workspace.py](diffusion_policy/workspace/train_diffusion_transformer_hybrid_workspace.py)

`get_scheduler(..., last_epoch=self.global_step-1)` — but `last_epoch` counts *scheduler*
steps, and the scheduler is stepped once per optimizer step, i.e. once every
`gradient_accumulate_every` batches. Passing `global_step` directly places a resumed run
`accum`× too far along the cosine.

This is latent on a fresh run, where `global_step == 0` gives the correct `-1`. It only
fires on the first resume — which is exactly when it happened here: a run resumed at
epoch 300 jumped from LR 3.5e-5 back to 9.2e-5 and began *rising*, because it landed past
the end of the schedule in the wrapped region. Task success fell 0.300 → 0.120 and took
~100 epochs to recover to 0.280.

Now `last_epoch=(self.global_step // cfg.training.gradient_accumulate_every) - 1`.

Note the same line exists in the seven other `train_*_workspace.py` files and is still
wrong there; only the transformer hybrid workspace is used by RoboCasa365.

---

## Things that are not bugs

Worth recording so they are not "fixed" later:

- **`training.device: "cuda:0"` is never read.** Device placement comes from
  `self.model.device` via accelerate. It does not pin training to GPU 0.
- **`val_every`, `max_val_steps` and the whole `val_dataloader` block are dead.** The
  validation block in the workspace is commented out upstream. `val_loss` is never
  logged. The val dataloader is still built and `accelerator.prepare`d, so its workers
  spawn and do nothing.
- **`dataloader.batch_size` is the *global* batch**, not per-GPU — but only because of
  fix 7 below. With `num_processes=2`, `batch_size: 32` is 16 per GPU.
- **`training.debug=True` cannot be used with the robocasa task configs.** It force-sets
  `rollout_every = 1`. Use explicit overrides for a short run instead; see
  [ENVIRONMENT.md](ENVIRONMENT.md).

## Verified

- Single-GPU short run: exit 0, `train_loss ≈ 1.18`, checkpoints written.
- `accelerate launch --multi_gpu --num_processes=2`: exit 0, both ranks train,
  `logs.json.txt` parses.
- One rollout episode, `PickPlaceCounterToCabinet`, `n_envs=1`, 750 steps: 63.2 s,
  plus 24.1 s of env construction per rollout event. Emits `test/mean_score`.
- **`n_envs: 8` fails on 12 GB cards** with
  `mujoco.FatalError: Offscreen framebuffer is not complete, error 0x8cdd` — GL
  framebuffer allocation failing once 8 offscreen renderers share the GPU with the
  resident model (sampled peak 11862 MiB of 12282 MiB). Size `n_envs` to the card.
- **A 2-day multi-GPU run** reached epoch 300 with task success climbing
  0.00 → 0.02 → 0.14 → 0.20 → 0.30 at epochs 50/100/150/200/300, confirming the rollout,
  checkpoint and wandb paths all work end to end under load. That run was on the
  pre-fix-7 configuration (effective batch 384, half-length cosine).
- **After fixes 7 and 8**, a short 2-GPU run gives one scheduler step per 6 global steps
  (was 3), which is what `num_training_steps` assumes.
