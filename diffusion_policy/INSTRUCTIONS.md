# Running training and evaluation

*Last updated: 2026-09-17. Verified on 2× RTX 4070 Ti (12 GB each), 24 cores, 62 GB RAM.*

Setup and dependency background: [ENVIRONMENT.md](ENVIRONMENT.md).
What was changed in the source and why: [CODE_FIXES.md](CODE_FIXES.md).

All commands assume:

```bash
conda activate robocasa
cd /media/saks/disk8TB/robocasa/diffusion_policy
```

---

## 1. Debug run — fast, no rollouts (~3 min)

The default check after touching training code. Two epochs of three steps each, with
sampling and checkpointing exercised.

```bash
python train.py \
  --config-name=train_diffusion_transformer_bs192 \
  task=robocasa/target_atomic_seen \
  training.num_epochs=2 training.max_train_steps=3 \
  training.sample_every=1 training.checkpoint_every=1 \
  training.rollout_every=null \
  dataloader.batch_size=8 val_dataloader.batch_size=8 \
  dataloader.num_workers=4 val_dataloader.num_workers=2 \
  logging.mode=offline logging.project=debug
```

Expect exit code 0, `train_loss ≈ 1.2`, and checkpoints under
`data/outputs/<date>/<time>_.../checkpoints/`.

## 2. Debug run — with rollouts (~8 min)

Use this whenever you touch the env runner, policy inference, or anything eval-related.
It is the only debug that exercises the rollout path.

```bash
accelerate launch --multi_gpu --num_processes=2 train.py \
  --config-name=train_diffusion_transformer_bs192 \
  task=robocasa/target_atomic_seen \
  training.num_epochs=2 training.max_train_steps=3 \
  training.rollout_every=1 training.checkpoint_every=1 training.sample_every=1 \
  task.env_runner.n_test=2 task.env_runner.n_envs=1 task.env_runner.n_test_vis=0 \
  logging.mode=offline logging.project=debug
```

Rollouts are memory-critical on 12 GB cards — see §5 before raising `n_envs`.

## 3. Full training run

Everything is in the config, so there are no overrides:

```bash
nohup accelerate launch --multi_gpu --num_processes=2 train.py \
  --config-name=train_diffusion_transformer_bs192 \
  task=robocasa/target_atomic_seen \
  > train_$(date +%m%d_%H%M).log 2>&1 &
```

**Use a timestamped log filename, not a fixed one.** `>` truncates, so launching twice
against the same path makes the second run wipe the first run's log — while the first
run's file descriptor keeps writing at its old offset. The result looks exactly like your
training died when it is in fact still running. If you hit this, the shell log is
disposable: hydra writes its own `train.log` and `logs.json.txt` per run directory.

Before launching, check nothing is already using the GPUs — a second concurrent run will
fail with an OOM that looks unrelated to the duplicate:

```bash
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
pgrep -af "train.py" | head
```

This picks up:

| setting | value | note |
|---|---|---|
| `dataloader.batch_size` | 32 | **global**, split across GPUs → 16/GPU |
| `gradient_accumulate_every` | 6 | 32 × 6 = 192 effective, which the lr was tuned for |
| `num_epochs` / `max_train_steps` | 1000 / 500 | 500 steps per epoch |
| `rollout_every` | 50 | 20 rollout events over the run |
| `env_runner.n_test` / `n_envs` | 50 / 2 | 50 episodes, 2 at a time |
| `env_runner.env_name` | `PickPlaceCounterToCabinet` | one task only; see §6 |
| `logging.mode` | `online` | logs to wandb project `diffusion_policy_bs192` |

Budget roughly **~30 min per rollout event** (50 episodes ÷ 2 envs = 25 chunks of ~70 s),
so ~10 h of the run is rollouts. Set `training.rollout_every=100` to halve that.

## 4. Evaluation

```bash
python eval_robocasa.py \
  -c data/outputs/<date>/<time>_.../checkpoints/latest.ckpt \
  -t PickPlaceCounterToCabinet CloseFridge TurnOnSinkFaucet \
  -s target \
  -n 50 \
  -e 2
```

- `-t/--task_set` takes multiple task names (required).
- `-s/--split` is required — `target` or `pretrain`.
- `-n/--num_rollouts` defaults to 50.
- `-e/--num_envs` **defaults to 5, which will OOM on a 12 GB card.** Pass `-e 2` or lower.

`eval_robocasa.py` overwrites `env_kwargs` per task and sets `max_steps` from
`get_task_horizon`, so it sweeps tasks correctly — unlike in-training rollouts, which are
pinned to a single task. **Reported benchmark numbers should come from here.**

Then aggregate:

```bash
python diffusion_policy/scripts/get_eval_stats.py --dir <outputs-dir>
```

---

## 5. GPU memory: sizing `n_envs`

Each rollout env is a separate process running its own offscreen MuJoCo renderer on the
training GPU. Measured cost: **~1.6 GB VRAM per env** at 3×256×256 cameras, against
~4.7–7.6 GB of rank-0 training state plus ~0.8 GB of NCCL buffers on a 11.6 GB usable card.

Measured on this machine, during real 2-GPU training:

| `n_envs` | result |
|---|---|
| 8 | fails — `mujoco.FatalError: Offscreen framebuffer is not complete, error 0x8cdd` |
| 4 | fails — `torch.OutOfMemoryError` during the rollout |
| 2 | passed


`n_envs` is purely a parallelism knob and has no effect on `n_test`, so it does not change
the statistical quality of the result.

Fragmentation mitigation worth trying before lowering `n_envs`:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

## 6. How much to trust in-training rollout numbers

In-training rollouts evaluate **one task** (`PickPlaceCounterToCabinet`) out of the 18 in
the `target_atomic_seen` soup. `RobomimicImageRunner` handles a single `env_name` per
runner; this is a training-progress signal, not a benchmark number.

At `n_test: 50`, the 95% Wilson interval is ±11–13 points:

| observed | 95% CI |
|---|---|
| 20% | [11.2%, 33.0%] |
| 50% | [36.6%, 63.4%] |

So 50 episodes distinguishes 20% from 60%, but not 45% from 55%. Note this also means
`checkpoint.topk` — which ranks by `test_mean_score` with `k: 20` — is selecting among
checkpoints whose scores carry ±13 points of noise. Treat "best checkpoint" accordingly.

---

## Gotchas

**Make HF hermetic for long runs.** Startup loads CLIP (`openai/clip-vit-large-patch14`,
for the `lang_emb` text embeddings) and makes live HF API calls even when cached, so a
network blip can kill a launch. Once the cache is warm:

```bash
export HF_HUB_OFFLINE=1
```

The `UNEXPECTED` key report from that load is normal — it is CLIP's vision tower, which a
text-only `CLIPTextModelWithProjection` has no slots for. The policy's own vision encoder
is a separate `ResNet18ConvFiLM` per camera. `MISSING` keys would be a real problem;
`UNEXPECTED` ones are not.

**Reinstalling robomimic reverts a required fix.** Always use
`pip install --no-deps -e ../robomimic`, and re-apply the `diffusers.optimization` import
fix at `robomimic/robomimic/utils/torch_utils.py:142` if you re-clone. See
[CODE_FIXES.md](CODE_FIXES.md) §2.

**`max_steps` must track `env_name`.** `env_runner.max_steps` is 750 to match
`PickPlaceCounterToCabinet`. If you change `env_kwargs.env_name`, update `max_steps` to
that task's `get_task_horizon` or episodes will be truncated early.
