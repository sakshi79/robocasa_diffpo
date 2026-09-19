# Diffusion Policy

This is the Diffusion Policy fork repo for running RoboCasa benchmark experiments.
This fork is based on the original Diffusion Policy code, hosted at [https://github.com/real-stanford/diffusion_policy](https://github.com/real-stanford/diffusion_policy).

## Recommended system specs
For training we recommend a GPU with at least 24 Gb of memory, but 48 Gb+ is prefered.
For inference we recommend a GPU with at least 8 Gb of memory.

## Installation

This repo shares an environment with `robocasa`, `robosuite` and `robomimic`. Install
them together — installing `robomimic` normally will downgrade torch/numpy/diffusers to
versions that `robocasa` cannot run with (see [ENVIRONMENT.md](ENVIRONMENT.md)).

```
git clone https://github.com/robocasa-benchmark/diffusion_policy
cd diffusion_policy

conda env create -f conda_environment.yaml
conda activate robocasa

pip install -e .
pip install --no-deps -e ../robomimic     # --no-deps is required, see note below
pip install -e ../robosuite -e ../robocasa
```

> **Note on `--no-deps`:** `robomimic/setup.py` pins `numpy==1.23.2`, `torch==2.0.1`
> and `diffusers==0.11.1`. Those pins are incompatible with `robocasa` (which asserts
> `numpy==2.2.5`) and with `lerobot` (which needs `torch>=2.2.1` and
> `diffusers>=0.27.2`). Installing robomimic with `--no-deps` keeps its code while
> leaving the environment's versions intact.

Verify the install:
```
python -c "import torch, torchcodec, robocasa, lerobot; print(torch.__version__, torch.cuda.is_available())"
```

## Key files
- Training: [train.py](https://github.com/robocasa-benchmark/diffusion_policy/blob/main/train.py)
- Evaluation: [eval_robocasa.py](https://github.com/robocasa-benchmark/diffusion_policy/blob/main/eval_robocasa.py)

## Experiment workflow
```
# train model
python train.py \
--config-name=train_diffusion_transformer_bs192 \
task=robocasa/<dataset-soup>

# Evaluate model
python eval_robocasa.py \
--checkpoint <checkpoint-path> \
--task_set <task-set> \
--split <split>

# Report evaluation results
python diffusion_policy/scripts/get_eval_stats.py \
--dir <outputs-dir>
```
