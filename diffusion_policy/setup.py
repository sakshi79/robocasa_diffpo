from setuptools import setup, find_namespace_packages

# NOTE: these bounds are chosen to be co-installable with robocasa 1.0.1 and
# lerobot 0.3.3, which the RoboCasa365 task configs depend on at train time.
# In particular: robocasa pins numpy==2.2.5 and numba==0.61.2, and lerobot
# requires torch>=2.2.1,<2.8, torchvision>=0.21,<0.23 and diffusers>=0.27.2.
# Do not re-pin torch/numpy/diffusers to the upstream Diffusion Policy versions
# (torch 2.0.1 / numpy 1.23.2 / diffusers 0.11.1) -- they are mutually
# exclusive with robocasa. See ENVIRONMENT.md.
setup(
  name = 'diffusion_policy',
  # NOTE: the source tree has no __init__.py files, so find_packages() returns [] and
  # `pip install -e .` installs an importable-nothing finder. Everything then only works
  # when cwd happens to be the repo root. find_namespace_packages fixes that.
  packages = find_namespace_packages(include=['diffusion_policy*']),
  python_requires = '>=3.10',
  install_requires = [
    # config / experiment plumbing
    'hydra-core>=1.3.2',
    'omegaconf>=2.3.0',
    'wandb>=0.20.0',
    'dill>=0.3.5',
    'tqdm',
    'termcolor',
    'tensorboard',
    'tensorboardx',
    'psutil',
    'click',

    # numerics -- numpy/numba are pinned by robocasa; keep them consistent
    'numpy==2.2.5',
    'numba==0.61.2',
    'scipy>=1.15',
    'threadpoolctl>=3.1.0',

    # torch stack -- bounded by lerobot 0.3.3
    'torch>=2.2.1,<2.8.0',
    'torchvision>=0.21.0,<0.23.0',
    'torchcodec>=0.2.1,<0.6.0',

    # diffusion / model deps
    'diffusers>=0.27.2',
    'huggingface_hub>=0.34.2',
    'accelerate>=1.0.0',
    'einops>=0.8.0',

    # data / io
    'zarr>=2.12.0',
    'numcodecs>=0.10.2',
    'h5py>=3.7.0',
    'imageio[ffmpeg]>=2.34.0',
    'av>=14.2.0',
    'matplotlib>=3.6.0',
    'numpydantic>=1.10.0',
  ],
  extras_require = {
    # only needed for ray_train_multirun.py / ray_exec.py
    'ray': ['ray[default,tune]>=2.9.0'],
    # only needed for the pusht / lowdim toy envs, not for RoboCasa365
    'pusht': ['pygame>=2.1.2', 'pymunk>=6.2.1', 'shapely>=1.8.4', 'scikit-image', 'scikit-video'],
  },
)
