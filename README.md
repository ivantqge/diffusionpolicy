# Diffusion Policy with Push F Task

This repository extends the original [Diffusion Policy](https://github.com/real-stanford/diffusion_policy) with a new **PushF task** - learning to manipulate an F-shaped object instead of a T-shaped object.

## New Files Added for PushF Task

### Core Environment
```
diffusion_policy/env/pushf/
├── pushf_env.py                    # Base PushF environment with F-shaped object physics
├── pushf_keypoints_env.py          # Keypoint-based observations for PushF
├── pushf_image_env.py              # Image-based observations for PushF  
└── __init__.py                     # Package initialization
```

### Training Infra
```
diffusion_policy/env_runner/
├── pushf_keypoints_runner.py       # Environment runner for keypoint-based PushF training
└── pushf_image_runner.py           # Environment runner for image-based PushF training

diffusion_policy/dataset/
├── pushf_dataset.py                # Dataset loader for PushF keypoint demonstrations
└── pushf_image_dataset.py          # Dataset loader for PushF image demonstrations
```

### Config Files
```
diffusion_policy/config/task/
├── pushf_lowdim.yaml               # Task config for keypoint-based PushF
└── pushf_image.yaml                # Task config for image-based PushF

diffusion_policy/config/
└── train_diffusion_transformer_lowdim_pushf_workspace.yaml  # Training config for PushF
```

### Demo + Data Collection
```
demo_pushf.py                       # Interactive demo collection for PushF task
data/pushf_demo.zarr/              # Collected demonstration data (22 episodes)
```

## Modified Files
```
diffusion_policy/env/pusht/pymunk_keypoint_manager.py
└── Added create_from_pushf_env() method for F-shape keypoint generation

Few adjustments were made when importing packages:
- removed import for imagecodecs==2022.8.8, installed with pip instead
- downgraded huggingface_hub to 0.19.4 to be compatible with diffusers=0.11.1 (due to cached_download being deprecated)

