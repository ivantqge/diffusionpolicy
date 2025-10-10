# Diffusion Policy with Push F Task

This repository extends the original [Diffusion Policy](https://github.com/real-stanford/diffusion_policy) with a new **PushF task** - learning to manipulate an F-shaped object instead of a T-shaped object.

## New Files Added for PushF Task

### Core Environment
```
diffusion_policy/env/pushf/
├── pushf_env.py                    
├── pushf_keypoints_env.py         
├── pushf_image_env.py            
└── __init__.py                    
```

### Training Infra
```
diffusion_policy/env_runner/
├── pushf_keypoints_runner.py       
└── pushf_image_runner.py           

diffusion_policy/dataset/
├── pushf_dataset.py               
└── pushf_image_dataset.py          
```

### Config Files
```
diffusion_policy/config/task/
├── pushf_lowdim.yaml              
└── pushf_image.yaml               

diffusion_policy/config/
└── train_diffusion_transformer_lowdim_pushf_workspace.yaml 
```

### Demo + Data Collection
```
demo_pushf.py                      
data/pushf_demo.zarr/              # Collected demonstration data
```

## Modified Files
```
diffusion_policy/env/pusht/pymunk_keypoint_manager.py
└── Added create_from_pushf_env() method for F-shape keypoint generation
```

## Additional Adjustments

Few adjustments were made when importing packages:
- removed import for imagecodecs==2022.8.8, installed with pip instead
- downgraded huggingface_hub to 0.19.4 to be compatible with diffusers=0.11.1 (due to cached_download being deprecated)

