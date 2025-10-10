# Diffusion Policy with Push F Task

Cloned repo from https://github.com/real-stanford/diffusion_policy/ to add a Push F task on top of the pre-existing Push T task. 

Few adjustments were made when importing packages:
- removed import for imagecodecs==2022.8.8, installed with pip instead
- downgraded huggingface_hub to 0.19.4 to be compatible with diffusers=0.11.1 (due to cached_download being deprecated)

