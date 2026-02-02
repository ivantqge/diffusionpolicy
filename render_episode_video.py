"""
Render Episode Video

Creates a video from a train or validation episode showing the trajectory.
Can optionally overlay predicted progress from a progress predictor model.
"""

import pickle
import click
import numpy as np
import torch
try:
    import imageio
    HAS_IMAGEIO = True
except ImportError:
    try:
        import cv2
        HAS_CV2 = True
        HAS_IMAGEIO = False
    except ImportError:
        HAS_IMAGEIO = False
        HAS_CV2 = False
        print("Warning: Neither imageio nor cv2 found. Install one to create videos.")

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


def overlay_progress_on_frame(img, progress, true_progress=None, step_idx=None, total_steps=None):
    """
    Overlay progress prediction on a frame.
    
    Args:
        img: Image array (HWC, uint8, RGB)
        progress: Predicted progress [0, 1]
        true_progress: True progress [0, 1] (optional, for comparison)
        step_idx: Current step index (optional)
        total_steps: Total steps in episode (optional)
    """
    if not HAS_CV2:
        return img
    
    # Convert RGB to BGR for OpenCV
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    h, w = img_bgr.shape[:2]
    
    # Progress bar parameters
    bar_height = 2
    bar_width = 5#int(w * 0.6)
    bar_x = 5#int(w * 0.2)
    bar_y = 5
    bar_thickness = 1
    
    # Text overlay
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.2
    text_thickness = 1
    
    # Predicted progress text
    pred_text = f'Pred: {progress:.3f}'
    cv2.putText(img_bgr, pred_text, (bar_x, bar_y + bar_height), font, font_scale, 
               (0, 255, 0), text_thickness, cv2.LINE_AA)
    
    # True progress text if provided
    if true_progress is not None:
        true_text = f'True: {true_progress:.3f}'
        cv2.putText(img_bgr, true_text, (bar_x, bar_y + bar_height + 8), font, font_scale, 
                   (0, 0, 255), text_thickness, cv2.LINE_AA)
    
    
    # Convert back to RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb


def create_video_from_episode(episode, out_path, fps=10, render_size=None, 
                              model=None, device='cpu', start_img=None):
    """
    Create a video from an episode using stored images.
    
    Args:
        episode: List of episode steps, each with 'img' key
        out_path: Output video path (.mp4)
        fps: Frames per second
        render_size: Optional resize (width, height). If None, uses original size.
        model: Optional progress predictor model for overlaying predictions
        device: Device for model inference
        start_img: Start image for Siamese model (if needed)
    """
    if len(episode) == 0:
        print("Error: Episode is empty")
        return
    
    # Get start image for Siamese model if needed
    if model is not None and model.use_siamese and start_img is None:
        if len(episode) > 0 and 'img' in episode[0]:
            start_img = episode[0]['img']
    
    # Collect all frames
    frames = []
    for step_idx, step in enumerate(episode):
        if 'img' not in step:
            continue
        
        img = step['img']
        
        # Ensure image is in correct format (HWC, uint8, RGB)
        if img.dtype != np.uint8:
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)
        
        # Predict progress if model is provided
        predicted_progress = None
        if model is not None:
            try:
                from infer_progress import predict, extract_action_chunk
                
                # Get current state
                state = step.get('state', None)
                if state is not None:
                    # Extract action chunk if needed
                    action_chunk = None
                    if model.action_chunk_horizon > 0:
                        action_chunk = extract_action_chunk(episode, step_idx, 
                                                          model.action_chunk_horizon, 
                                                          model.action_dim)
                    
                    # Predict progress
                    predicted_progress = predict(model, img, state, device=device, 
                                               action_chunk=action_chunk, 
                                               start_img=start_img)
            except Exception as e:
                print(f"Warning: Could not predict progress for step {step_idx}: {e}")
                predicted_progress = None
        
        # Get true progress if available
        true_progress = step.get('normalized_timestep', None)
        
        # Overlay progress on frame
        if predicted_progress is not None or true_progress is not None:
            img = overlay_progress_on_frame(img, predicted_progress or 0.0, 
                                          true_progress=true_progress,
                                          step_idx=step_idx, 
                                          total_steps=len(episode))
        
        # Resize if needed
        if render_size is not None and HAS_CV2:
            h, w = render_size
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
        
        frames.append(img)
    
    if len(frames) == 0:
        print("Error: No valid frames found in episode")
        return
    
    print(f"Creating video with {len(frames)} frames at {fps} fps...")
    
    # Write video
    if HAS_IMAGEIO:
        imageio.mimwrite(out_path, frames, fps=fps, codec='libx264', quality=8)
    elif HAS_CV2:
        # Use OpenCV VideoWriter
        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
        for frame in frames:
            # OpenCV expects BGR, but our frames are RGB, so convert
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        out.release()
    else:
        print("Error: No video writing library available")
        return
    
    print(f"Video saved to: {out_path}")


def create_video_from_episode_rendered(episode, env, out_path, fps=10):
    """
    Create a video by rendering each step of the episode using the environment.
    This ensures consistent rendering even if stored images differ.
    
    Args:
        episode: List of episode steps with 'state' key
        env: PushT environment instance
        out_path: Output video path (.mp4)
        fps: Frames per second
    """
    if len(episode) == 0:
        print("Error: Episode is empty")
        return
    
    frames = []
    for step_idx, step in enumerate(episode):
        if 'state' not in step:
            continue
        
        # Set environment state
        state = step['state']  # [agent_x, agent_y, block_x, block_y, block_angle]
        env._set_state(state)
        
        # Render frame
        img = env.render(mode='rgb_array')
        
        # Ensure correct format
        if img.dtype != np.uint8:
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)
        
        frames.append(img)
    
    if len(frames) == 0:
        print("Error: No valid frames found in episode")
        return
    
    print(f"Creating video with {len(frames)} frames at {fps} fps...")
    
    # Write video
    if HAS_IMAGEIO:
        imageio.mimwrite(out_path, frames, fps=fps, codec='libx264', quality=8)
    elif HAS_CV2:
        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
        for frame in frames:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        out.release()
    else:
        print("Error: No video writing library available")
        return
    
    print(f"Video saved to: {out_path}")


@click.command()
@click.option('--dataset_path', default='pusht_episodes.pkl', help='Path to episodes .pkl file')
@click.option('--train_ratio', default=0.8, type=float, help='Train/val split ratio')
@click.option('--split', type=click.Choice(['train', 'val', 'both'], case_sensitive=False), 
              default='train', help='Which split to render')
@click.option('--episode_idx', type=int, default=0, help='Episode index (0-indexed)')
@click.option('--fps', type=int, default=10, help='Frames per second for video')
@click.option('--render_size', type=int, default=None, help='Resize frames to this size (square)')
@click.option('--use_stored_images', is_flag=True, default=True, 
              help='Use stored images from episode (faster) vs render from state')
@click.option('--model_path', default=None, help='Path to progress predictor model (optional, for overlaying predictions)')
@click.option('--mode', default='categorical', help='Model mode (categorical or regression)')
@click.option('--action_chunk_horizon', default=8, type=int, help='Action chunk horizon for model')
@click.option('--device', default='auto', help='Device for model inference (auto, cpu, cuda)')
@click.option('--out_path', default=None, help='Output video path (auto-generated if not provided)')
def main(dataset_path, train_ratio, split, episode_idx, fps, render_size, use_stored_images, 
         model_path, mode, action_chunk_horizon, device, out_path):
    """
    Create a video from a train or validation episode.
    """
    # Load model if provided
    model = None
    if model_path is not None:
        print(f"Loading progress predictor model from {model_path}...")
        try:
            from infer_progress import load_model
            if device == 'auto':
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model = load_model(
                model_path=model_path,
                mode=mode,
                action_chunk_horizon=action_chunk_horizon,
                action_dim=2
            )
            model = model.to(device)
            model.eval()
            print(f"Model loaded on {device}")
        except Exception as e:
            print(f"Warning: Could not load model: {e}")
            print("Continuing without progress overlay...")
            model = None
    
    print(f"Loading dataset from {dataset_path}...")
    with open(dataset_path, 'rb') as f:
        episodes = pickle.load(f)
    
    # Split into train/val
    split_idx = int(train_ratio * len(episodes))
    train_episodes = episodes[:split_idx]
    val_episodes = episodes[split_idx:]
    
    print(f"Train episodes: {len(train_episodes)}, Val episodes: {len(val_episodes)}")
    
    # Select episode
    if split.lower() == 'train':
        if episode_idx >= len(train_episodes):
            print(f"Error: Episode index {episode_idx} out of range (max: {len(train_episodes)-1})")
            return
        episode = train_episodes[episode_idx]
        split_name = 'train'
    elif split.lower() == 'val':
        if episode_idx >= len(val_episodes):
            print(f"Error: Episode index {episode_idx} out of range (max: {len(val_episodes)-1})")
            return
        episode = val_episodes[episode_idx]
        split_name = 'val'
    else:
        # 'both' - render first train and first val
        if len(train_episodes) > 0 and len(val_episodes) > 0:
            print("Rendering both train and val episodes...")
            # Render train
            episode = train_episodes[0]
            split_name = 'train'
            out_path_train = out_path or f'episode_video_train_ep0.mp4'
            start_img_train = episode[0]['img'] if len(episode) > 0 and 'img' in episode[0] else None
            if use_stored_images:
                create_video_from_episode(episode, out_path_train, fps=fps, render_size=render_size,
                                         model=model, device=device, start_img=start_img_train)
            else:
                from diffusion_policy.env.pusht.pusht_keypoints_env import PushTKeypointsEnv
                kp_kwargs = PushTKeypointsEnv.genenerate_keypoint_manager_params()
                env = PushTKeypointsEnv(render_size=render_size or 96, render_action=False, **kp_kwargs)
                create_video_from_episode_rendered(episode, env, out_path_train, fps=fps)
            
            # Render val
            episode = val_episodes[0]
            split_name = 'val'
            out_path_val = out_path or f'episode_video_val_ep0.mp4'
            start_img_val = episode[0]['img'] if len(episode) > 0 and 'img' in episode[0] else None
            if use_stored_images:
                create_video_from_episode(episode, out_path_val, fps=fps, render_size=render_size,
                                         model=model, device=device, start_img=start_img_val)
            else:
                create_video_from_episode_rendered(episode, env, out_path_val, fps=fps)
            return
        else:
            print("Error: Cannot render both - missing train or val episodes")
            return
    
    print(f"Rendering {split_name} episode {episode_idx} ({len(episode)} steps)...")
    
    # Generate output path if not provided
    if out_path is None:
        model_suffix = '_with_progress' if model is not None else ''
        out_path = f'episode_video_{split_name}_ep{episode_idx}{model_suffix}.mp4'
    
    # Get start image for Siamese model if needed
    start_img = None
    if model is not None and model.use_siamese:
        if len(episode) > 0 and 'img' in episode[0]:
            start_img = episode[0]['img']
    
    # Create video
    if use_stored_images:
        create_video_from_episode(episode, out_path, fps=fps, render_size=render_size,
                                 model=model, device=device, start_img=start_img)
    else:
        from diffusion_policy.env.pusht.pusht_keypoints_env import PushTKeypointsEnv
        kp_kwargs = PushTKeypointsEnv.genenerate_keypoint_manager_params()
        env = PushTKeypointsEnv(render_size=render_size or 96, render_action=False, **kp_kwargs)
        create_video_from_episode_rendered(episode, env, out_path, fps=fps)


if __name__ == "__main__":
    main()
