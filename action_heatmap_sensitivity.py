"""
Action Sensitivity Heatmap for Progress Predictor

Generates a PushT environment with random T position, then creates a grid of 
cursor (agent) positions and computes predicted progress for different future actions.
Visualizes the standard deviation of predictions as a heatmap - showing which regions
are most sensitive to action choice.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import click
import pickle
from tqdm import tqdm
from diffusion_policy.env.pusht.pusht_keypoints_env import PushTKeypointsEnv
from infer_progress import load_model, predict


def generate_action_directions(n_directions=8, magnitude=50.0):
    """Generate action directions at regular angle intervals"""
    angles = np.deg2rad(np.arange(0, 360, 360 / n_directions))
    directions = np.array([np.cos(angles), np.sin(angles)]).T * magnitude
    return directions


def compute_action_sensitivity(model, env, block_pose, agent_pose, grid_size=8, n_directions=8, action_magnitude=50.0, device='cpu'):
    """
    Compute action sensitivity for a grid of agent positions with fixed block position.
    
    Args:
        block_pose: [block_x, block_y, block_angle] from training data
        agent_pose: [agent_x, agent_y] original agent position from episode
        grid_size: Number of grid cells for agent positions
    
    Returns:
        grid_std: Standard deviation of predictions across actions for each grid cell
        grid_mean: Mean prediction across actions for each grid cell
        grid_predictions: Full predictions array [grid_size, grid_size, n_directions]
        x_positions: X coordinates of grid
        y_positions: Y coordinates of grid
        env_img: Environment snapshot with this block position and original agent position
    """
    # Get environment bounds (PushT is 512x512)
    env_size = 512
    
    # Generate grid of agent positions
    x_positions = np.linspace(20, env_size - 20, grid_size)
    y_positions = np.linspace(20, env_size - 20, grid_size)
    
    directions = generate_action_directions(n_directions, action_magnitude)
    
    # Storage for results
    grid_predictions = np.zeros((grid_size, grid_size, n_directions))
    
    horizon = model.action_chunk_horizon if model.action_chunk_horizon > 0 else 8
    
    # Set block position and original agent position
    # State format: [agent_x, agent_y, block_x, block_y, block_angle]
    state = [agent_pose[0], agent_pose[1], block_pose[0], block_pose[1], block_pose[2]]
    env._set_state(state)
    
    # Capture environment snapshot with original agent and block positions
    env_img = env.render(mode='rgb_array')
    
    for i, y in enumerate(y_positions):
        for j, x in enumerate(x_positions):
            # Move agent to this grid position
            agent_pos = np.array([x, y])
            state = [x, y, block_pose[0], block_pose[1], block_pose[2]]
            env._set_state(state)
            
            # Render image with agent at this position
            img_array = env.render(mode='rgb_array')
            img_array = img_array.transpose(2, 0, 1).astype(np.float32)
            
            # Create state for this agent position
            state = np.concatenate([agent_pos, block_pose]).astype(np.float32)
            
            # Predict progress for each action direction
            for k, direction in enumerate(directions):
                action_chunk = []
                for t in range(horizon):
                    progress_fraction = (t + 1) / horizon
                    target_pos = agent_pos + direction * progress_fraction
                    target_pos = np.clip(target_pos, 0, env_size).astype(np.float32)
                    action_chunk.append(target_pos)
                
                action_chunk = np.array(action_chunk, dtype=np.float32)
                progress = predict(model, img_array, state, device=device, action_chunk=action_chunk)
                grid_predictions[i, j, k] = progress
    
    # Compute statistics
    grid_std = np.std(grid_predictions, axis=2)
    grid_mean = np.mean(grid_predictions, axis=2)
    
    return grid_std, grid_mean, grid_predictions, x_positions, y_positions, env_img


def load_training_block_positions(dataset_path='pusht_episodes.pkl', train_ratio=0.8, max_episodes=50):
    """
    Load block (T) positions and agent positions from training episodes
    
    Returns:
        block_poses: List of [block_x, block_y, block_angle] from first step of each training episode
        agent_poses: List of [agent_x, agent_y] from first step of each training episode
        episode_info: List of dicts with 'split' (train/val), 'ep_idx', 'frame_idx' for each block pose
    """
    print(f"Loading dataset from {dataset_path}...")
    with open(dataset_path, 'rb') as f:
        episodes = pickle.load(f)
    
    # Use same train/val split as training
    split_idx = int(train_ratio * len(episodes))
    train_episodes = episodes[:split_idx]
    val_episodes = episodes[split_idx:]
    
    print(f"Extracting block positions from {len(train_episodes)} training episodes...")
    block_poses = []
    agent_poses = []
    episode_info = []
    val_block_poses = []
    val_agent_poses = []
    val_episode_info = []
    
    # Train episodes
    for ep_idx, episode in enumerate(train_episodes[:max_episodes]):
        if len(episode) > 0:
            first_step = episode[0]
            block_pose = first_step['state'][2:5].astype(np.float32)  # [block_x, block_y, block_angle]
            agent_pose = first_step['state'][:2].astype(np.float32)  # [agent_x, agent_y]
            block_poses.append(block_pose)
            agent_poses.append(agent_pose)
            episode_info.append({
                'split': 'train',
                'ep_idx': ep_idx,
                'frame_idx': 0
            })

    # Val episodes
    for ep_idx, episode in enumerate(val_episodes):
        if len(episode) > 0:
            first_step = episode[0]
            block_pose = first_step['state'][2:5].astype(np.float32)  # [block_x, block_y, block_angle]
            agent_pose = first_step['state'][:2].astype(np.float32)  # [agent_x, agent_y]
            val_block_poses.append(block_pose)
            val_agent_poses.append(agent_pose)
            val_episode_info.append({
                'split': 'val',
                'ep_idx': ep_idx,
                'frame_idx': 0
            })
    
    print(f"Extracted {len(block_poses)} block positions")
    return block_poses, agent_poses, episode_info, val_block_poses, val_agent_poses, val_episode_info


def plot_sensitivity_heatmap(grid_std, grid_mean, x_positions, y_positions, env_img, 
                              block_pose, agent_pose, out_path='action_sensitivity_heatmap.png',
                              title_label=None, render_size=96):
    """Plot the action sensitivity heatmap"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    if title_label:
        fig.suptitle(title_label, fontsize=14, fontweight='bold', y=0.98)
    
    # Environment coordinate system is 512x512
    env_size = 512
    
    # Plot 1: Environment snapshot
    extent_env = [0, env_size, env_size, 0]  # [left, right, bottom, top] for imshow
    axes[0].imshow(env_img, extent=extent_env, aspect='auto')
    axes[0].set_title('Environment State')
    axes[0].axis('off')
    # Mark block position
    axes[0].scatter([block_pose[0]], [block_pose[1]], s=300, c='purple', 
                   marker='*', edgecolors='white', linewidths=2, 
                   label='Block (T)', zorder=10)
    
    # Plot 2: Standard deviation heatmap (action sensitivity)
    extent = [x_positions[0], x_positions[-1], y_positions[-1], y_positions[0]]
    im1 = axes[1].imshow(grid_std, cmap='hot', extent=extent, aspect='auto')
    axes[1].set_title('Action Sensitivity (Std Dev of Predictions)')
    axes[1].set_xlabel('X Position')
    axes[1].set_ylabel('Y Position')
    # Mark block position with purple marker
    axes[1].scatter([block_pose[0]], [block_pose[1]], s=300, c='purple', 
                   marker='*', edgecolors='white', linewidths=2, 
                   label='Block (T)', zorder=10)
    # Mark agent position
    axes[1].scatter([agent_pose[0]], [agent_pose[1]], s=200, c='blue', 
                   marker='o', edgecolors='white', linewidths=2, 
                   label='Agent', zorder=10)
    axes[1].legend(loc='upper right', fontsize=9)
    plt.colorbar(im1, ax=axes[1], label='Std Dev')
    
    # Plot 3: Mean prediction heatmap
    im2 = axes[2].imshow(grid_mean, cmap='RdYlGn', extent=extent, aspect='auto', vmin=0, vmax=1)
    axes[2].set_title('Mean Predicted Progress')
    axes[2].set_xlabel('X Position')
    axes[2].set_ylabel('Y Position')
    # Mark block position with purple marker
    axes[2].scatter([block_pose[0]], [block_pose[1]], s=300, c='purple', 
                   marker='*', edgecolors='white', linewidths=2, 
                   label='Block (T)', zorder=10)
    # Mark agent position
    axes[2].scatter([agent_pose[0]], [agent_pose[1]], s=200, c='blue', 
                   marker='o', edgecolors='white', linewidths=2, 
                   label='Agent', zorder=10)
    axes[2].legend(loc='upper right', fontsize=9)
    plt.colorbar(im2, ax=axes[2], label='Progress')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96] if title_label else None)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved heatmap to {out_path}")



@click.command()
@click.option('-m', '--model_path', default='progress_models/weights/pp_regress_action8.pth',
              help='Path to progress predictor model')
@click.option('--mode', default='regression', help='Model mode (categorical or regression)')
@click.option('--dataset_path', default='pusht_episodes.pkl', help='Path to dataset')
@click.option('--n_episodes', default=5, type=int, help='Number of training episodes to test')
@click.option('--grid_size', default=8, type=int, help='Grid resolution for agent positions')
@click.option('--n_directions', default=8, type=int, help='Number of action directions to test')
@click.option('--action_magnitude', default=50.0, type=float, help='Action magnitude in pixels')
@click.option('--action_chunk_horizon', default=8, type=int, help='Action chunk horizon')
@click.option('--device', default='auto', help='Device (auto, cpu, cuda)')
@click.option('--out_prefix', default='action_sensitivity', help='Output file prefix')
def main(model_path, mode, dataset_path, n_episodes, grid_size, n_directions, action_magnitude, action_chunk_horizon, device, out_prefix):
    """Generate action sensitivity heatmap using block positions from training data"""
    
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {model_path}...")
    model = load_model(
        model_path=model_path,
        mode=mode,
        action_chunk_horizon=action_chunk_horizon,
        action_dim=2
    )
    model = model.to(device)
    print("Model loaded!")
    
    # Load block positions and agent positions from training data
    block_poses, agent_poses, episode_info, val_block_poses, val_agent_poses, val_episode_info = load_training_block_positions(dataset_path=dataset_path, max_episodes=n_episodes)
    
    # Create environment
    kp_kwargs = PushTKeypointsEnv.genenerate_keypoint_manager_params()
    env = PushTKeypointsEnv(render_size=96, render_action=False, **kp_kwargs)
    env.reset()  # Initialize environment once
    
    # Generate heatmap for each block position (train episodes)
    for idx, (block_pose, agent_pose, info) in enumerate(zip(block_poses, agent_poses, episode_info)):
        print(f"\n=== Episode {idx+1}/{len(block_poses)} ===")
        print(f"Block position: [{block_pose[0]:.1f}, {block_pose[1]:.1f}], angle: {block_pose[2]:.2f}")
        print(f"Agent position: [{agent_pose[0]:.1f}, {agent_pose[1]:.1f}]")
        
        # Compute action sensitivity for grid of agent positions
        grid_std, grid_mean, grid_predictions, x_pos, y_pos, env_img = compute_action_sensitivity(
            model, env, block_pose, agent_pose, grid_size=grid_size, n_directions=n_directions,
            action_magnitude=action_magnitude, device=device
        )
        
        print(f"Std range: [{grid_std.min():.4f}, {grid_std.max():.4f}]")
        print(f"Mean range: [{grid_mean.min():.4f}, {grid_mean.max():.4f}]")
        
        # Create title label similar to visualize_action_chunks
        split_label = info['split'].capitalize()
        title_label = f"{split_label} Ep {info['ep_idx']+1} | Frame {info['frame_idx']}"
        
        # Get render size from environment
        render_size = env.render_size if hasattr(env, 'render_size') else 96
        
        # Plot heatmap
        plot_sensitivity_heatmap(
            grid_std, grid_mean, x_pos, y_pos, env_img, block_pose, agent_pose,
            out_path=f'{out_prefix}_ep{idx+1}.png',
            title_label=title_label,
            render_size=render_size
        )
    
    # Generate heatmap for validation episodes if available
    if len(val_block_poses) > 0:
        print(f"\n=== Processing {len(val_block_poses)} validation episodes ===")
        for idx, (block_pose, agent_pose, info) in enumerate(zip(val_block_poses, val_agent_poses, val_episode_info)):
            print(f"\n=== Val Episode {idx+1}/{len(val_block_poses)} ===")
            print(f"Block position: [{block_pose[0]:.1f}, {block_pose[1]:.1f}], angle: {block_pose[2]:.2f}")
            print(f"Agent position: [{agent_pose[0]:.1f}, {agent_pose[1]:.1f}]")
            
            # Compute action sensitivity for grid of agent positions
            grid_std, grid_mean, grid_predictions, x_pos, y_pos, env_img = compute_action_sensitivity(
                model, env, block_pose, agent_pose, grid_size=grid_size, n_directions=n_directions,
                action_magnitude=action_magnitude, device=device
            )
            
            print(f"Std range: [{grid_std.min():.4f}, {grid_std.max():.4f}]")
            print(f"Mean range: [{grid_mean.min():.4f}, {grid_mean.max():.4f}]")
            
            # Create title label
            split_label = info['split'].capitalize()
            title_label = f"{split_label} Ep {info['ep_idx']+1} | Frame {info['frame_idx']}"
            
            # Plot heatmap
            plot_sensitivity_heatmap(
                grid_std, grid_mean, x_pos, y_pos, env_img, block_pose, agent_pose,
                out_path=f'{out_prefix}_val_ep{idx+1}.png',
                title_label=title_label,
                render_size=render_size
            )

if __name__ == "__main__":
    main()
