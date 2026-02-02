"""
Visualize Training Data Coverage and Validation Prediction Errors

Creates:
1. 2D density plot of where the agent (cursor) spent time in training data
2. Heatmap of validation prediction errors at those coordinates
3. Overlay to see if errors correlate with low data coverage
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.ndimage import gaussian_filter
import click
import pickle
from infer_progress import load_model, predict, extract_action_chunk


def extract_agent_positions(episodes):
    """
    Extract all agent positions from episodes
    
    Returns:
        positions: numpy array [N, 2] of [agent_x, agent_y] positions
    """
    positions = []
    for episode in episodes:
        for step in episode:
            if 'state' in step and len(step['state']) >= 2:
                agent_pos = step['state'][:2]  # [agent_x, agent_y]
                positions.append(agent_pos)
    
    return np.array(positions, dtype=np.float32)


def compute_prediction_errors(model, episodes, device, use_siamese=True):
    """
    Compute prediction errors for all steps in episodes
    
    Returns:
        positions: numpy array [N, 2] of agent positions
        errors: numpy array [N] of absolute prediction errors
    """
    positions = []
    errors = []
    
    for episode in episodes:
        start_img = episode[0]['img'] if use_siamese else None
        
        for step_idx, step in enumerate(episode):
            if 'img' not in step or 'state' not in step:
                continue
            
            agent_pos = step['state'][:2]
            state = step['state']
            img = step['img']
            true_progress = step.get('normalized_timestep', step_idx / max(1, len(episode)-1))
            
            # Extract action chunk
            action_chunk = extract_action_chunk(episode, step_idx, 
                                               model.action_chunk_horizon, model.action_dim)
            
            # Predict progress
            pred_progress = predict(model, img, state, device=device,
                                   action_chunk=action_chunk, start_img=start_img)
            
            # Compute absolute error
            error = abs(pred_progress - true_progress)
            
            positions.append(agent_pos)
            errors.append(error)
    
    return np.array(positions, dtype=np.float32), np.array(errors, dtype=np.float32)


def create_density_heatmap(positions, grid_size=100, env_size=512, bandwidth=None):
    """
    Create a 2D density heatmap from agent positions using KDE
    
    Args:
        positions: [N, 2] array of agent positions
        grid_size: Resolution of the heatmap (grid_size x grid_size)
        env_size: Size of environment (512x512)
        bandwidth: KDE bandwidth (None for auto)
    
    Returns:
        density_grid: [grid_size, grid_size] density values
        x_grid: x coordinates for grid
        y_grid: y coordinates for grid
    """
    if len(positions) == 0:
        return np.zeros((grid_size, grid_size)), np.linspace(0, env_size, grid_size), np.linspace(0, env_size, grid_size)
    
    # Create grid
    x_min, x_max = 0, env_size
    y_min, y_max = 0, env_size
    x_grid = np.linspace(x_min, x_max, grid_size)
    y_grid = np.linspace(y_min, y_max, grid_size)
    X, Y = np.meshgrid(x_grid, y_grid)
    grid_points = np.vstack([X.ravel(), Y.ravel()]).T
    
    # Compute KDE
    try:
        kde = gaussian_kde(positions.T, bw_method=bandwidth)
        density = kde(grid_points.T)
        density_grid = density.reshape(grid_size, grid_size)
    except:
        # Fallback to histogram if KDE fails
        density_grid, x_edges, y_edges = np.histogram2d(
            positions[:, 0], positions[:, 1],
            bins=grid_size, range=[[x_min, x_max], [y_min, y_max]]
        )
        density_grid = density_grid.T  # Transpose to match imshow convention
    
    # Normalize to [0, 1]
    if density_grid.max() > 0:
        density_grid = density_grid / density_grid.max()
    
    return density_grid, x_grid, y_grid


def create_error_heatmap(positions, errors, grid_size=100, env_size=512):
    """
    Create a 2D heatmap of prediction errors by binning positions
    
    Args:
        positions: [N, 2] array of agent positions
        errors: [N] array of prediction errors
        grid_size: Resolution of the heatmap
        env_size: Size of environment
    
    Returns:
        error_grid: [grid_size, grid_size] mean error values
        x_grid: x coordinates for grid
        y_grid: y coordinates for grid
    """
    if len(positions) == 0:
        return np.zeros((grid_size, grid_size)), np.linspace(0, env_size, grid_size), np.linspace(0, env_size, grid_size)
    
    # Create bins
    x_min, x_max = 0, env_size
    y_min, y_max = 0, env_size
    x_grid = np.linspace(x_min, x_max, grid_size)
    y_grid = np.linspace(y_min, y_max, grid_size)
    
    # Bin positions and compute mean error per bin
    error_grid = np.zeros((grid_size, grid_size))
    count_grid = np.zeros((grid_size, grid_size))
    
    x_bins = np.digitize(positions[:, 0], x_grid) - 1
    y_bins = np.digitize(positions[:, 1], y_grid) - 1
    
    # Clip to valid range
    x_bins = np.clip(x_bins, 0, grid_size - 1)
    y_bins = np.clip(y_bins, 0, grid_size - 1)
    
    for i in range(len(positions)):
        x_bin = x_bins[i]
        y_bin = y_bins[i]
        error_grid[y_bin, x_bin] += errors[i]
        count_grid[y_bin, x_bin] += 1
    
    # Compute mean error per bin
    mask = count_grid > 0
    error_grid[mask] = error_grid[mask] / count_grid[mask]
    
    # Smooth the error grid
    error_grid = gaussian_filter(error_grid, sigma=1.0)
    
    return error_grid, x_grid, y_grid


@click.command()
@click.option('-m', '--model_path', default='progress_models/weights/pp_cat_action_start.pth',
              help='Path to progress predictor model')
@click.option('--mode', type=click.Choice(['categorical', 'regression']), default='categorical',
              help='Model mode')
@click.option('--action_chunk_horizon', default=8, type=int,
              help='Action chunk horizon')
@click.option('--action_dim', default=2, type=int,
              help='Action dimension')
@click.option('--num_bins', default=50, type=int,
              help='Number of bins for categorical mode')
@click.option('--use_siamese', is_flag=True, default=True,
              help='Use Siamese architecture')
@click.option('--dataset_path', default='pusht_episodes.pkl',
              help='Path to dataset')
@click.option('--device', default='auto',
              help='Device (auto, cpu, cuda)')
@click.option('--train_ratio', default=0.8, type=float,
              help='Train/val split ratio')
@click.option('--grid_size', default=100, type=int,
              help='Resolution of heatmaps (grid_size x grid_size)')
@click.option('--out_path', default='data_coverage_and_errors.png',
              help='Output file path')
def main(model_path, mode, action_chunk_horizon, action_dim, num_bins, use_siamese,
         dataset_path, device, train_ratio, grid_size, out_path):
    """
    Visualize training data coverage and validation prediction errors.
    """
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {model_path}...")
    model = load_model(
        model_path=model_path,
        mode=mode,
        action_chunk_horizon=action_chunk_horizon,
        action_dim=action_dim,
        num_bins=num_bins,
        use_siamese=use_siamese
    )
    model = model.to(device)
    print("Model loaded successfully!")
    
    # Load dataset
    print(f"Loading dataset from {dataset_path}...")
    with open(dataset_path, 'rb') as f:
        all_episodes = pickle.load(f)
    
    # Split into train/val
    split_idx = int(train_ratio * len(all_episodes))
    train_episodes = all_episodes[:split_idx]
    val_episodes = all_episodes[split_idx:]
    
    print(f"Train episodes: {len(train_episodes)}, Val episodes: {len(val_episodes)}")
    
    # Extract training agent positions
    print("Extracting training agent positions...")
    train_positions = extract_agent_positions(train_episodes)
    print(f"Extracted {len(train_positions)} training positions")
    
    # Compute validation prediction errors
    print("Computing validation prediction errors...")
    val_positions, val_errors = compute_prediction_errors(model, val_episodes, device, use_siamese)
    print(f"Computed errors for {len(val_positions)} validation positions")
    print(f"Mean error: {val_errors.mean():.4f}, Std: {val_errors.std():.4f}")
    print(f"Max error: {val_errors.max():.4f}, Min error: {val_errors.min():.4f}")
    
    # Create density heatmap of training data
    print("Creating training data density heatmap...")
    density_grid, x_grid, y_grid = create_density_heatmap(
        train_positions, grid_size=grid_size, env_size=512
    )
    
    # Create error heatmap
    print("Creating validation error heatmap...")
    error_grid, _, _ = create_error_heatmap(
        val_positions, val_errors, grid_size=grid_size, env_size=512
    )
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    
    # Left: Training data density
    ax1 = axes[0]
    im1 = ax1.imshow(density_grid, extent=[0, 512, 512, 0], 
                     origin='upper', cmap='viridis', interpolation='bilinear')
    ax1.set_title('Training Data Density\n(Where agent spent time)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('X Position', fontsize=12)
    ax1.set_ylabel('Y Position', fontsize=12)
    ax1.set_xlim(0, 512)
    ax1.set_ylim(512, 0)
    plt.colorbar(im1, ax=ax1, label='Normalized Density')
    ax1.grid(True, alpha=0.3)
    
    # Middle: Validation error heatmap
    ax2 = axes[1]
    im2 = ax2.imshow(error_grid, extent=[0, 512, 512, 0], 
                     origin='upper', cmap='hot', interpolation='bilinear')
    ax2.set_title('Validation Prediction Error\n(Absolute Error)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('X Position', fontsize=12)
    ax2.set_ylabel('Y Position', fontsize=12)
    ax2.set_xlim(0, 512)
    ax2.set_ylim(512, 0)
    plt.colorbar(im2, ax=ax2, label='Mean Absolute Error')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved visualization to: {out_path}")
    
    # Print statistics
    print("\n=== Analysis ===")
    print(f"Training positions: {len(train_positions)}")
    print(f"Validation positions: {len(val_positions)}")
    print(f"Mean validation error: {val_errors.mean():.4f}")
    print(f"Max validation error: {val_errors.max():.4f}")
    
    # Check correlation: areas with low density but high error
    # Sample points from low-density regions and check their errors
    low_density_threshold = 0.1  # Bottom 10% of density
    density_threshold = np.percentile(density_grid, 10)
    
    # Find validation points in low-density regions
    low_density_errors = []
    high_density_errors = []
    
    for i, pos in enumerate(val_positions):
        x_idx = int(np.clip(pos[0] / 512 * grid_size, 0, grid_size - 1))
        y_idx = int(np.clip(pos[1] / 512 * grid_size, 0, grid_size - 1))
        local_density = density_grid[y_idx, x_idx]
        
        if local_density < density_threshold:
            low_density_errors.append(val_errors[i])
        else:
            high_density_errors.append(val_errors[i])
    
    if len(low_density_errors) > 0 and len(high_density_errors) > 0:
        print(f"\nLow-density regions (bottom 10%):")
        print(f"  Mean error: {np.mean(low_density_errors):.4f}")
        print(f"  Count: {len(low_density_errors)}")
        print(f"\nHigh-density regions (top 90%):")
        print(f"  Mean error: {np.mean(high_density_errors):.4f}")
        print(f"  Count: {len(high_density_errors)}")
        print(f"\nError increase in low-density: {np.mean(low_density_errors) - np.mean(high_density_errors):.4f}")


if __name__ == "__main__":
    main()
