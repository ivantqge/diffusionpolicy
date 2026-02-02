"""
Compare Expert Action vs Random Walks to Same Destination

For sampled frames from train/val sets:
1. Get expert's action chunk and its final destination
2. Generate 7 random walks that all end at the same destination
3. Compare predicted progress for expert path vs random walks
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import click
import pickle
from infer_progress import load_model, predict, extract_action_chunk


def generate_random_walk_to_destination(agent_pos, destination, horizon=8, noise_scale=30.0, seed=None):
    """
    Generate a random walk from agent_pos to destination.
    
    The walk takes a meandering path but ends at the destination.
    
    Args:
        agent_pos: Starting position [x, y]
        destination: Final position [x, y]
        horizon: Number of steps
        noise_scale: Scale of random perturbations
        seed: Random seed for reproducibility
    
    Returns:
        action_chunk: [horizon, 2] trajectory
    """
    if seed is not None:
        np.random.seed(seed)
    
    agent_pos = np.array(agent_pos, dtype=np.float32)
    destination = np.array(destination, dtype=np.float32)
    
    # Generate random intermediate waypoints
    # Linear interpolation + noise, but force the last point to be exactly destination
    action_chunk = []
    
    for i in range(horizon):
        t = (i + 1) / horizon  # Progress from 0 to 1
        
        # Linear interpolation
        linear_pos = agent_pos + t * (destination - agent_pos)
        
        if i < horizon - 1:
            # Add noise to intermediate points
            # Noise decreases as we approach the destination
            noise_factor = 1.0 - t  # More noise early, less noise later
            noise = np.random.randn(2) * noise_scale * noise_factor
            pos = linear_pos + noise
            pos = np.clip(pos, 0, 512).astype(np.float32)
        else:
            # Last point is exactly the destination
            pos = destination.copy()
            pos = np.clip(pos, 0, 512).astype(np.float32)
        
        action_chunk.append(pos)
    
    return np.array(action_chunk, dtype=np.float32)


def generate_random_walks_to_destination(agent_pos, destination, n_walks=7, horizon=8, noise_scale=30.0):
    """
    Generate multiple random walks that all end at the same destination.
    
    Args:
        agent_pos: Starting position [x, y]
        destination: Final position [x, y]
        n_walks: Number of random walks to generate
        horizon: Number of steps per walk
        noise_scale: Scale of random perturbations
    
    Returns:
        action_chunks: List of n_walks action chunks, each [horizon, 2]
    """
    action_chunks = []
    for i in range(n_walks):
        chunk = generate_random_walk_to_destination(
            agent_pos, destination, horizon, noise_scale, seed=i*1000 + np.random.randint(10000)
        )
        action_chunks.append(chunk)
    
    return action_chunks


def evaluate_all_paths(model, img, state, start_img, device, 
                       expert_action_chunk, random_walks):
    """
    Evaluate expert path and all random walks.
    
    Returns:
        expert_progress: Predicted progress for expert path
        random_predictions: List of predictions for random walks
        best_random_idx: Index of best random walk
    """
    # Evaluate expert
    expert_progress = predict(model, img, state, device=device,
                             action_chunk=expert_action_chunk, start_img=start_img)
    
    # Evaluate random walks
    random_predictions = []
    for chunk in random_walks:
        progress = predict(model, img, state, device=device,
                          action_chunk=chunk, start_img=start_img)
        random_predictions.append(progress)
    
    best_random_idx = np.argmax(random_predictions) if random_predictions else -1
    
    return expert_progress, random_predictions, best_random_idx


def plot_comparison(fig_title, img, agent_state, block_pose, 
                   expert_action_chunk, random_walks,
                   expert_progress, random_predictions, true_progress,
                   out_path):
    """
    Create visualization showing expert path vs random walks to same destination.
    Left: All paths overlaid (expert + random walks)
    Right: Bar chart comparing progress predictions
    """
    # Normalize image
    img_display = img
    if img_display.dtype != np.float32 and img_display.dtype != np.float64:
        img_display = img_display.astype(np.float32)
    if img_display.max() > 1.5:
        img_display = img_display / 255.0
    img_display = np.clip(img_display, 0, 1)
    
    H, W = img_display.shape[:2]
    scale = W / 512.0
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)
    fig.suptitle(fig_title, fontsize=14, fontweight='bold')
    
    # Left panel: All paths overlaid
    ax1.imshow(img_display)
    ax1.set_title(f"Expert Path vs Random Walks\n(All end at same destination)", 
                  fontsize=12, fontweight='bold')
    ax1.set_xlim(0, W)
    ax1.set_ylim(H, 0)
    ax1.axis("off")
    
    # Plot agent position
    agent_scaled = agent_state[:2] * scale
    ax1.scatter([agent_scaled[0]], [agent_scaled[1]], s=150, c='blue', 
                edgecolors='white', linewidths=2, label='Agent (start)', zorder=15)
    
    # Plot block position
    block_scaled = block_pose[:2] * scale
    ax1.scatter([block_scaled[0]], [block_scaled[1]], s=150, c='purple', 
                marker='*', edgecolors='white', linewidths=2, label='Block', zorder=15)
    
    # Color palette for random walks (grays/blues)
    random_colors = plt.cm.Blues(np.linspace(0.3, 0.7, len(random_walks)))
    
    # Draw random walks first (behind expert)
    for i, chunk in enumerate(random_walks):
        traj = [agent_scaled.copy()]
        for action in chunk:
            traj.append(action * scale)
        traj = np.array(traj)
        
        ax1.plot(traj[:, 0], traj[:, 1], '-', linewidth=1.5, 
                color=random_colors[i], alpha=0.6, zorder=5)
        ax1.scatter(traj[1:-1, 0], traj[1:-1, 1], s=20, 
                   color=random_colors[i], alpha=0.5, zorder=5)
    
    # Draw expert path (on top, highlighted)
    if expert_action_chunk is not None and len(expert_action_chunk) > 0:
        traj_expert = [agent_scaled.copy()]
        for action in expert_action_chunk:
            traj_expert.append(action * scale)
        traj_expert = np.array(traj_expert)
        
        ax1.plot(traj_expert[:, 0], traj_expert[:, 1], 'o-', linewidth=3, 
                color='orange', alpha=0.95, markersize=8, label='Expert path', zorder=10)
        
        # Mark destination (shared by all)
        dest = traj_expert[-1]
        ax1.scatter([dest[0]], [dest[1]], s=200, c='red', marker='X',
                   edgecolors='white', linewidths=2, label='Destination', zorder=20)
    
    # Add legend for random walks
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='orange', linewidth=3, marker='o', markersize=8, label=f'Expert ({expert_progress:.3f})'),
        Line2D([0], [0], color='steelblue', linewidth=2, alpha=0.6, label=f'Random walks (n={len(random_walks)})'),
        Line2D([0], [0], marker='X', color='red', markersize=10, linestyle='None', label='Shared destination'),
    ]
    ax1.legend(handles=legend_elements, loc='lower right', fontsize=9)
    
    info_text = (
        f"True Progress: {true_progress:.2f}\n"
        f"Agent: [{agent_state[0]:.1f}, {agent_state[1]:.1f}]\n"
        f"Block: [{block_pose[0]:.1f}, {block_pose[1]:.1f}]"
    )
    ax1.text(0.02, 0.98, info_text, transform=ax1.transAxes,
            va='top', ha='left', fontsize=10,
            bbox=dict(boxstyle='round,pad=0.35', facecolor='white', alpha=0.85))
    
    # Right panel: Bar chart of predictions
    labels = ['Expert'] + [f'Walk {i+1}' for i in range(len(random_walks))]
    values = [expert_progress] + list(random_predictions)
    colors = ['orange'] + ['steelblue'] * len(random_walks)
    
    bars = ax2.barh(labels, values, color=colors, edgecolor='white', linewidth=1)
    ax2.set_xlabel('Predicted Progress', fontsize=11)
    ax2.set_title('Progress Predictions\n(Same destination, different paths)', fontsize=12, fontweight='bold')
    ax2.set_xlim(0, 1)
    ax2.axvline(x=true_progress, color='red', linestyle='--', linewidth=2, label=f'True progress ({true_progress:.2f})')
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        ax2.text(val + 0.02, bar.get_y() + bar.get_height()/2, 
                f'{val:.3f}', va='center', fontsize=9)
    
    # Highlight if expert is best or not
    best_random = max(random_predictions) if random_predictions else 0
    if expert_progress >= best_random:
        result_text = "Expert path has highest prediction"
        result_color = 'green'
    else:
        diff = best_random - expert_progress
        result_text = f"Random walk beats expert by {diff:.3f}"
        result_color = 'red'
    
    ax2.text(0.5, 0.02, result_text, transform=ax2.transAxes,
            ha='center', fontsize=11, fontweight='bold', color=result_color,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))
    
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(axis='x', alpha=0.3)
    
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"Saved: {out_path}")


@click.command()
@click.option('-m', '--model_path', default='progress_models/weights/pp_regress_action8.pth',
              help='Path to progress predictor model')
@click.option('--mode', type=click.Choice(['categorical', 'regression']), default='regression',
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
@click.option('--n_episodes', default=5, type=int,
              help='Number of episodes to sample from each split')
@click.option('--n_frames_per_episode', default=3, type=int,
              help='Number of frames per episode')
@click.option('--n_random_walks', default=7, type=int,
              help='Number of random walks to generate')
@click.option('--noise_scale', default=30.0, type=float,
              help='Noise scale for random walk perturbations')
@click.option('--out_prefix', default='action_comparison',
              help='Output file prefix')
def main(model_path, mode, action_chunk_horizon, action_dim, num_bins, use_siamese,
         dataset_path, device, n_episodes, n_frames_per_episode, n_random_walks, 
         noise_scale, out_prefix):
    """
    Compare expert action path vs random walks to the same destination.
    
    Tests if the progress predictor cares about the PATH taken or just the DESTINATION.
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
    train_ratio = 0.8
    split_idx = int(train_ratio * len(all_episodes))
    train_episodes = all_episodes[:split_idx]
    val_episodes = all_episodes[split_idx:]
    
    print(f"Train episodes: {len(train_episodes)}, Val episodes: {len(val_episodes)}")
    print(f"Generating {n_random_walks} random walks per frame (noise_scale={noise_scale})")
    
    # Track statistics
    expert_wins = 0
    random_wins = 0
    total_frames = 0
    
    # Process train episodes
    print("\n=== Processing Train Episodes ===")
    for ep_idx, episode in enumerate(train_episodes[:n_episodes]):
        if len(episode) == 0:
            continue
        
        print(f"\nEpisode {ep_idx + 1}:")
        
        # Sample frames evenly spaced (skip very end to have expert actions)
        max_frame = max(0, len(episode) - action_chunk_horizon - 1)
        if max_frame <= 0:
            continue
        frame_indices = np.linspace(0, max_frame, n_frames_per_episode, dtype=int)
        
        for frame_idx in frame_indices:
            step = episode[frame_idx]
            
            if 'img' not in step or 'state' not in step:
                continue
            
            # Get current state
            agent_state = step['state'][:2]  # [agent_x, agent_y]
            block_pose = step['state'][2:5]  # [block_x, block_y, block_angle]
            state = step['state']
            
            # Get image
            img = step['img']
            start_img = episode[0]['img'] if hasattr(model, 'use_siamese') and model.use_siamese else None
            
            # Get expert's actual action chunk
            expert_action_chunk = extract_action_chunk(episode, frame_idx, 
                                                      action_chunk_horizon, action_dim)
            
            if expert_action_chunk is None or len(expert_action_chunk) < action_chunk_horizon:
                continue
            
            # Get expert's destination (final position of action chunk)
            destination = expert_action_chunk[-1]  # [x, y]
            
            # Generate random walks to the same destination
            random_walks = generate_random_walks_to_destination(
                agent_pos=agent_state,
                destination=destination,
                n_walks=n_random_walks,
                horizon=action_chunk_horizon,
                noise_scale=noise_scale
            )
            
            # Evaluate all paths
            expert_progress, random_predictions, best_random_idx = evaluate_all_paths(
                model, img, state, start_img, device,
                expert_action_chunk, random_walks
            )
            
            true_progress = step.get('normalized_timestep', frame_idx / max(1, len(episode)-1))
            
            # Track wins
            total_frames += 1
            best_random = max(random_predictions) if random_predictions else 0
            if expert_progress >= best_random:
                expert_wins += 1
            else:
                random_wins += 1
            
            # Create visualization
            title = f"Train Ep {ep_idx+1} | Frame {frame_idx} | Expert vs {n_random_walks} Random Walks"
            out_path = f'{out_prefix}_train_ep{ep_idx+1}_frame{frame_idx}.png'
            
            plot_comparison(
                fig_title=title,
                img=img,
                agent_state=agent_state,
                block_pose=block_pose,
                expert_action_chunk=expert_action_chunk,
                random_walks=random_walks,
                expert_progress=expert_progress,
                random_predictions=random_predictions,
                true_progress=true_progress,
                out_path=out_path
            )
            
            print(f"  Frame {frame_idx}: Expert={expert_progress:.3f}, "
                  f"Best Random={best_random:.3f}, True={true_progress:.2f}")
    
    # Process val episodes
    print("\n=== Processing Val Episodes ===")
    for ep_idx, episode in enumerate(val_episodes[:n_episodes]):
        if len(episode) == 0:
            continue
        
        print(f"\nEpisode {ep_idx + 1}:")
        
        # Sample frames evenly spaced
        max_frame = max(0, len(episode) - action_chunk_horizon - 1)
        if max_frame <= 0:
            continue
        frame_indices = np.linspace(0, max_frame, n_frames_per_episode, dtype=int)
        
        for frame_idx in frame_indices:
            step = episode[frame_idx]
            
            if 'img' not in step or 'state' not in step:
                continue
            
            # Get current state
            agent_state = step['state'][:2]  # [agent_x, agent_y]
            block_pose = step['state'][2:5]  # [block_x, block_y, block_angle]
            state = step['state']
            
            # Get image
            img = step['img']
            start_img = episode[0]['img'] if hasattr(model, 'use_siamese') and model.use_siamese else None
            
            # Get expert's actual action chunk
            expert_action_chunk = extract_action_chunk(episode, frame_idx, 
                                                      action_chunk_horizon, action_dim)
            
            if expert_action_chunk is None or len(expert_action_chunk) < action_chunk_horizon:
                continue
            
            # Get expert's destination
            destination = expert_action_chunk[-1]
            
            # Generate random walks to the same destination
            random_walks = generate_random_walks_to_destination(
                agent_pos=agent_state,
                destination=destination,
                n_walks=n_random_walks,
                horizon=action_chunk_horizon,
                noise_scale=noise_scale
            )
            
            # Evaluate all paths
            expert_progress, random_predictions, best_random_idx = evaluate_all_paths(
                model, img, state, start_img, device,
                expert_action_chunk, random_walks
            )
            
            true_progress = step.get('normalized_timestep', frame_idx / max(1, len(episode)-1))
            
            # Track wins
            total_frames += 1
            best_random = max(random_predictions) if random_predictions else 0
            if expert_progress >= best_random:
                expert_wins += 1
            else:
                random_wins += 1
            
            # Create visualization
            title = f"Val Ep {ep_idx+1} | Frame {frame_idx} | Expert vs {n_random_walks} Random Walks"
            out_path = f'{out_prefix}_val_ep{ep_idx+1}_frame{frame_idx}.png'
            
            plot_comparison(
                fig_title=title,
                img=img,
                agent_state=agent_state,
                block_pose=block_pose,
                expert_action_chunk=expert_action_chunk,
                random_walks=random_walks,
                expert_progress=expert_progress,
                random_predictions=random_predictions,
                true_progress=true_progress,
                out_path=out_path
            )
            
            print(f"  Frame {frame_idx}: Expert={expert_progress:.3f}, "
                  f"Best Random={best_random:.3f}, True={true_progress:.2f}")
    
    # Print summary
    print("\n" + "="*50)
    print("SUMMARY: Does the model care about PATH or just DESTINATION?")
    print("="*50)
    print(f"Total frames analyzed: {total_frames}")
    print(f"Expert path wins: {expert_wins} ({100*expert_wins/max(1,total_frames):.1f}%)")
    print(f"Random walk wins: {random_wins} ({100*random_wins/max(1,total_frames):.1f}%)")
    if expert_wins > random_wins:
        print("\n=> Model appears to care about the PATH, not just destination!")
    else:
        print("\n=> Model may only care about DESTINATION, not the path taken.")
    print("\nDone!")


if __name__ == "__main__":
    main()
