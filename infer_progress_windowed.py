"""
Inference script for windowed progress predictor models.

This script handles evaluation and visualization for the windowed GRU-based
progress predictor architecture.
"""

import torch
import numpy as np
from progress_predictor_windowed import ProgressPredictorWindowed
import pickle
import matplotlib.pyplot as plt


def load_model(model_path='progress_predictor_best.pth', num_bins=50, pretrained=True, 
               freeze_encoder=True, mode=None, visual_feat_dim=512, state_feat_dim=64, 
               token_dim=256, gru_hidden_dim=128, window_length=4):
    """
    Load windowed progress predictor model
    """
    model = ProgressPredictorWindowed(
        agent_state_dim=5,
        pretrained=pretrained,
        num_bins=num_bins,
        freeze_encoder=freeze_encoder,
        mode=mode,
        encoder_ckpt_path=None,
        visual_feat_dim=visual_feat_dim,
        state_feat_dim=state_feat_dim,
        token_dim=token_dim,
        gru_hidden_dim=gru_hidden_dim,
        dropout=0.1
    )
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    model.window_length = window_length  # Store for inference
    print(f"Loaded windowed model in {mode} mode")
    print(f"Architecture: Windowed (L={window_length})")
    return model


def predict_windowed(model, episode, step_idx, device='cpu', return_distribution=False, 
                    state_mean=None, state_std=None):
    """
    Predict progress using windowed model
    Args:
        model: ProgressPredictorWindowed model
        episode: List of episode steps
        step_idx: Current step index (target frame)
        device: device to run on
        return_distribution: if True, also return probability distribution over bins
        state_mean: State normalization mean (from training)
        state_std: State normalization std (from training)
    Returns:
        progress value [0, 1] (and optionally distribution [num_bins] for categorical mode)
    """
    def process_image(img_array):
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            img_array = img_array.transpose(2, 0, 1)  # HWC -> CHW
        img_array = img_array.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
        img_array = (img_array - mean) / std
        return img_array
    
    # Create window: (o0, o_ta, o_ta+s, ..., o_ta+(L-2)s)
    # For inference: start frame + frames leading to step_idx
    L = model.window_length
    episode_length = len(episode)
    
    # Simple windowing: [0, step_idx - (L-2), step_idx - (L-3), ..., step_idx]
    # Ensure we have enough frames
    # Build window that matches training: [0, ta, ta+s, ..., ta+(L-2)s] with last == step_idx
    if step_idx <= 0:
        window_indices = [0] * L
    elif step_idx < L - 1:
        # too early: pad with earliest frames
        window_indices = [0] + [min(i, episode_length - 1) for i in range(1, L)]
    else:
        stride_choices = [2, 4, 6] # same as training
        s = None
        for cand in reversed(stride_choices):
            if step_idx - (L - 2) * cand >= 1:
                s = cand
                break
        if s is None:
            s = 1

        ta = step_idx - (L - 2) * s  # ensures last index = step_idx
        window_indices = [0] + [ta + i * s for i in range(L - 1)]

    # Clamp indices just in case
    window_indices = [min(max(idx, 0), episode_length - 1) for idx in window_indices]

    
    # Get images and states
    images_list = []
    states_list = []
    
    for idx in window_indices:
        idx = min(idx, episode_length - 1)  # Clamp
        step = episode[idx]
        img_processed = process_image(step['img'])
        state = step['state'].astype(np.float32)
        
        # Normalize state if provided
        if state_mean is not None and state_std is not None:
            if isinstance(state_mean, torch.Tensor):
                state = (state - state_mean.numpy()) / (state_std.numpy() + 1e-6)
            else:
                state = (state - state_mean) / (state_std + 1e-6)
        
        images_list.append(img_processed)
        states_list.append(state)
    
    # Stack into sequences: [L, C, H, W] and [L, state_dim]
    images_seq = np.stack(images_list, axis=0)  # [L, C, H, W]
    states_seq = np.stack(states_list, axis=0)  # [L, state_dim]
    
    # Convert to tensors and add batch dimension
    images_t = torch.FloatTensor(images_seq).unsqueeze(0).to(device)  # [1, L, C, H, W]
    states_t = torch.FloatTensor(states_seq).unsqueeze(0).to(device)  # [1, L, state_dim]
    
    with torch.no_grad():
        output = model(images_t, states_t)
        
        if model.mode == 'categorical':
            progress = model.progress_from_logits(output)
            if return_distribution:
                distribution = model.get_distribution(output)
                return progress.item(), distribution[0].cpu().numpy()
            return progress.item()
        else:
            progress = output.squeeze()
            if return_distribution:
                dist = np.zeros(model.num_bins)
                bin_idx = int(np.clip(progress.item() * model.num_bins, 0, model.num_bins - 1))
                dist[bin_idx] = 1.0
                return progress.item(), dist
            return progress.item()


def plot_progress_over_time(model, episodes, ep_idx=0, device='cpu', out_path='progress_over_time.png',
                           state_mean=None, state_std=None):
    """
    Plot prediction vs real progress over the course of one episode with std bands
    """
    episode = episodes[ep_idx]
    
    # Get predictions and distributions for all timesteps
    preds, targets, dist_stds = [], [], []
    bin_centers = np.arange(0.5, model.num_bins) / model.num_bins
    
    for step_idx, step in enumerate(episode):
        if model.mode == 'categorical':
            pred, dist = predict_windowed(model, episode, step_idx, device, return_distribution=True,
                                        state_mean=state_mean, state_std=state_std)
            mean_dist = np.sum(dist * bin_centers)
            variance = np.sum(dist * (bin_centers - mean_dist) ** 2)
            dist_std = np.sqrt(variance)
            dist_stds.append(dist_std)
        else:
            pred = predict_windowed(model, episode, step_idx, device, return_distribution=False,
                                  state_mean=state_mean, state_std=state_std)
        
        true = step['normalized_timestep']
        preds.append(pred)
        targets.append(true)
    
    timesteps = np.arange(len(preds))
    
    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(timesteps, targets, 'g-', label='True Progress', linewidth=2, alpha=0.8)
    ax.plot(timesteps, preds, 'r-', label='Predicted Progress', linewidth=2, alpha=0.8)
    
    if model.mode == 'categorical' and dist_stds:
        preds_array = np.array(preds)
        stds_array = np.array(dist_stds)
        ax.fill_between(timesteps, preds_array - stds_array, preds_array + stds_array, 
                         alpha=0.3, color='red', label='±1 std')
    
    ax.set_xlabel('Timestep', fontsize=12)
    ax.set_ylabel('Progress [0, 1]', fontsize=12)
    ax.set_title(f'Progress Prediction Over Time - Episode {ep_idx}', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    mae = np.mean(np.abs(np.array(preds) - np.array(targets)))
    if model.mode == 'categorical' and dist_stds:
        mean_std = np.mean(dist_stds)
        ax.text(0.02, 0.98, f'MAE: {mae:.4f}\nMean σ: {mean_std:.4f}', transform=ax.transAxes, 
                fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    else:
        ax.text(0.02, 0.98, f'MAE: {mae:.4f}', transform=ax.transAxes, 
                fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")


def plot_progress_grid(model, episodes, ep_indices=None, device='cpu', out_path='progress_over_time_grid.png', 
                      episodes_per_page=5, actual_ep_numbers=None, state_mean=None, state_std=None):
    """
    Plot prediction vs real progress over time for multiple episodes in a grid
    Always shows exactly episodes_per_page episodes (default 5) per PNG
    
    Args:
        actual_ep_numbers: Optional list of actual episode numbers for display (if None, uses ep_indices)
    """
    if ep_indices is None:
        ep_indices = list(range(min(episodes_per_page, len(episodes))))
    
    # Have 5 episodes per page
    n_episodes = min(len(ep_indices), episodes_per_page)
    ep_indices = ep_indices[:n_episodes]  

    if actual_ep_numbers is None:
        actual_ep_numbers = ep_indices
    else:
        actual_ep_numbers = actual_ep_numbers[:n_episodes]
    
    n_cols = 3
    n_rows = 2  #
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_episodes == 1:
        axes = np.array([axes])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    else:
        axes = axes.reshape(n_rows, n_cols)
    
    for idx, ep_idx in enumerate(ep_indices):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        
        episode = episodes[ep_idx]
        
        # Get predictions and distributions for all timesteps
        preds, targets, dist_stds = [], [], []
        bin_centers = np.arange(0.5, model.num_bins) / model.num_bins
        
        for step_idx, step in enumerate(episode):
            if model.mode == 'categorical':
                pred, dist = predict_windowed(model, episode, step_idx, device, return_distribution=True,
                                            state_mean=state_mean, state_std=state_std)
                # Compute std of distribution
                mean_dist = np.sum(dist * bin_centers)
                variance = np.sum(dist * (bin_centers - mean_dist) ** 2)
                dist_std = np.sqrt(variance)
                dist_stds.append(dist_std)
            else:
                pred = predict_windowed(model, episode, step_idx, device, return_distribution=False,
                                      state_mean=state_mean, state_std=state_std)
            
            true = step['normalized_timestep']
            preds.append(pred)
            targets.append(true)
        
        timesteps = np.arange(len(preds))
        
        # Plot
        ax.plot(timesteps, targets, 'g-', label='True', linewidth=2, alpha=0.8)
        ax.plot(timesteps, preds, 'r-', label='Pred', linewidth=2, alpha=0.8)
        
        # Add std for each predicted point only for categorical mode
        if model.mode == 'categorical' and dist_stds:
            preds_array = np.array(preds)
            stds_array = np.array(dist_stds)
            ax.fill_between(timesteps, preds_array - stds_array, preds_array + stds_array, 
                             alpha=0.3, color='red', label='±1 std')
        
        ax.set_xlabel('Timestep', fontsize=10)
        ax.set_ylabel('Progress', fontsize=10)
        ax.set_title(f'Episode {actual_ep_numbers[idx]}', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        
        # Calculate and display MAE and mean std
        mae = np.mean(np.abs(np.array(preds) - np.array(targets)))
        if model.mode == 'categorical' and dist_stds:
            mean_std = np.mean(dist_stds)
            ax.text(0.02, 0.98, f'MAE: {mae:.4f}\nσ: {mean_std:.4f}', transform=ax.transAxes,
                    fontsize=9, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            ax.text(0.02, 0.98, f'MAE: {mae:.4f}', transform=ax.transAxes,
                    fontsize=9, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        if idx == 0:
            ax.legend(fontsize=9)
    
    for idx in range(n_episodes, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")


def compute_state_stats(episodes):
    """Compute state normalization statistics from episodes"""
    all_states = []
    for ep in episodes:
        for step in ep:
            if 'state' in step:
                all_states.append(step['state'])
    all_states = np.asarray(all_states, dtype=np.float32)
    state_mean = torch.as_tensor(all_states.mean(axis=0), dtype=torch.float32)
    state_std = torch.as_tensor(all_states.std(axis=0), dtype=torch.float32)
    return state_mean, state_std


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Run inference with windowed progress predictor')
    parser.add_argument('-m', '--model_path', type=str, 
                       default='progress_predictor_best.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--mode', type=str, default='categorical',
                       choices=['categorical', 'regression'],
                       help='Model mode')
    parser.add_argument('--num_bins', type=int, default=50,
                       help='Number of bins for categorical mode')
    parser.add_argument('--window_length', type=int, default=4,
                       help='Window length L for windowed model')
    parser.add_argument('--visual_feat_dim', type=int, default=512,
                       help='Visual feature dimension for windowed model')
    parser.add_argument('--state_feat_dim', type=int, default=64,
                       help='State feature dimension for windowed model')
    parser.add_argument('--token_dim', type=int, default=256,
                       help='Token dimension for windowed model')
    parser.add_argument('--gru_hidden_dim', type=int, default=128,
                       help='GRU hidden dimension for windowed model')
    parser.add_argument('--dataset', type=str, default='pusht_episodes.pkl',
                       help='Path to dataset')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device (auto, cpu, cuda)')
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model from {args.model_path}...")
    model = load_model(
        model_path=args.model_path,
        mode=args.mode,
        num_bins=args.num_bins,
        window_length=args.window_length,
        visual_feat_dim=args.visual_feat_dim,
        state_feat_dim=args.state_feat_dim,
        token_dim=args.token_dim,
        gru_hidden_dim=args.gru_hidden_dim
    )
    
    device = args.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    print(f"Using device: {device}")
    
    # Load dataset
    print(f"Loading dataset from {args.dataset}...")
    with open(args.dataset, 'rb') as f:
        all_episodes = pickle.load(f)
    
    # training split
    train_ratio = 0.8
    split_idx = int(train_ratio * len(all_episodes))
    train_episodes = all_episodes[:split_idx]
    val_episodes = all_episodes[split_idx:]
    
    print(f"Total episodes: {len(all_episodes)}")
    print(f"Train episodes: {len(train_episodes)} (episodes 0 to {split_idx-1})")
    print(f"Validation episodes: {len(val_episodes)} (episodes {split_idx} to {len(all_episodes)-1})")
    
    # Compute state normalization stats from training set
    print("Computing state normalization statistics...")
    state_mean, state_std = compute_state_stats(train_episodes)
    
    # Progress over time plots for validation set (all episodes, 5 per PNG)
    print("\nGenerating progress over time plots for validation set...")
    episodes_per_page = 5
    n_val_pages = (len(val_episodes) + episodes_per_page - 1) // episodes_per_page
    
    for page_idx in range(n_val_pages):
        start_idx = page_idx * episodes_per_page
        end_idx = min(start_idx + episodes_per_page, len(val_episodes))
        ep_indices = list(range(start_idx, end_idx))
        
        # Adjust indices to match actual episode numbers in full dataset
        actual_ep_numbers = [split_idx + i for i in ep_indices]
        
        out_path = f'progress_over_time_val_page{page_idx+1}.png'
        plot_progress_grid(model, val_episodes, ep_indices=ep_indices, 
                          device=device, out_path=out_path, episodes_per_page=episodes_per_page,
                          actual_ep_numbers=actual_ep_numbers,
                          state_mean=state_mean, state_std=state_std)
    
    # Sample train episodes (5 total)
    print("\nGenerating progress over time plots for train set (sampled)...")
    n_train_samples = min(5, len(train_episodes))
    # Sample evenly spaced episodes from train set
    train_sample_indices = np.linspace(0, len(train_episodes)-1, n_train_samples, dtype=int).tolist()
    # Actual episode numbers in full dataset
    train_actual_ep_numbers = train_sample_indices  # Train episodes are 0 to split_idx-1
    
    plot_progress_grid(model, train_episodes, ep_indices=train_sample_indices,
                      device=device, out_path='progress_over_time_train.png', 
                      episodes_per_page=episodes_per_page,
                      actual_ep_numbers=train_actual_ep_numbers,
                      state_mean=state_mean, state_std=state_std)
    
    print("\nDone! Generated visualizations:")
    print(f"  - progress_over_time_val_page*.png ({n_val_pages} files, 5 episodes each)")
    print(f"  - progress_over_time_train.png (5 sampled train episodes)")


if __name__ == "__main__":
    main()
