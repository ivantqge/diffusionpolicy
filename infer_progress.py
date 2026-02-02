import torch
import numpy as np
from progress_predictor import ProgressPredictor
import pickle
import matplotlib.pyplot as plt


def load_model(model_path='progress_models/weights/progress_predictor_regress_best_frozen.pth', num_bins=50, pretrained=True, freeze_encoder=True, mode=None, action_chunk_horizon=0, action_dim=2, use_siamese=False):
    """
    Load model
    """
    
    model = ProgressPredictor(num_bins=num_bins, pretrained=pretrained, freeze_encoder=freeze_encoder, mode=mode, action_chunk_horizon=action_chunk_horizon, action_dim=action_dim, use_siamese=use_siamese)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    print(f"Loaded model in {mode} mode")
    print(f"Architecture: {'Siamese (two-tower)' if use_siamese else 'Single-tower'}")
    if action_chunk_horizon > 0:
        print(f"Action chunk horizon: {action_chunk_horizon}, Action dim: {action_dim}")
    return model


def extract_action_chunk(episode, step_idx, horizon, action_dim=2):
    """
    Extract action chunk from episode: future actions starting from step_idx
    Args:
        episode: List of episode steps
        step_idx: Current step index
        horizon: Number of future actions to extract
        action_dim: Dimension of action space
    Returns:
        action_chunk: numpy array [horizon, action_dim] or None if horizon == 0
        Contains: [action[t], action[t+1], ..., action[t+horizon-1]]
    """
    if horizon == 0:
        return None
    
    action_chunk_list = []
    episode_length = len(episode)
    
    for i in range(horizon):
        future_idx = step_idx + i
        if future_idx < episode_length and 'action' in episode[future_idx]:
            action_chunk_list.append(episode[future_idx]['action'])
        else:
            if action_chunk_list:
                action_chunk_list.append(action_chunk_list[-1])  # Repeat last action
            else:
                agent_pos = episode[step_idx]['state'][:action_dim].astype(np.float32)
                action_chunk_list.append(agent_pos)
    
    return np.array(action_chunk_list, dtype=np.float32)


def extract_repetitive_action_chunk(episode, step_idx, horizon, action_dim=2):
    """
    Extract repetitive action chunk: repeat current agent position horizon times
    Used for ablation study to test importance of action chunk information
    Args:
        episode: List of episode steps
        step_idx: Current step index
        horizon: Number of actions to repeat
        action_dim: Dimension of action space
    Returns:
        action_chunk: numpy array [horizon, action_dim] or None if horizon == 0
        Contains: [agent_pos, agent_pos, ..., agent_pos] (repeated horizon times)
    """
    if horizon == 0:
        return None
    
    step = episode[step_idx]
    agent_pos = step['state'][:action_dim].astype(np.float32)
    
    # Repeat agent position horizon times
    action_chunk = np.tile(agent_pos, (horizon, 1))
    
    return action_chunk


def predict(model, img, state, device='cpu', return_distribution=False, action_chunk=None, start_img=None):
    """
    Predict progress from image and state
    Args:
        model: ProgressPredictor model (categorical or regression mode)
        img: image array (current image)
        state: agent state array
        device: device to run on
        return_distribution: if True, also return probability distribution over bins (only for categorical mode)
        action_chunk: action chunk array [horizon, action_dim] or None
        start_img: start/initial image array (required if use_siamese=True)
    Returns:
        progress value [0, 1] (and optionally distribution [num_bins] for categorical mode)
    """
    def process_image(img_array):
        # Convert HWC to CHW if needed
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            img_array = img_array.transpose(2, 0, 1)  # HWC -> CHW
        img_array = img_array.astype(np.float32)
        
        # Normalize to [0, 1] first
        img_array = img_array / 255.0
        
        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
        img_array = (img_array - mean) / std
        return img_array
    
    img_processed = process_image(img)
    img_t = torch.FloatTensor(img_processed).unsqueeze(0).to(device)
    state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
    
    action_chunk_t = None
    if action_chunk is not None:
        action_chunk_t = torch.FloatTensor(action_chunk).unsqueeze(0).to(device)
    
    start_img_t = None
    if model.use_siamese:
        if start_img is None:
            raise ValueError("start_img is required when model.use_siamese=True")
        start_img_processed = process_image(start_img)
        start_img_t = torch.FloatTensor(start_img_processed).unsqueeze(0).to(device)
    
    with torch.no_grad():
        if model.use_siamese:
            output = model(img_t, state_t, action_chunk=action_chunk_t, start_image=start_img_t)
        else:
            output = model(img_t, state_t, action_chunk=action_chunk_t)
        
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


def visualize(model, episodes, ep_idx=0, device='cpu', n_cols=8, use_repetitive_actions=False):
    episode = episodes[ep_idx]
    start_img = episode[0]['img'] if model.use_siamese else None
    idxs = np.linspace(0, len(episode)-1, n_cols, dtype=int)
    
    preds, targets, imgs, distributions = [], [], [], []
    for i in idxs:
        step = episode[i]
        if use_repetitive_actions:
            action_chunk = extract_repetitive_action_chunk(episode, i, model.action_chunk_horizon, model.action_dim)
        else:
            action_chunk = extract_action_chunk(episode, i, model.action_chunk_horizon, model.action_dim)
        pred, dist = predict(model, step['img'], step['state'], device, return_distribution=True, action_chunk=action_chunk, start_img=start_img)
        preds.append(pred)
        targets.append(step['normalized_timestep'])
        imgs.append(step['img'])
        distributions.append(dist)
    
    fig, axes = plt.subplots(2, n_cols, figsize=(2*n_cols, 4))
    if n_cols == 1:
        axes = axes.reshape(2, 1)
    
    bin_centers = np.arange(0.5, model.num_bins) / model.num_bins
    
    for i, (img, true, pred, dist) in enumerate(zip(imgs, targets, preds, distributions)):
        if model.mode == 'categorical':
            mean_dist = np.sum(dist * bin_centers)
            variance = np.sum(dist * (bin_centers - mean_dist) ** 2)
            dist_std = np.sqrt(variance)
            std_text = f'\nσ:{dist_std:.3f}'
        else:
            std_text = ''  #
        
        # Normalize image for display (HWC float32 [0, 255] -> [0, 1])
        img_disp = np.clip(img / 255.0, 0, 1)
  
        # Top row: image
        axes[0, i].imshow(img_disp, cmap='gray' if len(img_disp.shape) == 2 else None)
        axes[0, i].text(0.5, 0.95, f'T:{true:.2f}\nP:{pred:.3f}{std_text}', 
                    transform=axes[0, i].transAxes, ha='center', va='top',
                    bbox=dict(boxstyle='round', alpha=0.8, facecolor='white'))
        axes[0, i].axis('off')
        
        # Bottom row: distribution
        axes[1, i].bar(bin_centers, dist, width=0.02, alpha=0.7, color='blue')
        axes[1, i].axvline(true, color='green', linestyle='--', linewidth=2, label='True')
        axes[1, i].axvline(pred, color='red', linestyle='--', linewidth=2, label='Pred')
        axes[1, i].set_xlim(0, 1)
        axes[1, i].set_ylim(0, max(dist) * 1.1)
        axes[1, i].set_xlabel('Progress')
        if i == 0:
            axes[1, i].set_ylabel('Prob')
            axes[1, i].legend(fontsize=8)
        axes[1, i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('progress_viz.png', dpi=150)
    print("Saved: progress_viz.png")


def visualize_grid(model, episodes, ep_indices=None, device='cpu', n_cols=8, out_path='progress_viz_grid.png', show_distribution=True, use_repetitive_actions=False):
    if ep_indices is None:
        ep_indices = list(range(min(5, len(episodes))))
    n_rows = len(ep_indices)
    
    if show_distribution:
        fig, axes = plt.subplots(n_rows * 2, n_cols, figsize=(2*n_cols, 4*n_rows))
        if n_rows == 1:
            axes = axes.reshape(2, n_cols)
    else:
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(2*n_cols, 2*n_rows))
        if n_rows == 1:
            axes = np.expand_dims(axes, 0)
        else:
            axes = axes.reshape(n_rows, n_cols)
    
    bin_centers = np.arange(0.5, model.num_bins) / model.num_bins

    for r, ep_idx in enumerate(ep_indices):
        episode = episodes[ep_idx]
        start_img = episode[0]['img'] if model.use_siamese else None
        idxs = np.linspace(0, len(episode)-1, n_cols, dtype=int)
        for c, i in enumerate(idxs):
            step = episode[i]
            if use_repetitive_actions:
                action_chunk = extract_repetitive_action_chunk(episode, i, model.action_chunk_horizon, model.action_dim)
            else:
                action_chunk = extract_action_chunk(episode, i, model.action_chunk_horizon, model.action_dim)
            pred, dist = predict(model, step['img'], step['state'], device, return_distribution=True, action_chunk=action_chunk, start_img=start_img)
            true = step['normalized_timestep']
            img = step['img']

            img_disp = img
            if not isinstance(img_disp, np.ndarray):
                img_disp = np.asarray(img_disp)
            if img_disp.dtype != np.uint8:
                img_disp = (img_disp / 255.0).clip(0.0, 1.0)

            if show_distribution:
                if model.mode == 'categorical':
                    mean_dist = np.sum(dist * bin_centers)
                    variance = np.sum(dist * (bin_centers - mean_dist) ** 2)
                    dist_std = np.sqrt(variance)
                    std_text = f'\nσ:{dist_std:.3f}'
                else:
                    std_text = ''  
                
                # Top row: image
                img_ax = axes[r*2, c]
                img_ax.imshow(img_disp, cmap='gray' if len(img_disp.shape) == 2 else None)
                img_ax.text(0.5, 0.95, f'T:{true:.2f}\nP:{pred:.3f}{std_text}', transform=img_ax.transAxes,
                        ha='center', va='top', bbox=dict(boxstyle='round', alpha=0.8, facecolor='white'))
                img_ax.axis('off')
                
                # Bottom row: distribution
                dist_ax = axes[r*2+1, c]
                dist_ax.bar(bin_centers, dist, width=0.02, alpha=0.7, color='blue')
                dist_ax.axvline(true, color='green', linestyle='--', linewidth=1.5, label='True')
                dist_ax.axvline(pred, color='red', linestyle='--', linewidth=1.5, label='Pred')
                dist_ax.set_xlim(0, 1)
                dist_ax.set_ylim(0, max(dist) * 1.1)
                dist_ax.set_xlabel('Progress', fontsize=8)
                if c == 0:
                    dist_ax.set_ylabel('Prob', fontsize=8)
                    dist_ax.legend(fontsize=6)
                dist_ax.grid(True, alpha=0.3)
                dist_ax.tick_params(labelsize=6)
            else:
                ax = axes[r, c]
                ax.imshow(img_disp, cmap='gray' if len(img_disp.shape) == 2 else None)
                ax.text(0.5, 0.95, f'T:{true:.2f}\nP:{pred:.3f}', transform=ax.transAxes,
                        ha='center', va='top', bbox=dict(boxstyle='round', alpha=0.8))
                ax.axis('off')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")


def plot_progress_over_time(model, episodes, ep_idx=0, device='cpu', out_path='progress_over_time.png', use_repetitive_actions=False):
    """
    Plot prediction vs real progress over the course of one episode with std bands
    """
    episode = episodes[ep_idx]
    start_img = episode[0]['img'] if model.use_siamese else None
    
    # Get predictions and distributions for all timesteps
    preds, targets, dist_stds = [], [], []
    bin_centers = np.arange(0.5, model.num_bins) / model.num_bins
    
    for step_idx, step in enumerate(episode):
        if use_repetitive_actions:
            action_chunk = extract_repetitive_action_chunk(episode, step_idx, model.action_chunk_horizon, model.action_dim)
        else:
            action_chunk = extract_action_chunk(episode, step_idx, model.action_chunk_horizon, model.action_dim)
        if model.mode == 'categorical':
            pred, dist = predict(model, step['img'], step['state'], device, return_distribution=True, action_chunk=action_chunk, start_img=start_img)

            # Compute stats
            mean_dist = np.sum(dist * bin_centers)
            variance = np.sum(dist * (bin_centers - mean_dist) ** 2)
            dist_std = np.sqrt(variance)
            dist_stds.append(dist_std)
        else:
            pred = predict(model, step['img'], step['state'], device, return_distribution=False, action_chunk=action_chunk, start_img=start_img)
        
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


def plot_progress_grid(model, episodes, ep_indices=None, device='cpu', out_path='progress_over_time_grid.png', episodes_per_page=5, actual_ep_numbers=None, use_repetitive_actions=False):
    """
    Plot prediction vs real progress over time for multiple episodes in a grid
    Always shows exactly episodes_per_page episodes (default 5) per PNG
    
    Args:
        actual_ep_numbers: Optional list of actual episode numbers for display (if None, uses ep_indices)
        use_repetitive_actions: If True, use repetitive action chunks (ablation study)
    """
    if ep_indices is None:
        ep_indices = list(range(min(episodes_per_page, len(episodes))))
    
    # Always use 5 episodes per page
    n_episodes = min(len(ep_indices), episodes_per_page)
    ep_indices = ep_indices[:n_episodes]  # Take only first 5
    
    # Use actual episode numbers for display if provided, otherwise use indices
    if actual_ep_numbers is None:
        actual_ep_numbers = ep_indices
    else:
        actual_ep_numbers = actual_ep_numbers[:n_episodes]
    
    n_cols = 3
    n_rows = 2  # 2 rows x 3 cols = 6, but we'll only use 5
    
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
        start_img = episode[0]['img'] if model.use_siamese else None
        
        # Get predictions and distributions for all timesteps
        preds, targets, dist_stds = [], [], []
        bin_centers = np.arange(0.5, model.num_bins) / model.num_bins
        
        for step_idx, step in enumerate(episode):
            if use_repetitive_actions:
                action_chunk = extract_repetitive_action_chunk(episode, step_idx, model.action_chunk_horizon, model.action_dim)
            else:
                action_chunk = extract_action_chunk(episode, step_idx, model.action_chunk_horizon, model.action_dim)
            if model.mode == 'categorical':
                pred, dist = predict(model, step['img'], step['state'], device, return_distribution=True, action_chunk=action_chunk, start_img=start_img)
                # Compute std of distribution
                mean_dist = np.sum(dist * bin_centers)
                variance = np.sum(dist * (bin_centers - mean_dist) ** 2)
                dist_std = np.sqrt(variance)
                dist_stds.append(dist_std)
            else:
                pred = predict(model, step['img'], step['state'], device, return_distribution=False, action_chunk=action_chunk, start_img=start_img)
                dist_stds.append(0.0)
                if model.mode == 'categorical':
                    dist_stds.append(0.0)
            
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
    
    # Hide unused subplots
    for idx in range(n_episodes, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")


def plot_bin_aggregation(model, episodes, device='cpu', num_bins=50, out_path='progress_bin_aggregation.png', use_repetitive_actions=False):
    """
    Aggregate predictions by true progress bins across all episodes.
    For each bin, calculate mean and std of predicted values.
    
    Args:
        model: ProgressPredictor model
        episodes: List of episodes
        device: Device to run on
        num_bins: Number of bins to use (50 for both categorical and regression)
        out_path: Output path for the plot
        use_repetitive_actions: If True, use repetitive action chunks (ablation study)
    """
    # Collect all predictions and true values across all episodes
    all_preds = []
    all_targets = []
    
    print("Collecting predictions across all episodes...")
    for ep_idx, episode in enumerate(episodes):
        start_img = episode[0]['img'] if model.use_siamese else None
        for step_idx, step in enumerate(episode):
            true_progress = step['normalized_timestep']
            if use_repetitive_actions:
                action_chunk = extract_repetitive_action_chunk(episode, step_idx, model.action_chunk_horizon, model.action_dim)
            else:
                action_chunk = extract_action_chunk(episode, step_idx, model.action_chunk_horizon, model.action_dim)
            pred = predict(model, step['img'], step['state'], device, return_distribution=False, action_chunk=action_chunk, start_img=start_img)
            all_preds.append(pred)
            all_targets.append(true_progress)
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    print(f"Collected {len(all_preds)} predictions")
    
    # Create bins based on true progress (ground truth bins)
    # For both categorical and regression, use num_bins bins
    bin_edges = np.linspace(0, 1, num_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Bin the data by true progress
    bin_indices = np.digitize(all_targets, bin_edges) - 1
    # Handle edge case: values exactly at 1.0 go to last bin
    bin_indices = np.clip(bin_indices, 0, num_bins - 1)
    
    # Calculate mean and std for each bin
    bin_means = []
    bin_stds = []
    bin_counts = []
    
    for bin_idx in range(num_bins):
        mask = (bin_indices == bin_idx)
        if np.any(mask):
            bin_preds = all_preds[mask]
            bin_means.append(np.mean(bin_preds))
            bin_stds.append(np.std(bin_preds))
            bin_counts.append(np.sum(mask))
        else:
            bin_means.append(np.nan)
            bin_stds.append(np.nan)
            bin_counts.append(0)
    
    bin_means = np.array(bin_means)
    bin_stds = np.array(bin_stds)
    bin_counts = np.array(bin_counts)
    
    # Create visualization
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    # Mean prediction per bin with error bars
    valid_mask = ~np.isnan(bin_means)
    ax.errorbar(bin_centers[valid_mask], bin_means[valid_mask], yerr=bin_stds[valid_mask],
                 fmt='o-', capsize=3, capthick=1.5, linewidth=2, markersize=6,
                 label='Mean ± Std', color='blue', alpha=0.7)
    # Add diagonal line (perfect prediction)
    ax.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect Prediction', alpha=0.5)
    ax.set_xlabel('True Progress Bin Center', fontsize=12)
    ax.set_ylabel('Mean Predicted Progress', fontsize=12)
    ax.set_title(f'Mean and Std of Predictions per True Progress Bin (n={len(all_preds)} predictions)', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    # Add text with summary statistics
    overall_mae = np.mean(np.abs(all_preds - all_targets))
    overall_std = np.std(all_preds - all_targets)
    mean_bin_std = np.nanmean(bin_stds[valid_mask])
    
    stats_text = f'Overall MAE: {overall_mae:.4f}\nOverall Std Error: {overall_std:.4f}\nMean Bin Std: {mean_bin_std:.4f}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Run inference with progress predictor')
    parser.add_argument('-m', '--model_path', type=str, 
                       default='progress_predictor_best.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--mode', type=str, default='categorical',
                       choices=['categorical', 'regression'],
                       help='Model mode')
    parser.add_argument('--action_chunk_horizon', type=int, default=8,
                       help='Action chunk horizon (must match training)')
    parser.add_argument('--action_dim', type=int, default=2,
                       help='Action dimension')
    parser.add_argument('--num_bins', type=int, default=50,
                       help='Number of bins for categorical mode')
    parser.add_argument('--use_siamese', action='store_true', default=True,
                       help='Use Siamese two-tower encoder (default: True)')
    parser.add_argument('--no_siamese', dest='use_siamese', action='store_false',
                       help='Disable Siamese architecture (use single-tower)')
    parser.add_argument('--dataset', type=str, default='pusht_episodes.pkl',
                       help='Path to dataset')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device (auto, cpu, cuda)')
    parser.add_argument('--use_repetitive_actions', action='store_true',
                       help='Use repetitive action chunks (8x current agent position) for ablation study')
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model from {args.model_path}...")
    model = load_model(
        model_path=args.model_path,
        mode=args.mode,
        action_chunk_horizon=args.action_chunk_horizon,
        action_dim=args.action_dim,
        num_bins=args.num_bins,
        use_siamese=args.use_siamese
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
    
    if args.use_repetitive_actions:
        print("\n ABLATION MODE: Using repetitive action chunks (8x current agent position)")
        suffix = "_ablation"
    else:
        print("\n Using real future action chunks")
        suffix = ""
    
    # Progress over time plots for validation set (all episodes, 5 per PNG)
    print("\n Generating progress over time plots for validation set...")
    episodes_per_page = 5
    n_val_pages = (len(val_episodes) + episodes_per_page - 1) // episodes_per_page
    
    for page_idx in range(n_val_pages):
        start_idx = page_idx * episodes_per_page
        end_idx = min(start_idx + episodes_per_page, len(val_episodes))
        ep_indices = list(range(start_idx, end_idx))
        
        # Adjust indices to match actual episode numbers in full dataset
        actual_ep_numbers = [split_idx + i for i in ep_indices]
        
        out_path = f'progress_over_time_val_page{page_idx+1}{suffix}.png'
        plot_progress_grid(model, val_episodes, ep_indices=ep_indices, 
                          device=device, out_path=out_path, episodes_per_page=episodes_per_page,
                          actual_ep_numbers=actual_ep_numbers,
                          use_repetitive_actions=args.use_repetitive_actions)
    
    # Sample train episodes (5 total)
    print("\nGenerating progress over time plots for train set (sampled)...")
    n_train_samples = min(5, len(train_episodes))
    # Sample evenly spaced episodes from train set
    train_sample_indices = np.linspace(0, len(train_episodes)-1, n_train_samples, dtype=int).tolist()
    # Actual episode numbers in full dataset
    train_actual_ep_numbers = train_sample_indices  # Train episodes are 0 to split_idx-1
    
    plot_progress_grid(model, train_episodes, ep_indices=train_sample_indices,
                      device=device, out_path=f'progress_over_time_train{suffix}.png', 
                      episodes_per_page=episodes_per_page,
                      actual_ep_numbers=train_actual_ep_numbers,
                      use_repetitive_actions=args.use_repetitive_actions)
    
    print("\nDone! Generated visualizations:")
    print(f"  - progress_over_time_val_page*{suffix}.png ({n_val_pages} files, 5 episodes each)")
    print(f"  - progress_over_time_train{suffix}.png (5 sampled train episodes)")


if __name__ == "__main__":
    main()

