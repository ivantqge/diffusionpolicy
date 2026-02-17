"""
Inference script for dual-head progress predictor models.

This script handles evaluation and visualization for the dual-head
progress predictor (push rate + reposition rate).
"""

import torch
import numpy as np
from progress_predictor_dual_head import ProgressPredictorDualHead
import pickle
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def load_model(model_path='progress_predictor_best.pth', 
               visual_feat_dim=512, state_feat_dim=64, 
               token_dim=256, gru_hidden_dim=128, window_length=8,
               pretrained=True, freeze_encoder=True, dropout=0.1):
    """
    Load dual-head progress predictor model.
    """
    model = ProgressPredictorDualHead(
        agent_state_dim=5,
        pretrained=pretrained,
        freeze_encoder=freeze_encoder,
        encoder_ckpt_path=None,
        visual_feat_dim=visual_feat_dim,
        state_feat_dim=state_feat_dim,
        token_dim=token_dim,
        gru_hidden_dim=gru_hidden_dim,
        dropout=dropout
    )
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    model.window_length = window_length
    print(f"Loaded dual-head model from {model_path}")
    print(f"Architecture: Dual-Head (Push + Reposition), Window L={window_length}")
    return model


def predict_rates(model, episode, step_idx, device='cpu', 
                  state_mean=None, state_std=None):
    """
    Predict push and reposition rates using dual-head model.
    
    Args:
        model: ProgressPredictorDualHead model
        episode: List of episode steps
        step_idx: Current step index (target frame)
        device: device to run on
        state_mean: State normalization mean
        state_std: State normalization std
    
    Returns:
        push_rate: predicted push rate
        repo_rate: predicted reposition rate
    """
    def process_image(img_array):
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            img_array = img_array.transpose(2, 0, 1)  # HWC -> CHW
        img_array = img_array.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
        img_array = (img_array - mean) / std
        return img_array
    
    L = model.window_length
    episode_length = len(episode)
    
    # Build window indices (same logic as training)
    if step_idx <= 0:
        window_indices = [0] * L
    elif step_idx < L - 1:
        window_indices = [0] + [min(i, episode_length - 1) for i in range(1, L)]
    else:
        stride_choices = [2, 4, 6]
        s = None
        for cand in reversed(stride_choices):
            if step_idx - (L - 2) * cand >= 1:
                s = cand
                break
        if s is None:
            s = 1
        ta = step_idx - (L - 2) * s
        window_indices = [0] + [ta + i * s for i in range(L - 1)]
    
    window_indices = [min(max(idx, 0), episode_length - 1) for idx in window_indices]
    
    # Get images and states
    images_list = []
    states_list = []
    
    for idx in window_indices:
        step = episode[idx]
        img_processed = process_image(step['img'])
        state = step['state'].astype(np.float32)
        
        if state_mean is not None and state_std is not None:
            if isinstance(state_mean, torch.Tensor):
                state = (state - state_mean.numpy()) / (state_std.numpy() + 1e-6)
            else:
                state = (state - state_mean) / (state_std + 1e-6)
        
        images_list.append(img_processed)
        states_list.append(state)
    
    images_seq = np.stack(images_list, axis=0)
    states_seq = np.stack(states_list, axis=0)
    
    images_t = torch.FloatTensor(images_seq).unsqueeze(0).to(device)
    states_t = torch.FloatTensor(states_seq).unsqueeze(0).to(device)
    
    with torch.no_grad():
        push_rate, repo_rate = model(images_t, states_t)
        return push_rate.item(), repo_rate.item()


def compute_state_stats(episodes):
    """Compute state normalization statistics from episodes."""
    all_states = []
    for ep in episodes:
        for step in ep:
            if 'state' in step:
                all_states.append(step['state'])
    all_states = np.asarray(all_states, dtype=np.float32)
    state_mean = torch.as_tensor(all_states.mean(axis=0), dtype=torch.float32)
    state_std = torch.as_tensor(all_states.std(axis=0), dtype=torch.float32)
    return state_mean, state_std


def compute_global_progress(model, episode, device='cpu', state_mean=None, state_std=None,
                            stats=None, scale_factor=None, push_weight=1.0, repo_weight=0.3,
                            causal=True):
    """
    Compute global progress [0, 1] by integrating predicted rates over time.
    
    Args:
        model: ProgressPredictorDualHead model
        episode: List of episode steps (only used up to current timestep)
        device: device to run on
        state_mean: State normalization mean
        state_std: State normalization std
        stats: Rate normalization statistics (for denormalization and scale_factor)
        scale_factor: Fixed normalization factor (if None, estimated from stats when causal=True)
        push_weight: Weight for push rate contribution
        repo_weight: Weight for repo rate contribution
        causal: If True, use fixed scale_factor (no future knowledge needed).
                If False, normalize by final cumulative value (requires full episode).
    
    Returns:
        progress_values: array of progress values [0, 1] for each timestep
        push_rates: array of predicted push rates (denormalized)
        repo_rates: array of predicted repo rates (denormalized)
        is_pushing: array of phase labels
    """
    push_rates = []
    repo_rates = []
    is_pushing_list = []
    
    # Collect predictions for all timesteps
    for step_idx, step in enumerate(episode):
        push_pred, repo_pred = predict_rates(model, episode, step_idx, device,
                                             state_mean=state_mean, state_std=state_std)
        push_rates.append(push_pred)
        repo_rates.append(repo_pred)
        is_pushing_list.append(step.get('is_pushing', True))
    
    push_rates = np.array(push_rates)
    repo_rates = np.array(repo_rates)
    is_pushing = np.array(is_pushing_list)
    
    # Denormalize rates if stats provided
    if stats is not None:
        push_mean = stats.get('push_rate_mean', 0.0)
        push_std = stats.get('push_rate_std', 1.0)
        repo_mean = stats.get('repo_rate_mean', 0.0)
        repo_std = stats.get('repo_rate_std', 1.0)
        
        push_rates_denorm = push_rates * push_std + push_mean
        repo_rates_denorm = repo_rates * repo_std + repo_mean
    else:
        push_rates_denorm = push_rates
        repo_rates_denorm = repo_rates
    
    # Compute effective rate at each timestep based on phase
    effective_rates = np.where(is_pushing, 
                               push_weight * push_rates_denorm,
                               repo_weight * repo_rates_denorm)
    
    # Cumulative sum of all rates (negative rates decrease progress)
    cumulative_progress = np.cumsum(effective_rates)
    
    if causal:
        # Use fixed scale factor for CAUSAL normalization (no future knowledge)
        if scale_factor is None:
            # Estimate scale factor from stats if available
            if stats is not None:
                avg_episode_length = stats.get('avg_episode_length', 100)
                avg_push_rate = max(stats.get('push_rate_mean', 0.0), 0.01)
                avg_repo_rate = max(stats.get('repo_rate_mean', 0.0), 0.01)
                push_fraction = stats.get('push_fraction', 0.5)
                
                # Expected total = avg_length * (push_frac * push_rate + repo_frac * repo_rate)
                scale_factor = avg_episode_length * (
                    push_fraction * push_weight * avg_push_rate +
                    (1 - push_fraction) * repo_weight * avg_repo_rate
                )
                scale_factor = max(scale_factor, 1.0)  # Avoid division by zero
            else:
                # Fallback: use a reasonable default
                scale_factor = 100.0
        
        progress_values = np.clip(cumulative_progress / scale_factor, 0.0, 1.0)
    else:
        # Non-causal: normalize by final cumulative value (requires full episode)
        final_value = cumulative_progress[-1]
        if abs(final_value) > 1e-6:
            progress_values = cumulative_progress / final_value
            # Shift so minimum is 0, maximum is 1
            min_val = progress_values.min()
            max_val = progress_values.max()
            if max_val - min_val > 1e-6:
                progress_values = (progress_values - min_val) / (max_val - min_val)
            else:
                progress_values = np.linspace(0, 1, len(episode))
        else:
            # Fallback to linear if no net progress
            progress_values = np.linspace(0, 1, len(episode))
    
    return progress_values, push_rates_denorm, repo_rates_denorm, is_pushing


def compute_global_progress_online(model, episode, step_idx, device='cpu', 
                                   state_mean=None, state_std=None, stats=None,
                                   scale_factor=None, push_weight=1.0, repo_weight=0.3,
                                   cumulative_so_far=0.0):
    """
    Compute global progress for a SINGLE timestep in an online/streaming fashion.
    
    This is the function to use during real-time inference where you process
    one frame at a time and don't have access to future frames.
    
    Args:
        model: ProgressPredictorDualHead model
        episode: List of episode steps (only steps 0 to step_idx are used)
        step_idx: Current timestep to predict
        device: device to run on
        state_mean, state_std: State normalization
        stats: Rate statistics for denormalization
        scale_factor: Fixed normalization factor
        push_weight, repo_weight: Weights for rate contributions
        cumulative_so_far: Running sum of progress from previous steps
    
    Returns:
        progress: Current progress value [0, 1]
        cumulative: Updated cumulative sum (pass to next call)
        push_rate: Predicted push rate (denormalized)
        repo_rate: Predicted repo rate (denormalized)
        is_pushing: Phase label for current step
    """
    # Get prediction for current timestep
    push_pred, repo_pred = predict_rates(model, episode, step_idx, device,
                                         state_mean=state_mean, state_std=state_std)
    
    # Get phase label
    is_pushing = episode[step_idx].get('is_pushing', True)
    
    # Denormalize rates
    if stats is not None:
        push_rate = push_pred * stats.get('push_rate_std', 1.0) + stats.get('push_rate_mean', 0.0)
        repo_rate = repo_pred * stats.get('repo_rate_std', 1.0) + stats.get('repo_rate_mean', 0.0)
    else:
        push_rate = push_pred
        repo_rate = repo_pred
    
    # Compute effective rate based on phase
    if is_pushing:
        effective_rate = push_weight * push_rate
    else:
        effective_rate = repo_weight * repo_rate
    
    # Allow negative rates to decrease progress
    cumulative = cumulative_so_far + effective_rate
    
    # Determine scale factor
    if scale_factor is None:
        if stats is not None:
            avg_episode_length = stats.get('avg_episode_length', 100)
            avg_push_rate = max(stats.get('push_rate_mean', 0.0), 0.01)
            avg_repo_rate = max(stats.get('repo_rate_mean', 0.0), 0.01)
            push_fraction = stats.get('push_fraction', 0.5)
            scale_factor = avg_episode_length * (
                push_fraction * push_weight * avg_push_rate +
                (1 - push_fraction) * repo_weight * avg_repo_rate
            )
            scale_factor = max(scale_factor, 1.0)
        else:
            scale_factor = 100.0
    
    # Compute progress (clamped to [0, 1])
    progress = max(0.0, min(cumulative / scale_factor, 1.0))
    
    return progress, cumulative, push_rate, repo_rate, is_pushing


def compute_ground_truth_progress(episode, stats=None, scale_factor=None, 
                                  push_weight=1.0, repo_weight=0.3, causal=True):
    """
    Compute ground truth global progress from rate labels.
    
    Args:
        episode: List of episode steps with 'push_rate', 'repo_rate', 'is_pushing'
        stats: Rate statistics (for scale_factor estimation)
        scale_factor: Fixed normalization factor
        push_weight, repo_weight: Weights for rate contributions
        causal: If True, use fixed scale_factor (no future knowledge needed).
                If False, normalize by final cumulative value (requires full episode).
    
    Returns:
        progress_values: array of progress values [0, 1] for each timestep
    """
    push_rates = []
    repo_rates = []
    is_pushing_list = []
    
    for step in episode:
        # Use raw rates (not normalized) for ground truth
        push_rates.append(step.get('push_rate', 0.0))
        repo_rates.append(step.get('repo_rate', 0.0))
        is_pushing_list.append(step.get('is_pushing', True))
    
    push_rates = np.array(push_rates)
    repo_rates = np.array(repo_rates)
    is_pushing = np.array(is_pushing_list)
    
    # Compute effective rates
    effective_rates = np.where(is_pushing,
                               push_weight * push_rates,
                               repo_weight * repo_rates)
    
    # Cumulative sum of all rates (negative rates decrease progress)
    cumulative_progress = np.cumsum(effective_rates)
    
    if causal:
        # Use fixed scale factor (causal - no future knowledge)
        if scale_factor is None:
            if stats is not None:
                avg_episode_length = stats.get('avg_episode_length', 100)
                avg_push_rate = max(stats.get('push_rate_mean', 0.0), 0.01)
                avg_repo_rate = max(stats.get('repo_rate_mean', 0.0), 0.01)
                push_fraction = stats.get('push_fraction', 0.5)
                scale_factor = avg_episode_length * (
                    push_fraction * push_weight * avg_push_rate +
                    (1 - push_fraction) * repo_weight * avg_repo_rate
                )
                scale_factor = max(scale_factor, 1.0)
            else:
                scale_factor = 100.0
        
        progress_values = np.clip(cumulative_progress / scale_factor, 0.0, 1.0)
    else:
        # Non-causal: normalize by final cumulative value (requires full episode)
        final_value = cumulative_progress[-1]
        if abs(final_value) > 1e-6:
            progress_values = cumulative_progress / final_value
            # Shift so minimum is 0, maximum is 1
            min_val = progress_values.min()
            max_val = progress_values.max()
            if max_val - min_val > 1e-6:
                progress_values = (progress_values - min_val) / (max_val - min_val)
            else:
                progress_values = np.linspace(0, 1, len(episode))
        else:
            progress_values = np.linspace(0, 1, len(episode))
    
    return progress_values


def plot_global_progress(model, episodes, ep_idx=0, device='cpu',
                         out_path='global_progress.png',
                         state_mean=None, state_std=None, stats=None, causal=True):
    """
    Plot global progress prediction vs ground truth for one episode.
    Shows both the integrated progress and the underlying rates.
    """
    episode = episodes[ep_idx]
    
    # Compute predicted global progress
    pred_progress, push_rates, repo_rates, is_pushing = compute_global_progress(
        model, episode, device, state_mean, state_std, stats, causal=causal
    )
    
    # Compute ground truth global progress from rate labels
    gt_progress = compute_ground_truth_progress(episode, stats, causal=causal)
    
    # Also get the simple normalized timestep as baseline
    baseline_progress = np.array([step.get('normalized_timestep', i / max(1, len(episode) - 1)) 
                                  for i, step in enumerate(episode)])
    
    timesteps = np.arange(len(episode))
    
    # Create figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    # Plot 1: Global Progress
    ax1.plot(timesteps, baseline_progress, 'k--', label='Linear (baseline)', linewidth=1.5, alpha=0.5)
    ax1.plot(timesteps, gt_progress, 'g-', label='GT Progress (from rates)', linewidth=2, alpha=0.8)
    ax1.plot(timesteps, pred_progress, 'b-', label='Predicted Progress', linewidth=2, alpha=0.8)
    
    # Shade phases
    for i in range(len(timesteps)):
        if is_pushing[i]:
            ax1.axvspan(i - 0.5, i + 0.5, alpha=0.1, color='green')
        else:
            ax1.axvspan(i - 0.5, i + 0.5, alpha=0.1, color='red')
    
    ax1.set_ylabel('Global Progress [0, 1]', fontsize=12)
    ax1.set_title(f'Global Progress Over Time - Episode {ep_idx}', fontsize=14)
    ax1.legend(fontsize=10, loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.05, 1.05)
    
    # Compute progress MAE
    progress_mae = np.mean(np.abs(pred_progress - gt_progress))
    ax1.text(0.02, 0.98, f'Progress MAE: {progress_mae:.4f}', transform=ax1.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Plot 2: Push Rates (during pushing phases)
    # Note: push_rates from compute_global_progress are already denormalized
    gt_push_rates = np.array([step.get('push_rate', 0.0) for step in episode])
    ax2.plot(timesteps, gt_push_rates, 'g-', label='GT Push Rate', linewidth=1.5, alpha=0.7)
    ax2.plot(timesteps, push_rates, 'b-', label='Pred Push Rate', linewidth=1.5, alpha=0.7)
    
    for i in range(len(timesteps)):
        if is_pushing[i]:
            ax2.axvspan(i - 0.5, i + 0.5, alpha=0.15, color='green')
    
    ax2.set_ylabel('Push Rate', fontsize=12)
    ax2.set_title('Push Rates (green = pushing phase)', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    # Plot 3: Repo Rates (during repositioning phases)
    # Note: repo_rates from compute_global_progress are already denormalized
    gt_repo_rates = np.array([step.get('repo_rate', 0.0) for step in episode])
    ax3.plot(timesteps, gt_repo_rates, 'g-', label='GT Repo Rate', linewidth=1.5, alpha=0.7)
    ax3.plot(timesteps, repo_rates, 'r-', label='Pred Repo Rate', linewidth=1.5, alpha=0.7)
    
    for i in range(len(timesteps)):
        if not is_pushing[i]:
            ax3.axvspan(i - 0.5, i + 0.5, alpha=0.15, color='red')
    
    ax3.set_xlabel('Timestep', fontsize=12)
    ax3.set_ylabel('Repo Rate', fontsize=12)
    ax3.set_title('Reposition Rates (red = repositioning phase)', fontsize=12)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    # Add legend for phase shading
    legend_elements = [
        Patch(facecolor='green', alpha=0.15, label='Pushing Phase'),
        Patch(facecolor='red', alpha=0.15, label='Repositioning Phase')
    ]
    fig.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")


def plot_global_progress_grid(model, episodes, ep_indices=None, device='cpu',
                              out_path='global_progress_grid.png', episodes_per_page=4,
                              actual_ep_numbers=None, state_mean=None, state_std=None, stats=None,
                              causal=True):
    """
    Plot global progress for multiple episodes in a grid.
    """
    if ep_indices is None:
        ep_indices = list(range(min(episodes_per_page, len(episodes))))
    
    n_episodes = min(len(ep_indices), episodes_per_page)
    ep_indices = ep_indices[:n_episodes]
    
    if actual_ep_numbers is None:
        actual_ep_numbers = ep_indices
    else:
        actual_ep_numbers = actual_ep_numbers[:n_episodes]
    
    n_cols = 2
    n_rows = (n_episodes + 1) // 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx, ep_idx in enumerate(ep_indices):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        
        episode = episodes[ep_idx]
        
        # Compute progress
        pred_progress, _, _, is_pushing = compute_global_progress(
            model, episode, device, state_mean, state_std, stats, causal=causal
        )
        gt_progress = compute_ground_truth_progress(episode, stats, causal=causal)
        baseline = np.array([step.get('normalized_timestep', i / max(1, len(episode) - 1))
                            for i, step in enumerate(episode)])
        
        timesteps = np.arange(len(episode))
        
        # Plot
        ax.plot(timesteps, baseline, 'k--', label='Linear', linewidth=1, alpha=0.4)
        ax.plot(timesteps, gt_progress, 'g-', label='GT', linewidth=2, alpha=0.8)
        ax.plot(timesteps, pred_progress, 'b-', label='Pred', linewidth=2, alpha=0.8)
        
        # Shade phases
        for i in range(len(timesteps)):
            if is_pushing[i]:
                ax.axvspan(i - 0.5, i + 0.5, alpha=0.1, color='green')
            else:
                ax.axvspan(i - 0.5, i + 0.5, alpha=0.1, color='red')
        
        ax.set_ylabel('Progress', fontsize=10)
        ax.set_title(f'Episode {actual_ep_numbers[idx]}', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)
        
        if idx == 0:
            ax.legend(fontsize=9)
        
        # MAE
        mae = np.mean(np.abs(pred_progress - gt_progress))
        ax.text(0.02, 0.98, f'MAE: {mae:.3f}', transform=ax.transAxes,
               fontsize=9, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        if row == n_rows - 1:
            ax.set_xlabel('Timestep', fontsize=10)
    
    # Hide empty subplots
    for idx in range(n_episodes, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")


def plot_rates_over_time(model, episodes, ep_idx=0, device='cpu', 
                         out_path='rates_over_time.png',
                         state_mean=None, state_std=None, stats=None):
    """
    Plot predicted vs actual rates over time for one episode.
    Shows push rate during pushing phases and repo rate during repositioning.
    """
    episode = episodes[ep_idx]
    
    # Get predictions and ground truth
    push_preds, push_targets = [], []
    repo_preds, repo_targets = [], []
    is_pushing_list = []
    
    for step_idx, step in enumerate(episode):
        push_pred, repo_pred = predict_rates(model, episode, step_idx, device,
                                             state_mean=state_mean, state_std=state_std)
        
        is_pushing = step.get('is_pushing', True)
        is_pushing_list.append(is_pushing)
        
        # Get ground truth (use normalized if available)
        if 'push_rate_normalized' in step:
            push_target = step['push_rate_normalized']
            repo_target = step['repo_rate_normalized']
        else:
            push_target = step.get('push_rate', 0.0)
            repo_target = step.get('repo_rate', 0.0)
        
        push_preds.append(push_pred)
        push_targets.append(push_target)
        repo_preds.append(repo_pred)
        repo_targets.append(repo_target)
    
    timesteps = np.arange(len(episode))
    is_pushing_arr = np.array(is_pushing_list)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Plot push rates
    ax1.plot(timesteps, push_targets, 'g-', label='Target Push Rate', linewidth=2, alpha=0.7)
    ax1.plot(timesteps, push_preds, 'b-', label='Predicted Push Rate', linewidth=2, alpha=0.7)
    
    # Shade pushing phases
    for i in range(len(timesteps)):
        if is_pushing_arr[i]:
            ax1.axvspan(i - 0.5, i + 0.5, alpha=0.2, color='green')
    
    ax1.set_ylabel('Push Rate', fontsize=12)
    ax1.set_title(f'Push Rate Over Time - Episode {ep_idx}', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    # Compute push MAE (only during pushing)
    push_mask = is_pushing_arr
    if push_mask.sum() > 0:
        push_mae = np.mean(np.abs(np.array(push_preds)[push_mask] - np.array(push_targets)[push_mask]))
        ax1.text(0.02, 0.98, f'Push MAE: {push_mae:.4f}', transform=ax1.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Plot reposition rates
    ax2.plot(timesteps, repo_targets, 'g-', label='Target Repo Rate', linewidth=2, alpha=0.7)
    ax2.plot(timesteps, repo_preds, 'r-', label='Predicted Repo Rate', linewidth=2, alpha=0.7)
    
    # Shade repositioning phases
    for i in range(len(timesteps)):
        if not is_pushing_arr[i]:
            ax2.axvspan(i - 0.5, i + 0.5, alpha=0.2, color='red')
    
    ax2.set_xlabel('Timestep', fontsize=12)
    ax2.set_ylabel('Reposition Rate', fontsize=12)
    ax2.set_title(f'Reposition Rate Over Time - Episode {ep_idx}', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    # Compute repo MAE (only during repositioning)
    repo_mask = ~is_pushing_arr
    if repo_mask.sum() > 0:
        repo_mae = np.mean(np.abs(np.array(repo_preds)[repo_mask] - np.array(repo_targets)[repo_mask]))
        ax2.text(0.02, 0.98, f'Repo MAE: {repo_mae:.4f}', transform=ax2.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Add legend for phase shading
    legend_elements = [
        Patch(facecolor='green', alpha=0.2, label='Pushing Phase'),
        Patch(facecolor='red', alpha=0.2, label='Repositioning Phase')
    ]
    fig.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")


def plot_rates_grid(model, episodes, ep_indices=None, device='cpu', 
                    out_path='rates_grid.png', episodes_per_page=4,
                    actual_ep_numbers=None, state_mean=None, state_std=None):
    """
    Plot rates over time for multiple episodes in a grid.
    """
    if ep_indices is None:
        ep_indices = list(range(min(episodes_per_page, len(episodes))))
    
    n_episodes = min(len(ep_indices), episodes_per_page)
    ep_indices = ep_indices[:n_episodes]
    
    if actual_ep_numbers is None:
        actual_ep_numbers = ep_indices
    else:
        actual_ep_numbers = actual_ep_numbers[:n_episodes]
    
    n_cols = 2
    n_rows = n_episodes
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
    if n_episodes == 1:
        axes = axes.reshape(1, -1)
    
    for idx, ep_idx in enumerate(ep_indices):
        ax_push = axes[idx, 0]
        ax_repo = axes[idx, 1]
        
        episode = episodes[ep_idx]
        
        push_preds, push_targets = [], []
        repo_preds, repo_targets = [], []
        is_pushing_list = []
        
        for step_idx, step in enumerate(episode):
            push_pred, repo_pred = predict_rates(model, episode, step_idx, device,
                                                 state_mean=state_mean, state_std=state_std)
            
            is_pushing = step.get('is_pushing', True)
            is_pushing_list.append(is_pushing)
            
            if 'push_rate_normalized' in step:
                push_target = step['push_rate_normalized']
                repo_target = step['repo_rate_normalized']
            else:
                push_target = step.get('push_rate', 0.0)
                repo_target = step.get('repo_rate', 0.0)
            
            push_preds.append(push_pred)
            push_targets.append(push_target)
            repo_preds.append(repo_pred)
            repo_targets.append(repo_target)
        
        timesteps = np.arange(len(episode))
        is_pushing_arr = np.array(is_pushing_list)
        
        # Push rate subplot
        ax_push.plot(timesteps, push_targets, 'g-', label='Target', linewidth=1.5, alpha=0.7)
        ax_push.plot(timesteps, push_preds, 'b-', label='Pred', linewidth=1.5, alpha=0.7)
        for i in range(len(timesteps)):
            if is_pushing_arr[i]:
                ax_push.axvspan(i - 0.5, i + 0.5, alpha=0.15, color='green')
        ax_push.set_ylabel('Push Rate', fontsize=10)
        ax_push.set_title(f'Episode {actual_ep_numbers[idx]} - Push', fontsize=11)
        ax_push.grid(True, alpha=0.3)
        ax_push.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        if idx == 0:
            ax_push.legend(fontsize=9)
        
        push_mask = is_pushing_arr
        if push_mask.sum() > 0:
            push_mae = np.mean(np.abs(np.array(push_preds)[push_mask] - np.array(push_targets)[push_mask]))
            ax_push.text(0.02, 0.98, f'MAE: {push_mae:.3f}', transform=ax_push.transAxes,
                        fontsize=9, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Repo rate subplot
        ax_repo.plot(timesteps, repo_targets, 'g-', label='Target', linewidth=1.5, alpha=0.7)
        ax_repo.plot(timesteps, repo_preds, 'r-', label='Pred', linewidth=1.5, alpha=0.7)
        for i in range(len(timesteps)):
            if not is_pushing_arr[i]:
                ax_repo.axvspan(i - 0.5, i + 0.5, alpha=0.15, color='red')
        ax_repo.set_ylabel('Repo Rate', fontsize=10)
        ax_repo.set_title(f'Episode {actual_ep_numbers[idx]} - Reposition', fontsize=11)
        ax_repo.grid(True, alpha=0.3)
        ax_repo.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        if idx == 0:
            ax_repo.legend(fontsize=9)
        
        repo_mask = ~is_pushing_arr
        if repo_mask.sum() > 0:
            repo_mae = np.mean(np.abs(np.array(repo_preds)[repo_mask] - np.array(repo_targets)[repo_mask]))
            ax_repo.text(0.02, 0.98, f'MAE: {repo_mae:.3f}', transform=ax_repo.transAxes,
                        fontsize=9, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        if idx == n_episodes - 1:
            ax_push.set_xlabel('Timestep', fontsize=10)
            ax_repo.set_xlabel('Timestep', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Run inference with dual-head progress predictor')
    parser.add_argument('-m', '--model_path', type=str, 
                       default='progress_predictor_best.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--window_length', type=int, default=8,
                       help='Window length L')
    parser.add_argument('--visual_feat_dim', type=int, default=512,
                       help='Visual feature dimension')
    parser.add_argument('--state_feat_dim', type=int, default=64,
                       help='State feature dimension')
    parser.add_argument('--token_dim', type=int, default=256,
                       help='Token dimension')
    parser.add_argument('--gru_hidden_dim', type=int, default=128,
                       help='GRU hidden dimension')
    parser.add_argument('--dataset', type=str, default='pusht_episodes_rates.pkl',
                       help='Path to dataset (with rate labels)')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device (auto, cpu, cuda)')
    parser.add_argument('--skip_rates', action='store_true',
                       help='Skip rate plots (only generate global progress)')
    parser.add_argument('--non_causal', action='store_true',
                       help='Use non-causal normalization (normalize by final value, requires full episode)')
    
    args = parser.parse_args()
    
    causal = not args.non_causal
    
    # Load model
    print(f"Loading model from {args.model_path}...")
    model = load_model(
        model_path=args.model_path,
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
        data = pickle.load(f)
    
    if isinstance(data, dict) and 'episodes' in data:
        episodes = data['episodes']
        stats = data.get('stats', {})
    else:
        episodes = data
        stats = {}
    
    # Split
    train_ratio = 0.8
    split_idx = int(train_ratio * len(episodes))
    train_episodes = episodes[:split_idx]
    val_episodes = episodes[split_idx:]
    
    print(f"Total episodes: {len(episodes)}")
    print(f"Train episodes: {len(train_episodes)}")
    print(f"Validation episodes: {len(val_episodes)}")
    print(f"Rate stats: push_mean={stats.get('push_rate_mean', 'N/A'):.4f}, "
          f"push_std={stats.get('push_rate_std', 'N/A'):.4f}, "
          f"repo_mean={stats.get('repo_rate_mean', 'N/A'):.4f}, "
          f"repo_std={stats.get('repo_rate_std', 'N/A'):.4f}")
    
    # Print causal progress computation info
    avg_ep_len = stats.get('avg_episode_length', 100)
    push_frac = stats.get('push_fraction', 0.5)
    print(f"Causal progress stats: avg_episode_length={avg_ep_len:.1f}, push_fraction={push_frac:.3f}")
    
    # Compute state stats from training set
    print("Computing state normalization statistics...")
    state_mean, state_std = compute_state_stats(train_episodes)
    
    episodes_per_page = 4
    
    if not args.skip_rates:
        # Generate rate plots for validation set
        print("\nGenerating rate plots for validation set...")
        n_val_pages = (len(val_episodes) + episodes_per_page - 1) // episodes_per_page
        
        for page_idx in range(min(n_val_pages, 3)):  # Limit to 3 pages
            start_idx = page_idx * episodes_per_page
            end_idx = min(start_idx + episodes_per_page, len(val_episodes))
            ep_indices = list(range(start_idx, end_idx))
            actual_ep_numbers = [split_idx + i for i in ep_indices]
            
            out_path = f'rates_val_page{page_idx+1}.png'
            plot_rates_grid(model, val_episodes, ep_indices=ep_indices,
                           device=device, out_path=out_path, 
                           episodes_per_page=episodes_per_page,
                           actual_ep_numbers=actual_ep_numbers,
                           state_mean=state_mean, state_std=state_std)
        
        # Single detailed rate plot for first validation episode
        if len(val_episodes) > 0:
            plot_rates_over_time(model, val_episodes, ep_idx=0, device=device,
                                out_path='rates_detailed_val0.png',
                                state_mean=state_mean, state_std=state_std, stats=stats)
    
    # Generate global progress plots
    mode_str = "causal" if causal else "non-causal"
    print(f"\nGenerating global progress plots for validation set ({mode_str} mode)...")
    n_val_pages = (len(val_episodes) + episodes_per_page - 1) // episodes_per_page
    
    for page_idx in range(min(n_val_pages, 3)):
        start_idx = page_idx * episodes_per_page
        end_idx = min(start_idx + episodes_per_page, len(val_episodes))
        ep_indices = list(range(start_idx, end_idx))
        actual_ep_numbers = [split_idx + i for i in ep_indices]
        
        out_path = f'global_progress_val_page{page_idx+1}.png'
        plot_global_progress_grid(model, val_episodes, ep_indices=ep_indices,
                                  device=device, out_path=out_path,
                                  episodes_per_page=episodes_per_page,
                                  actual_ep_numbers=actual_ep_numbers,
                                  state_mean=state_mean, state_std=state_std, stats=stats,
                                  causal=causal)
    
    # Detailed global progress plot for first validation episode
    if len(val_episodes) > 0:
        plot_global_progress(model, val_episodes, ep_idx=0, device=device,
                            out_path='global_progress_detailed_val0.png',
                            state_mean=state_mean, state_std=state_std, stats=stats,
                            causal=causal)
    
    # Compute overall metrics
    print("\nComputing overall metrics on validation set...")
    all_progress_maes = []
    for ep_idx in range(len(val_episodes)):
        episode = val_episodes[ep_idx]
        pred_progress, _, _, _ = compute_global_progress(
            model, episode, device, state_mean, state_std, stats, causal=causal
        )
        gt_progress = compute_ground_truth_progress(episode, stats, causal=causal)
        mae = np.mean(np.abs(pred_progress - gt_progress))
        all_progress_maes.append(mae)
    
    print(f"Global Progress MAE (validation, {mode_str}): {np.mean(all_progress_maes):.4f} +/- {np.std(all_progress_maes):.4f}")
    
    print("\nDone! Generated visualizations:")
    if not args.skip_rates:
        print("  - rates_val_page*.png (rate predictions)")
        print("  - rates_detailed_val0.png")
    print("  - global_progress_val_page*.png (integrated progress)")
    print("  - global_progress_detailed_val0.png")


if __name__ == "__main__":
    main()
