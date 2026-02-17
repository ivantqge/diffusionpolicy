"""
Script to load and extract rate-of-change features from push-T training data
for the dual-head progress predictor.

Extracts:
- Phase labels (pushing vs repositioning based on n_contacts)
- Push rate: rate of T-block approach to goal (distance + angle)
- Reposition rate: rate of agent approach to next contact position

Usage:
    python load_pusht_data_rates.py
"""

import numpy as np
from diffusion_policy.common.replay_buffer import ReplayBuffer
import pickle


# Default goal pose for PushT environment
DEFAULT_GOAL_POSE = np.array([256.0, 256.0, np.pi / 4])


def angular_distance(angle1, angle2):
    """
    Compute shortest angular distance between two angles (in radians).
    Returns value in [0, pi].
    """
    diff = (angle1 - angle2) % (2 * np.pi)
    if diff > np.pi:
        diff = 2 * np.pi - diff
    return diff


def find_contact_segments(n_contacts_array):
    """
    Find segments of contact and non-contact in an episode.
    
    Args:
        n_contacts_array: array of n_contacts values for each timestep
    
    Returns:
        List of (start_idx, end_idx, is_contact) tuples
    """
    segments = []
    if len(n_contacts_array) == 0:
        return segments
    
    current_contact = n_contacts_array[0] > 0
    segment_start = 0
    
    for t in range(1, len(n_contacts_array)):
        is_contact = n_contacts_array[t] > 0
        if is_contact != current_contact:
            segments.append((segment_start, t - 1, current_contact))
            segment_start = t
            current_contact = is_contact
    
    # Add final segment
    segments.append((segment_start, len(n_contacts_array) - 1, current_contact))
    return segments


def find_next_contact_idx(t, n_contacts_array):
    """
    Find the index of the next frame where contact occurs.
    Returns None if no future contact exists.
    """
    for i in range(t + 1, len(n_contacts_array)):
        if n_contacts_array[i] > 0:
            return i
    return None


def find_prev_contact_idx(t, n_contacts_array):
    """
    Find the index of the previous frame where contact occurred.
    Returns None if no previous contact exists.
    """
    for i in range(t - 1, -1, -1):
        if n_contacts_array[i] > 0:
            return i
    return None


def compute_push_rate(state_t, state_prev, goal_pose, lambda_angle=0.3, dt=1.0):
    """
    Compute rate of T-block approach to goal (positive = approaching).
    
    Args:
        state_t: current state [agent_x, agent_y, block_x, block_y, block_angle]
        state_prev: previous state
        goal_pose: target pose [x, y, angle]
        lambda_angle: weight for angle component
        dt: time step
    
    Returns:
        push_rate: signed float (positive = approaching goal)
    """
    # Extract block pose from state
    block_pos_t = state_t[2:4]
    block_angle_t = state_t[4]
    block_pos_prev = state_prev[2:4]
    block_angle_prev = state_prev[4]
    
    # Distance component
    dist_t = np.linalg.norm(block_pos_t - goal_pose[:2])
    dist_prev = np.linalg.norm(block_pos_prev - goal_pose[:2])
    dist_rate = -(dist_t - dist_prev) / dt  # positive when approaching
    
    # Angle component (shortest angular distance)
    angle_diff_t = angular_distance(block_angle_t, goal_pose[2])
    angle_diff_prev = angular_distance(block_angle_prev, goal_pose[2])
    angle_rate = -(angle_diff_t - angle_diff_prev) / dt  # positive when aligning
    
    # Combined rate (weighted)
    return dist_rate + lambda_angle * angle_rate


def compute_reposition_rate(state_t, state_prev, target_pos, dt=1.0):
    """
    Compute rate of agent approach to target position (positive = approaching).
    
    Args:
        state_t: current state [agent_x, agent_y, ...]
        state_prev: previous state
        target_pos: target position [x, y]
        dt: time step
    
    Returns:
        repo_rate: signed float (positive = approaching target)
    """
    agent_pos_t = state_t[:2]
    agent_pos_prev = state_prev[:2]
    
    dist_t = np.linalg.norm(agent_pos_t - target_pos)
    dist_prev = np.linalg.norm(agent_pos_prev - target_pos)
    
    return -(dist_t - dist_prev) / dt  # positive when approaching


def lerp(a, b, t):
    """Linear interpolation between a and b by factor t."""
    return a + t * (b - a)


def extract_trajectory_with_rates(replay_buffer, goal_pose=None, lambda_angle=0.3,
                                   state_key='state', action_key='action'):
    """
    Extract trajectory steps with rate-of-change labels for dual-head predictor.
    
    Args:
        replay_buffer: ReplayBuffer instance
        goal_pose: target pose for T-block [x, y, angle], defaults to [256, 256, pi/4]
        lambda_angle: weight for angle component in push rate
        state_key: key for state vector
        action_key: key for actions
    
    Returns:
        List of episodes, each episode is a list of step dictionaries
    """
    if goal_pose is None:
        goal_pose = DEFAULT_GOAL_POSE
    
    episodes = []
    n_episodes = replay_buffer.n_episodes
    
    # Statistics for normalization
    all_push_rates = []
    all_repo_rates = []
    
    for ep_idx in range(n_episodes):
        episode_data = replay_buffer.get_episode(ep_idx, copy=True)
        T = len(episode_data[action_key])
        
        if T < 2:
            continue
        
        # Get n_contacts for phase detection
        if 'n_contacts' in episode_data:
            n_contacts = episode_data['n_contacts'].flatten()
        else:
            # If n_contacts not available, assume always pushing (fallback)
            n_contacts = np.ones(T)
        
        # Pre-compute next contact indices for repositioning phases
        next_contact_indices = []
        for t in range(T):
            next_contact_indices.append(find_next_contact_idx(t, n_contacts))
        
        # Pre-compute previous contact indices
        prev_contact_indices = []
        for t in range(T):
            prev_contact_indices.append(find_prev_contact_idx(t, n_contacts))
        
        episode_steps = []
        
        for t in range(T):
            state_t = episode_data[state_key][t]
            is_pushing = n_contacts[t] > 0
            
            # Initialize rates
            push_rate = 0.0
            repo_rate = 0.0
            next_contact_pos = None
            smoothed_target = None
            
            if t > 0:
                state_prev = episode_data[state_key][t - 1]
                
                # Compute push rate (always compute for continuity, but only valid when pushing)
                push_rate = compute_push_rate(state_t, state_prev, goal_pose, lambda_angle)
                
                # Compute reposition rate (when not pushing)
                if not is_pushing:
                    next_idx = next_contact_indices[t]
                    prev_idx = prev_contact_indices[t]
                    
                    if next_idx is not None:
                        # Get next contact position (agent position at next contact)
                        next_contact_pos = episode_data[state_key][next_idx][:2].copy()
                        
                        # Compute smoothed target (interpolate from last contact to next)
                        if prev_idx is not None:
                            last_contact_pos = episode_data[state_key][prev_idx][:2]
                            # Progress through the repositioning segment
                            segment_length = next_idx - prev_idx
                            progress = (t - prev_idx) / segment_length if segment_length > 0 else 0.0
                            smoothed_target = lerp(last_contact_pos, next_contact_pos, progress)
                        else:
                            # No previous contact, use current position as start
                            smoothed_target = next_contact_pos
                        
                        repo_rate = compute_reposition_rate(state_t, state_prev, smoothed_target)
                    else:
                        # No future contact - use goal position as fallback target
                        repo_rate = compute_reposition_rate(state_t, state_prev, goal_pose[:2])
            
            # Collect rates for normalization
            if is_pushing and t > 0:
                all_push_rates.append(push_rate)
            elif not is_pushing and t > 0:
                all_repo_rates.append(repo_rate)
            
            # Build step dictionary
            step_dict = {
                'state': state_t,
                'action': episode_data[action_key][t],
                'timestep': t,
                'normalized_timestep': t / (T - 1) if T > 1 else 0.0,
                'episode_idx': ep_idx,
                'T': T,
                'is_pushing': is_pushing,
                'push_rate': push_rate,
                'repo_rate': repo_rate,
                'n_contacts': float(n_contacts[t]),
            }
            
            # Add optional debug info
            if next_contact_pos is not None:
                step_dict['next_contact_pos'] = next_contact_pos
            if smoothed_target is not None:
                step_dict['smoothed_target'] = smoothed_target
            
            # Add image and other keys
            for key in episode_data.keys():
                if key not in ['state', 'action', 'n_contacts']:
                    step_dict[key] = episode_data[key][t]
            
            episode_steps.append(step_dict)
        
        episodes.append(episode_steps)
    
    # Compute episode-level statistics for causal progress normalization
    episode_lengths = [len(ep) for ep in episodes]
    total_pushing_steps = sum(all_push_rates.__len__() for _ in [1])  # len of push rates list
    total_repo_steps = sum(all_repo_rates.__len__() for _ in [1])  # len of repo rates list
    total_steps = len(all_push_rates) + len(all_repo_rates)
    push_fraction = len(all_push_rates) / total_steps if total_steps > 0 else 0.5
    
    # Compute normalization statistics
    stats = {
        'push_rate_mean': np.mean(all_push_rates) if all_push_rates else 0.0,
        'push_rate_std': np.std(all_push_rates) if all_push_rates else 1.0,
        'repo_rate_mean': np.mean(all_repo_rates) if all_repo_rates else 0.0,
        'repo_rate_std': np.std(all_repo_rates) if all_repo_rates else 1.0,
        'goal_pose': goal_pose,
        'lambda_angle': lambda_angle,
        # Statistics for causal progress computation (no future knowledge needed)
        'avg_episode_length': np.mean(episode_lengths) if episode_lengths else 100.0,
        'push_fraction': push_fraction,
        'num_episodes': len(episodes),
        'total_pushing_steps': len(all_push_rates),
        'total_repo_steps': len(all_repo_rates),
    }
    
    print(f"\n--- Statistics for causal progress ---")
    print(f"  Average episode length: {stats['avg_episode_length']:.1f}")
    print(f"  Push fraction: {stats['push_fraction']:.3f}")
    print(f"  Total pushing steps: {stats['total_pushing_steps']}")
    print(f"  Total repo steps: {stats['total_repo_steps']}")
    
    return episodes, stats


def normalize_rates(episodes, stats):
    """
    Normalize push and reposition rates using computed statistics.
    
    Args:
        episodes: list of episodes with rate labels
        stats: normalization statistics dict
    
    Returns:
        episodes with normalized rates added
    """
    push_mean = stats['push_rate_mean']
    push_std = stats['push_rate_std']
    repo_mean = stats['repo_rate_mean']
    repo_std = stats['repo_rate_std']
    
    for episode in episodes:
        for step in episode:
            step['push_rate_normalized'] = (step['push_rate'] - push_mean) / (push_std + 1e-6)
            step['repo_rate_normalized'] = (step['repo_rate'] - repo_mean) / (repo_std + 1e-6)
    
    return episodes


def save_episodes_with_rates(episodes, stats, output_path="pusht_episodes_rates.pkl"):
    """
    Save episodes with rate labels and normalization statistics.
    """
    data = {
        'episodes': episodes,
        'stats': stats,
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"Saved {len(episodes)} episodes to {output_path}")
    print(f"Rate statistics:")
    print(f"  Push rate: mean={stats['push_rate_mean']:.4f}, std={stats['push_rate_std']:.4f}")
    print(f"  Repo rate: mean={stats['repo_rate_mean']:.4f}, std={stats['repo_rate_std']:.4f}")
    
    return data


def load_pusht_data_with_rates(zarr_path="data/pusht_cchi_v7_replay.zarr", 
                                goal_pose=None, lambda_angle=0.3):
    """
    Load push-T data and extract rate-of-change labels.
    
    Args:
        zarr_path: path to zarr replay buffer
        goal_pose: target pose for T-block
        lambda_angle: weight for angle component in push rate
    
    Returns:
        episodes: list of episodes with rate labels
        stats: normalization statistics
    """
    try:
        replay_buffer = ReplayBuffer.copy_from_path(zarr_path)
        
        print(f"Number of episodes: {replay_buffer.n_episodes}")
        print(f"Total timesteps: {replay_buffer.n_steps}")
        print(f"Available keys: {list(replay_buffer.keys())}")
        
        # Check data shapes
        print("\n--- Data shapes ---")
        for key in replay_buffer.keys():
            arr = replay_buffer[key]
            print(f"  {key}: shape={arr.shape}, dtype={arr.dtype}")
        
        # Extract trajectories with rates
        print("\n--- Extracting trajectories with rate labels ---")
        episodes, stats = extract_trajectory_with_rates(
            replay_buffer, 
            goal_pose=goal_pose,
            lambda_angle=lambda_angle
        )
        
        # Normalize rates
        episodes = normalize_rates(episodes, stats)
        
        return episodes, stats
        
    except Exception as e:
        print(f"Error loading data: {e}")
        import traceback
        traceback.print_exc()
        print(f"\nMake sure the data exists at: {zarr_path}")
        return None, None


def print_episode_summary(episodes, ep_idx=0):
    """Print summary of an episode's phase and rate information."""
    if ep_idx >= len(episodes):
        print(f"Episode {ep_idx} not found")
        return
    
    episode = episodes[ep_idx]
    print(f"\n--- Episode {ep_idx} Summary ({len(episode)} steps) ---")
    
    # Count phases
    pushing_steps = sum(1 for s in episode if s['is_pushing'])
    repo_steps = len(episode) - pushing_steps
    print(f"Pushing steps: {pushing_steps}, Repositioning steps: {repo_steps}")
    
    # Show some example steps
    print("\nSample steps:")
    for t in [0, len(episode)//4, len(episode)//2, 3*len(episode)//4, len(episode)-1]:
        if t < len(episode):
            step = episode[t]
            phase = "PUSH" if step['is_pushing'] else "REPO"
            print(f"  t={t:3d}: {phase}, push_rate={step['push_rate']:+.4f}, "
                  f"repo_rate={step['repo_rate']:+.4f}, n_contacts={step['n_contacts']:.0f}")


if __name__ == "__main__":
    print("=" * 60)
    print("Push-T Rate-of-Change Data Extraction")
    print("=" * 60)
    
    # Load data and extract rate labels
    episodes, stats = load_pusht_data_with_rates()
    
    if episodes is not None and len(episodes) > 0:
        # Print summary for first few episodes
        for ep_idx in [0, 1, 2]:
            if ep_idx < len(episodes):
                print_episode_summary(episodes, ep_idx)
        
        # Save to file
        save_episodes_with_rates(episodes, stats)
        
        print("\n" + "=" * 60)
        print("Data extraction complete!")
        print("=" * 60)
    else:
        print("\nFailed to load data")
