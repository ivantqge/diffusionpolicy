"""
Script to load and extract relevant features from push-T training data
Extracts (s, a, t) tuples from the data

Usage:
    python load_pusht_data.py
"""

import numpy as np
from diffusion_policy.common.replay_buffer import ReplayBuffer

def extract_trajectory_steps(replay_buffer, state_key='state', action_key='action'):
    """
    Extract (s, a, t) trajectory steps from ReplayBuffer
    
    Args:
        replay_buffer: ReplayBuffer instance
        state_key: key for state vector (e.g., 'state')
        action_key: key for actions (e.g., 'action')
    
    Returns:
        List of dictionaries with keys: 'state', 'action', 'timestep', 'normalized_timestep','episode_idx'
    """

    trajectory_steps = []
    n_episodes = replay_buffer.n_episodes
    all_keys = list(replay_buffer.keys())
        
    for ep_idx in range(n_episodes):
        episode_data = replay_buffer.get_episode(ep_idx, copy=True)
        
        T = len(episode_data[action_key])
        
        # Create (s, a, t) tuples for each timestep in the episode
        for t in range(T):
            # normalized version
            normalized_t = t / (T - 1) if T > 1 else 0.0
            
            step_dict = {
                'state': episode_data[state_key][t] if state_key in episode_data else None,
                'action': episode_data[action_key][t],
                'timestep': t,  
                'normalized_timestep': normalized_t,  
                'episode_idx': ep_idx,
                'T': T  
            }
            
            # Add all other keys as well to the dataset (img, keypoint, n_contacts, etc.)
            for key in episode_data.keys():
                if key not in ['state', 'action']:
                    step_dict[key] = episode_data[key][t]
            
            trajectory_steps.append(step_dict)
    
    return trajectory_steps


def load_pusht_data_direct():
    """
    Load the push-T data directly and extract trajectory steps
    """
    # Path to the pusht data
    zarr_path = "data/pusht_cchi_v7_replay.zarr"
    
    try:
        replay_buffer = ReplayBuffer.copy_from_path(zarr_path)
        
        print(f"Number of episodes: {replay_buffer.n_episodes}")
        print(f"Total timesteps: {replay_buffer.n_steps}")
        print(f"Available keys: {list(replay_buffer.keys())}")
        
        # Checking data shapes and stuff
        print("\n--- Data shapes ---")
        for key in replay_buffer.keys():
            arr = replay_buffer[key]
            print(f"  {key}: shape={arr.shape}, dtype={arr.dtype}")
        
        # Get episode info
        episode_lengths = replay_buffer.episode_lengths        
        trajectory_steps = extract_trajectory_steps(replay_buffer)
                
        # Show examples
        print("\n--- Example traj steps ---")
        for i in [0, 10, 20]: 
            if i < len(trajectory_steps):
                step = trajectory_steps[i]
                print(f"\nStep {i}:")
                print(f"  Episode: {step['episode_idx']}, Raw timestep: {step['timestep']}/{step.get('T', 'unknown')}")
                print(f"  Normalized timestep: {step['normalized_timestep']:.3f}")
                print(f"  State shape: {step['state'].shape}, Action shape: {step['action'].shape}")
                print(f"  State (first 5 values): {step['state'][:5]}")
                print(f"  Action: {step['action']}")
            
        return replay_buffer, trajectory_steps
        
    except Exception as e:
        print(f"Error loading data: {e}")
        import traceback
        traceback.print_exc()
        print("\nMake sure the data exists at: data/pusht_cchi_v7_replay.zarr")
        return None, None


def save_trajectories_separate(trajectory_steps, output_path="pusht_episodes.pkl"):
    """
    Save each episode separately as a list of dictionaries
    Each episode is a complete trajectory: [(s_0,a_0,t_0), (s_1,a_1,t_1), ...]
    """
    import pickle
    
    # Group by episode
    episodes = {}
    for step in trajectory_steps:
        ep_idx = step['episode_idx']
        if ep_idx not in episodes:
            episodes[ep_idx] = []
        episodes[ep_idx].append(step)
    
    # Convert to list of episodes in order
    episode_list = [episodes[i] for i in sorted(episodes.keys())]
    
    with open(output_path, 'wb') as f:
        pickle.dump(episode_list, f)
    
    # just to check
    print(f"  Episode 0: {len(episode_list[0])} timesteps")
    if len(episode_list) > 1:
        print(f"  Episode 1: {len(episode_list[1])} timesteps")
    
    return episode_list

if __name__ == "__main__":
    print("=" * 60)
    print("Push-T Trajectory Extraction")
    print("=" * 60)
    
    # Load data and extract trajectories
    replay_buffer, trajectory_steps = load_pusht_data_direct()
    
    if trajectory_steps is not None and len(trajectory_steps) > 0:        
        # Save each episode separately (recommended)
        save_trajectories_separate(trajectory_steps)
                
        print("Data saved")
    else:
        print("\n Failed to load data")

