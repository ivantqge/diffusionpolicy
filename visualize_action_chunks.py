"""
Visualize Action Chunks from Episodes

Samples frames from train/test episodes and visualizes the ground truth action chunks
showing where the agent will move in the next 8 timesteps.
"""

import numpy as np
import matplotlib.pyplot as plt
import click
import pickle
from infer_progress import extract_action_chunk

def _plot_action_chunk(fig_title, img, agent_state, block_pose, action_chunk, out_path,
                       progress=None, horizon_label="t+1..t+8"):
    # Normalize image
    img_display = img
    if img_display.dtype != np.float32 and img_display.dtype != np.float64:
        img_display = img_display.astype(np.float32)
    if img_display.max() > 1.5:
        img_display = img_display / 255.0
    img_display = np.clip(img_display, 0, 1)

    # Use constrained layout (DON'T call tight_layout)
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(14, 7), constrained_layout=True
    )
    fig.suptitle(fig_title, fontsize=14, fontweight='bold')

    H, W = img_display.shape[:2]

    # ---------- Left: snapshot ----------
    ax1.imshow(img_display)
    ax1.scatter([agent_state[0]], [agent_state[1]], s=120, edgecolors='white', linewidths=2)
    ax1.set_title("Snapshot", fontsize=12, fontweight='bold')
    ax1.set_xlim(0, W); ax1.set_ylim(H, 0)
    ax1.axis("off")

    if progress is None:
        progress = 0.0
    info_text = (
        f"progress: {progress:.2f}\n"
        f"agent:  [{agent_state[0]:.1f}, {agent_state[1]:.1f}]\n"
        f"block:  [{block_pose[0]:.1f}, {block_pose[1]:.1f}]"
    )
    ax1.text(
        0.02, 0.98, info_text, transform=ax1.transAxes,
        va='top', ha='left', fontsize=10,
        bbox=dict(boxstyle='round,pad=0.35', facecolor='white', alpha=0.85, edgecolor='none')
    )

    # ---------- Right: trajectory ----------
    ax2.imshow(img_display, alpha=0.25)
    ax2.set_title(f"Action chunk ({horizon_label})", fontsize=12, fontweight='bold')
    ax2.set_xlim(0, W); ax2.set_ylim(H, 0)
    ax2.axis("off")
    print(W, H)
    # Build trajectory
    cur = agent_state.copy().astype(np.float32)
    cur[0] = cur[0] * 96 / 512
    cur[1] = cur[1] * 96 / 512
    traj = [cur.copy()]
    for a in action_chunk:
        a = np.asarray(a, dtype=np.float32)
        nxt = a * 96 / 512
        print(a)
        traj.append(nxt.copy())
        cur = nxt

    traj = np.stack(traj, axis=0)
    #print(traj[0, 0], traj[0, 1])
    # Plot path + points
    #ax2.plot(traj[:, 0], traj[:, 1], linewidth=3, alpha=0.9)
    ax2.scatter([traj[0, 0]], [traj[0, 1]], s=140, edgecolors='black', linewidths=2)  # start
    ax2.scatter(traj[1:, 0], traj[1:, 1], s=70, edgecolors='black', linewidths=1)

    # # Label only a few steps to avoid clutter
    H, W = img_display.shape[:2]
    offset = max(2, int(0.02 * W))

    k = len(traj) - 1
    label_steps = sorted(set([1, max(1, k//2), k]))

    for t in label_steps:
        x, y = traj[t]

        # offset away from the direction of motion (so it doesn't sit on the point)
        if t > 0:
            vx, vy = traj[t] - traj[t-1]
            n = np.hypot(vx, vy) + 1e-6
            ux, uy = vx / n, vy / n
            ox, oy = -uy, ux   # perpendicular
        else:
            ox, oy = 0.0, -1.0

        tx = np.clip(x + ox * offset, 0, W-1)
        ty = np.clip(y + oy * offset, 0, H-1)

        ax2.text(
            tx, ty, f"+{t}", fontsize=9, weight="bold",
            ha="left", va="bottom",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="black", alpha=0.7, edgecolor="none"),
            color="white",
            zorder=10
        )

    # Arrow indicating first move
    if len(traj) > 1:
        head = 0.5  
        ax2.annotate(
            "", xy=(traj[1,0], traj[1,1]), xytext=(traj[0,0], traj[0,1]),
            arrowprops=dict(arrowstyle=f"-|>,head_width={head},head_length={head}",
                            lw=1.5, color="yellow", alpha=0.9),
            zorder=9
        )

    # Save without bbox_inches='tight' (it often re-breaks layout)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)



def visualize_action_chunks_from_episodes(dataset_path='pusht_episodes.pkl', train_ratio=0.8, 
                                          n_episodes=5, n_frames_per_episode=3, 
                                          action_chunk_horizon=8, out_prefix='action_chunks'):
    """
    Sample frames from train/test episodes and visualize the ground truth action chunks.
    
    Args:
        dataset_path: Path to episodes .pkl file
        train_ratio: Train/val split ratio
        n_episodes: Number of episodes to sample from
        n_frames_per_episode: Number of frames to sample from each episode
        action_chunk_horizon: Number of future actions to visualize
        out_prefix: Prefix for output images
    """
    print(f"Loading dataset from {dataset_path}...")
    with open(dataset_path, 'rb') as f:
        episodes = pickle.load(f)
    
    # Split into train/val
    split_idx = int(train_ratio * len(episodes))
    train_episodes = episodes[:split_idx]
    val_episodes = episodes[split_idx:]
    
    print(f"Train episodes: {len(train_episodes)}, Val episodes: {len(val_episodes)}")
    
    # Sample frames from train episodes
    for ep_idx, episode in enumerate(train_episodes[:n_episodes]):
        if len(episode) == 0:
            continue
        
        # Sample frames evenly spaced throughout the episode
        frame_indices = np.linspace(0, len(episode) - 1, n_frames_per_episode, dtype=int)
        
        for frame_idx in frame_indices:
            step = episode[frame_idx]
            
            if 'img' not in step or 'action' not in step:
                continue
            
            # Extract ground truth action chunk
            action_chunk = extract_action_chunk(episode, frame_idx, action_chunk_horizon, action_dim=2)
            
            if action_chunk is None:
                continue
            
            # Get current state
            agent_state = step['state'][:2]  # [agent_x, agent_y]
            block_pose = step['state'][2:5]  # [block_x, block_y, block_angle]
            
            # Get image
            img = step['img']  # HWC format
            
            progress = step.get('normalized_timestep', frame_idx / max(1, (len(episode)-1)))
            title = f"Train Ep {ep_idx+1} | Frame {frame_idx}/{len(episode)-1}"
            out_path = f'{out_prefix}_train_ep{ep_idx+1}_frame{frame_idx}.png'

            _plot_action_chunk(
                fig_title=title,
                img=img,
                agent_state=agent_state,
                block_pose=block_pose,
                action_chunk=action_chunk,
                out_path=out_path,
                progress=progress,
                horizon_label=f"t+1..t+{action_chunk_horizon}",
            )

            plt.close()
            print(f"Saved: {out_path}")


@click.command()
@click.option('--dataset_path', default='pusht_episodes.pkl', help='Path to dataset')
@click.option('--train_ratio', default=0.8, type=float, help='Train/val split ratio')
@click.option('--n_episodes', default=5, type=int, help='Number of episodes to sample from')
@click.option('--n_frames_per_episode', default=3, type=int, help='Number of frames per episode')
@click.option('--action_chunk_horizon', default=8, type=int, help='Action chunk horizon')
@click.option('--out_prefix', default='action_chunks', help='Output file prefix')
def main(dataset_path, train_ratio, n_episodes, n_frames_per_episode, action_chunk_horizon, out_prefix):
    """Visualize ground truth action chunks from train/test episodes"""
    visualize_action_chunks_from_episodes(
        dataset_path=dataset_path,
        train_ratio=train_ratio,
        n_episodes=n_episodes,
        n_frames_per_episode=n_frames_per_episode,
        action_chunk_horizon=action_chunk_horizon,
        out_prefix=out_prefix
    )


if __name__ == "__main__":
    main()
