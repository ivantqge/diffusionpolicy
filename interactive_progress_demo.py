import torch
import numpy as np
import pygame
import click
from diffusion_policy.env.pusht.pusht_keypoints_env import PushTKeypointsEnv
from infer_progress import load_model, predict


@click.command()
@click.option('-m', '--model_path', default='progress_models/weights/progress_predictor_action_regression.pth', 
              help='Path to progress predictor model')
@click.option('-rs', '--render_size', default=96, type=int, help='Render size for environment')
@click.option('-hz', '--control_hz', default=10, type=int, help='Control frequency')
@click.option('--num_bins', default=50, type=int, help='Number of bins for categorical model')
@click.option('--action_chunk_horizon', default=8, type=int, help='Action chunk horizon (0 to disable)')
@click.option('--action_dim', default=2, type=int, help='Action dimension')
@click.option('--device', default='auto', help='Device to run on (auto, cpu, cuda)')
def main(model_path, render_size, control_hz, num_bins, action_chunk_horizon, action_dim, device):
    """
    Interactive demo for progress prediction.
    
    Controls:
    - Move mouse near blue circle to control agent
    - Press SPACEBAR to infer progress at current state
      (also prints predicted progress for 8 directions if using action chunks)
    - Press R to reset environment
    - Press Q to quit
    """
    
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load progress predictor model
    print(f"Loading model from {model_path}...")
    model = load_model(model_path=model_path, num_bins=num_bins, action_chunk_horizon=action_chunk_horizon, action_dim=action_dim)
    model = model.to(device)
    print("Model loaded successfully!")
    
    # Create PushT environment
    kp_kwargs = PushTKeypointsEnv.genenerate_keypoint_manager_params()
    env = PushTKeypointsEnv(render_size=render_size, render_action=False, **kp_kwargs)
    agent = env.teleop_agent()
    clock = pygame.time.Clock()
    
    # Initialize pygame display
    pygame.init()
    pygame.display.set_caption('Progress Prediction Demo')
    
    # Generate 8 direction vectors (45 degree increments) for 8-direction prediction
    # Directions: 0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°
    angles = np.deg2rad(np.arange(0, 360, 45))
    direction_vectors = np.array([np.cos(angles), np.sin(angles)]).T  # [8, 2]
    # Scale to reasonable action magnitude (e.g., 50 pixels)
    action_magnitude = 50.0
    direction_vectors = direction_vectors * action_magnitude
    
    # Current progress prediction
    current_progress = None
    last_inference_time = None
    
    # Episode loop
    while True:
        seed = 0
        print(f'Starting episode with seed {seed}')
        env.seed(seed)
        
        # Reset environment
        obs = env.reset()
        info = env._get_info()
        img_array = env.render(mode='human')
        
        # Store start image for Siamese architecture
        start_img = img_array.copy() if model.use_siamese else None
        
        # Action history buffer for action chunk extraction
        action_history = []
        
        # Episode state
        done = False
        retry = False
        
        # Step loop
        while not done:
            # Process events
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        # Infer progress on spacebar press
                        # Get current state: agent pos + block pose (5D)
                        state = np.concatenate([info['pos_agent'], info['block_pose']])
                        
                        img_array = env.render(mode='rgb_array')

                        # Prepare action chunk for future actions
                        action_chunk = None
                        if model.action_chunk_horizon > 0:
                            horizon = model.action_chunk_horizon
                            
                            current_action = None
                            if action_history:
                                current_action = action_history[-1]
                            else:
                                current_action = agent.act(obs)
                            
                            if current_action is not None:
                                if hasattr(current_action, 'x'):
                                    current_action = np.array([current_action.x, current_action.y], dtype=np.float32)
                                else:
                                    current_action = np.array(current_action, dtype=np.float32).flatten()
                
                                action_chunk = np.tile(current_action, (horizon, 1)).astype(np.float32)
                            else:
                                # otherwise use agent position
                                action_chunk = np.tile(info['pos_agent'], (horizon, 1)).astype(np.float32)
                        
                        # Run inference
                        progress = predict(model, img_array, state, device=device, action_chunk=action_chunk, start_img=start_img)
                        current_progress = progress
                        last_inference_time = pygame.time.get_ticks()
                        print(f"\nInferred progress: {progress:.4f} (state: agent={info['pos_agent']}, block={info['block_pose'][:2]})")
                        
                        # If using action chunks, also predict progress for 8 directions
                        if model.action_chunk_horizon > 0:
                            direction_names = ['Right (0°)', 'Down-Right (45°)', 'Down (90°)', 'Down-Left (135°)', 
                                               'Left (180°)', 'Up-Left (225°)', 'Up (270°)', 'Up-Right (315°)']
                            predictions = []
                            
                            # Get agent's current position
                            agent_pos = info['pos_agent']
                            
                            for direction in direction_vectors:
                                # Create progressive trajectory: agent moves further in direction over time
                                horizon = model.action_chunk_horizon
                                direction_flat = direction.flatten()
                                
                                dir_action_chunk = []
                                for i in range(horizon):
                                    progress_fraction = (i + 1) / horizon
                                    target_pos = agent_pos + direction_flat * progress_fraction
                                    target_pos = np.clip(target_pos, 0, 512).astype(np.float32)
                                    dir_action_chunk.append(target_pos)
                                
                                dir_action_chunk = np.array(dir_action_chunk, dtype=np.float32)
                                dir_progress = predict(model, img_array, state, device=device, action_chunk=dir_action_chunk, start_img=start_img)
                                predictions.append(dir_progress)
                            
                            print("\n--- Predicted Progress for 8 Directions ---")
                            print(f"  (Agent at: [{agent_pos[0]:.1f}, {agent_pos[1]:.1f}])")
                            for i, (name, pred) in enumerate(zip(direction_names, predictions)):
                                target = agent_pos + direction_vectors[i].flatten()
                                target = np.clip(target, 0, 512)
                                print(f"  {name}: {pred:.4f} (target: [{target[0]:.1f}, {target[1]:.1f}])")
                            print(f"  Best direction: {direction_names[np.argmax(predictions)]} ({max(predictions):.4f})")
                            print("--------------------------------------------")
                    
                    elif event.key == pygame.K_r:
                        # Reset environment
                        action_history = []  # Clear action history on reset
                        retry = True
                        break
                    elif event.key == pygame.K_q:
                        # Quit
                        pygame.quit()
                        exit(0)
            
            if retry:
                break
            
            # Get action from mouse
            act = agent.act(obs)
            
            # Store action in history
            if act is not None:
                if hasattr(act, 'x'):
                    act_array = np.array([act.x, act.y], dtype=np.float32)
                else:
                    act_array = np.array(act, dtype=np.float32).flatten()
                action_history.append(act_array)
            # Keep only recent history
            if len(action_history) > model.action_chunk_horizon * 2:
                action_history = action_history[-model.action_chunk_horizon * 2:]
            
            # Step environment
            obs, reward, done, info = env.step(act)

            done = False
            
            # Render environment
            img_array = env.render(mode='human')
            
            if current_progress is not None and last_inference_time is not None:
                elapsed = pygame.time.get_ticks() - last_inference_time
                if elapsed > 3000:  # 3 seconds
                    current_progress = None
            
            # Regulate control frequency
            clock.tick(control_hz)
        
        if retry:
            print(f'Retrying episode...')
        else:
            print(f'Episode completed. Press R to reset or Q to quit.')


if __name__ == "__main__":
    main()

