import os
import pickle
from typing import Any, Dict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from omegaconf import DictConfig
from tqdm import tqdm
import wandb

import numpy as np

from diffusion_policy.workspace.base_workspace import BaseWorkspace
from progress_predictor import ProgressPredictor
from progress_predictor_windowed import ProgressPredictorWindowed
from progress_predictor_transformer import ProgressPredictorTransformer
from progress_predictor_dual_head import ProgressPredictorDualHead, DualHeadLoss


class _ProgressDataset(Dataset):
    """
    Dataset that provides (image, agent_state, action_chunk, progress) from episodes .pkl
    - If mode='categorical': progress is class index [0, num_bins-1]
    - If mode='regression': progress is continuous value [0, 1]
    """

    def __init__(self, episodes, normalize_images: bool = True, num_bins: int = 50, mode: str = 'categorical', 
                 action_chunk_horizon: int = 0, windowed: bool = False, window_length: int = 4, is_training: bool = True,
                 state_mean=None, state_std=None, windows_per_episode: int = 16):
        self.normalize_images = normalize_images
        self.num_bins = num_bins
        self.mode = mode
        self.action_chunk_horizon = action_chunk_horizon
        self.windowed = windowed
        self.window_length = window_length
        self.is_training = is_training
        self.episodes = episodes  
        self.state_mean = None
        self.state_std = None

        # State normalization that is computed on training set if not provided 
        if (state_mean is None) or (state_std is None):
            if is_training:
                all_states = []
                for ep in episodes:
                    for step in ep:
                        if 'state' in step:
                            all_states.append(step['state'])
                all_states = np.asarray(all_states, dtype=np.float32)
                self.state_mean = torch.as_tensor(all_states.mean(axis=0), dtype=torch.float32)
                self.state_std = torch.as_tensor(all_states.std(axis=0), dtype=torch.float32)
            else:
                # validation should receive train stats
                self.state_mean = None
                self.state_std = None
        else:
            self.state_mean = state_mean.clone().detach()
            self.state_std = state_std.clone().detach()


        # if not using windowed architecture (GRU/transformer)

        if not windowed:
            # Original single-frame dataset
            self.data = []
            for ep_idx, episode in enumerate(episodes):
                for step_idx, step in enumerate(episode):
                    if 'img' not in step:
                        continue
                    self.data.append({
                        'image': step['img'],
                        'agent_state': step['state'],
                        'progress': step['normalized_timestep'],
                        'episode_idx': ep_idx,
                        'timestep': step['timestep'],
                        'step_idx': step_idx  # Store step index within episode
                    })
        else:
            # Windowed dataset: create windows from episodes
            self.data = []
            for ep_idx, episode in enumerate(episodes):
                if len(episode) < window_length:
                    continue  # Skip episodes that are too short
                
                # For each episode, we can create multiple windows
                # Sample anchor times and strides randomly during training
                num_samples = min(windows_per_episode, len(episode) - 1)
                anchor_candidates = np.arange(1, len(episode)) # exclude start, that we add in later on
                if num_samples < len(anchor_candidates):
                    chosen = np.random.choice(anchor_candidates, size=num_samples, replace=False)
                else:
                    chosen = anchor_candidates

                for step_idx in chosen:
                    if 'img' not in episode[step_idx]:
                        continue
                    self.data.append({
                        'episode_idx': ep_idx,
                        'anchor_idx': int(step_idx),
                        'episode_length': len(episode)
                    })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if not self.windowed:
            # Original single-frame sampling
            item = self.data[idx]

            img = item['image']  # HWC format (96, 96, 3) float32 [0, 255]
            
            # Get start image from episode[0]
            episode = self.episodes[item['episode_idx']]
            start_img = episode[0]['img']  # Start image from first step of episode
            
            # Convert HWC to CHW and normalize
            def process_image(img_array):
                img_processed = img_array.transpose(2, 0, 1)  # HWC -> CHW
                img_processed = img_processed.astype(np.float32) / 255.0  # Normalize to [0, 1]
                if self.normalize_images:
                    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
                    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
                    img_processed = (img_processed - mean) / std
                return torch.as_tensor(img_processed, dtype=torch.float32)
            
            img_tensor = process_image(img)
            start_img_tensor = process_image(start_img)
            agent_state = torch.as_tensor(item['agent_state'], dtype=torch.float32)
            if (self.state_mean is not None) and (self.state_std is not None):
                agent_state = (agent_state - self.state_mean) / (self.state_std + 1e-6)
            
            action_chunk = None
            if self.action_chunk_horizon > 0:
                episode = self.episodes[item['episode_idx']]
                step_idx = item['step_idx']
                episode_length = len(episode)

                action_chunk_list = []
                for i in range(self.action_chunk_horizon):
                    future_idx = step_idx + i
                    if future_idx < episode_length and 'action' in episode[future_idx]:
                        action_chunk_list.append(episode[future_idx]['action'])
                    else:
                        if action_chunk_list:
                            action_chunk_list.append(action_chunk_list[-1])  # Repeat last action
                        else:
                            action_chunk_list.append(agent_state) # use agent state as last action
                
                action_chunk = np.array(action_chunk_list, dtype=np.float32)  
                action_chunk = torch.as_tensor(action_chunk, dtype=torch.float32)
            
            # Handle progress based on mode, either categorical (50 bins) or regression (1 value)
            if self.mode == 'categorical':
                progress_continuous = item['progress']
                progress_continuous = np.clip(progress_continuous, 0.0, 0.999999)
                
                # Create Gaussian distributed target instead of one-hot
                bin_centers = (np.arange(self.num_bins) + 0.5) / self.num_bins
                target_value = progress_continuous
                sigma = 2.0 / self.num_bins
                gaussian_weights = np.exp(-0.5 * ((bin_centers - target_value) / sigma) ** 2)
                gaussian_weights = gaussian_weights / gaussian_weights.sum()  # Normalize to sum to 1
                
                progress = torch.as_tensor(gaussian_weights, dtype=torch.float32)
            else:  # regression
                progress = torch.as_tensor([item['progress']], dtype=torch.float32)

            result = {
                'image': img_tensor,
                'start_image': start_img_tensor,
                'agent_state': agent_state,
                'progress': progress,
                'episode_idx': item['episode_idx'],
                'timestep': item['timestep']
            }
            
            if action_chunk is not None:
                result['action_chunk'] = action_chunk
            
            return result
        else:
            # Windowed sampling: (o0, o_ta, o_ta+s, ..., o_ta+(L-1)s)
            item = self.data[idx]
            episode = self.episodes[item['episode_idx']]
            episode_length = item['episode_length']
            
            # Always include the start frame (index 0)
            # Then sample a window of length L-1 starting at anchor ta with stride s

            L = self.window_length
            stride_choices = [2, 4, 6]

            if self.is_training:
                s = np.random.choice(stride_choices)
                # We need indices: ta, ta+s, ..., ta+(L-2)*s all less than episode_length
                max_ta = (episode_length - 1) - (L - 2) * s
                # ensure ta >= 1 so it's not always the start frame
                if max_ta < 1:
                    # if episode is too short for this stride, fall back to stride=1
                    s = 1
                    max_ta = (episode_length - 1) - (L - 2) * s
                ta = np.random.randint(1, max_ta + 1)
            else:
                # deterministic sampling for validation
                s = 1
                max_ta = (episode_length - 1) - (L - 2) * s
                ta = min(item.get('anchor_idx', 1), max(1, max_ta))

            window_indices = [0] + [ta + i * s for i in range(L - 1)]
            
            # Get images and states for the window
            images_list = []
            states_list = []
            
            def process_image(img_array):
                img_processed = img_array.transpose(2, 0, 1)  # HWC -> CHW
                img_processed = img_processed.astype(np.float32) / 255.0
                if self.normalize_images:
                    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
                    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
                    img_processed = (img_processed - mean) / std
                return torch.as_tensor(img_processed, dtype=torch.float32)
            
            for win_idx in window_indices:
                step = episode[win_idx]
                img_tensor = process_image(step['img'])
                state_tensor = torch.as_tensor(step['state'], dtype=torch.float32)
                if (self.state_mean is not None) and (self.state_std is not None):
                    state_tensor = (state_tensor - self.state_mean) / (self.state_std + 1e-6)
                images_list.append(img_tensor)
                states_list.append(state_tensor)
            
            # Stack into sequences: [L, C, H, W] and [L, state_dim]
            images_seq = torch.stack(images_list, dim=0)  # [L, C, H, W]
            states_seq = torch.stack(states_list, dim=0)  # [L, state_dim]
            
            # Target is progress of last frame: y = p_ta+(L-1)s
            last_idx = window_indices[-1]
            target_progress = episode[last_idx].get('normalized_timestep', last_idx / max(1, episode_length - 1))
            
            # Handle progress based on mode
            if self.mode == 'categorical':
                progress_continuous = np.clip(target_progress, 0.0, 0.999999)
                bin_centers = (np.arange(self.num_bins) + 0.5) / self.num_bins
                sigma = 2.0 / self.num_bins
                gaussian_weights = np.exp(-0.5 * ((bin_centers - progress_continuous) / sigma) ** 2)
                gaussian_weights = gaussian_weights / gaussian_weights.sum()
                progress = torch.as_tensor(gaussian_weights, dtype=torch.float32)
            else:  # regression
                progress = torch.as_tensor([target_progress], dtype=torch.float32)
            
            return {
                'images': images_seq,  # [L, C, H, W]
                'states': states_seq,  # [L, state_dim]
                'progress': progress,
                'episode_idx': item['episode_idx']
            }


class _DualHeadDataset(Dataset):
    """
    Dataset for dual-head progress predictor with rate-of-change labels.
    
    Expects episodes with 'is_pushing', 'push_rate', 'repo_rate' fields
    (generated by load_pusht_data_rates.py).
    """
    
    def __init__(self, episodes, stats, normalize_images: bool = True,
                 window_length: int = 8, is_training: bool = True,
                 state_mean=None, state_std=None, windows_per_episode: int = 16,
                 use_normalized_rates: bool = True):
        self.normalize_images = normalize_images
        self.window_length = window_length
        self.is_training = is_training
        self.episodes = episodes
        self.stats = stats
        self.use_normalized_rates = use_normalized_rates
        
        # State normalization
        if (state_mean is None) or (state_std is None):
            if is_training:
                all_states = []
                for ep in episodes:
                    for step in ep:
                        if 'state' in step:
                            all_states.append(step['state'])
                all_states = np.asarray(all_states, dtype=np.float32)
                self.state_mean = torch.as_tensor(all_states.mean(axis=0), dtype=torch.float32)
                self.state_std = torch.as_tensor(all_states.std(axis=0), dtype=torch.float32)
            else:
                self.state_mean = None
                self.state_std = None
        else:
            self.state_mean = state_mean.clone().detach()
            self.state_std = state_std.clone().detach()
        
        # Build index of windows
        self.data = []
        for ep_idx, episode in enumerate(episodes):
            if len(episode) < window_length:
                continue
            
            num_samples = min(windows_per_episode, len(episode) - 1)
            anchor_candidates = np.arange(1, len(episode))
            if num_samples < len(anchor_candidates):
                chosen = np.random.choice(anchor_candidates, size=num_samples, replace=False)
            else:
                chosen = anchor_candidates
            
            for step_idx in chosen:
                if 'img' not in episode[step_idx]:
                    continue
                self.data.append({
                    'episode_idx': ep_idx,
                    'anchor_idx': int(step_idx),
                    'episode_length': len(episode)
                })
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        episode = self.episodes[item['episode_idx']]
        episode_length = item['episode_length']
        
        L = self.window_length
        stride_choices = [2, 4, 6]
        
        if self.is_training:
            s = np.random.choice(stride_choices)
            max_ta = (episode_length - 1) - (L - 2) * s
            if max_ta < 1:
                s = 1
                max_ta = (episode_length - 1) - (L - 2) * s
            ta = np.random.randint(1, max_ta + 1)
        else:
            s = 1
            max_ta = (episode_length - 1) - (L - 2) * s
            ta = min(item.get('anchor_idx', 1), max(1, max_ta))
        
        window_indices = [0] + [ta + i * s for i in range(L - 1)]
        
        # Get images and states for the window
        images_list = []
        states_list = []
        
        def process_image(img_array):
            img_processed = img_array.transpose(2, 0, 1)  # HWC -> CHW
            img_processed = img_processed.astype(np.float32) / 255.0
            if self.normalize_images:
                mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
                std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
                img_processed = (img_processed - mean) / std
            return torch.as_tensor(img_processed, dtype=torch.float32)
        
        for win_idx in window_indices:
            step = episode[win_idx]
            img_tensor = process_image(step['img'])
            state_tensor = torch.as_tensor(step['state'], dtype=torch.float32)
            if (self.state_mean is not None) and (self.state_std is not None):
                state_tensor = (state_tensor - self.state_mean) / (self.state_std + 1e-6)
            images_list.append(img_tensor)
            states_list.append(state_tensor)
        
        images_seq = torch.stack(images_list, dim=0)
        states_seq = torch.stack(states_list, dim=0)
        
        # Get labels from the last frame in the window
        last_idx = window_indices[-1]
        last_step = episode[last_idx]
        
        # Phase label
        is_pushing = last_step.get('is_pushing', True)
        
        # Rate labels (use normalized if available and requested)
        if self.use_normalized_rates and 'push_rate_normalized' in last_step:
            push_rate = last_step['push_rate_normalized']
            repo_rate = last_step['repo_rate_normalized']
        else:
            push_rate = last_step.get('push_rate', 0.0)
            repo_rate = last_step.get('repo_rate', 0.0)
        
        return {
            'images': images_seq,
            'states': states_seq,
            'is_pushing': torch.tensor(is_pushing, dtype=torch.bool),
            'push_rate': torch.tensor([push_rate], dtype=torch.float32),
            'repo_rate': torch.tensor([repo_rate], dtype=torch.float32),
            'episode_idx': item['episode_idx']
        }


class TrainProgressPredictorWorkspace(BaseWorkspace):
    """
    Hydra workspace to train the progress predictor using episodes .pkl
    """

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.device = torch.device(cfg.training.device)
        self.global_step = 0
        self.epoch = 0

    def _build_dataloaders(self):
        cfg = self.cfg
        num_bins = cfg.model.get('num_bins', 50)
        mode = cfg.model.get('mode', 'categorical')
        action_chunk_horizon = cfg.model.get('action_chunk_horizon', 0)
        windowed = cfg.model.get('windowed', False)
        window_length = cfg.model.get('window_length', 8)
        dual_head = cfg.model.get('dual_head', False)
        
        with open(cfg.dataset.episodes_path, 'rb') as f:
            data = pickle.load(f)
        
        # Handle dual-head format (dict with 'episodes' and 'stats')
        if dual_head:
            if isinstance(data, dict) and 'episodes' in data:
                episodes = data['episodes']
                stats = data.get('stats', {})
            else:
                raise ValueError("Dual-head mode requires data from load_pusht_data_rates.py "
                               "(dict with 'episodes' and 'stats' keys)")
        else:
            episodes = data
            stats = {}

        split_idx = int(cfg.dataset.train_ratio * len(episodes))
        train_eps = episodes[:split_idx]
        val_eps = episodes[split_idx:]

        windows_per_episode = cfg.model.get('windows_per_episode', 16)
        
        if dual_head:
            # Use dual-head dataset
            train_ds = _DualHeadDataset(
                train_eps, stats, normalize_images=True,
                window_length=window_length, is_training=True,
                windows_per_episode=windows_per_episode,
                use_normalized_rates=cfg.model.get('use_normalized_rates', True)
            )
            val_ds = _DualHeadDataset(
                val_eps, stats, normalize_images=True,
                window_length=window_length, is_training=False,
                state_mean=train_ds.state_mean, state_std=train_ds.state_std,
                windows_per_episode=windows_per_episode,
                use_normalized_rates=cfg.model.get('use_normalized_rates', True)
            )
            # Store stats for later use
            self.rate_stats = stats
        else:
            # Use original dataset
            train_ds = _ProgressDataset(train_eps, normalize_images=True, num_bins=num_bins, mode=mode, 
                                        action_chunk_horizon=action_chunk_horizon, windowed=windowed, 
                                        window_length=window_length, is_training=True,
                                        windows_per_episode=windows_per_episode)
            val_ds = _ProgressDataset(
                                        val_eps, normalize_images=True, num_bins=num_bins, mode=mode,
                                        action_chunk_horizon=action_chunk_horizon, windowed=windowed,
                                        window_length=window_length, is_training=False,
                                        state_mean=train_ds.state_mean, state_std=train_ds.state_std,
                                        windows_per_episode=windows_per_episode
                                    )

        train_loader = DataLoader(train_ds, batch_size=cfg.dataloader.batch_size, shuffle=True,
                                  num_workers=cfg.dataloader.num_workers, pin_memory=cfg.dataloader.pin_memory,
                                  persistent_workers=cfg.dataloader.persistent_workers)
        val_loader = DataLoader(val_ds, batch_size=cfg.val_dataloader.batch_size, shuffle=False,
                                num_workers=cfg.val_dataloader.num_workers, pin_memory=cfg.val_dataloader.pin_memory,
                                persistent_workers=cfg.val_dataloader.persistent_workers)
        return train_loader, val_loader

    def _build_model(self):
        cfg = self.cfg
        mode = cfg.model.get('mode', 'categorical')
        freeze_encoder = cfg.model.get('freeze_encoder', cfg.model.pretrained)
        encoder_ckpt_path = cfg.model.get('encoder_ckpt_path', None)
        action_chunk_horizon = cfg.model.get('action_chunk_horizon', 0)
        action_dim = cfg.model.get('action_dim', 2)
        use_siamese = cfg.model.get('use_siamese', True)
        windowed = cfg.model.get('windowed', False)
        architecture = cfg.model.get('architecture', 'gru')  # 'gru' or 'transformer'
        dual_head = cfg.model.get('dual_head', False)
        
        pretrained = cfg.model.pretrained if encoder_ckpt_path is None else False
        
        if dual_head:
            # Dual-head rate-of-change predictor
            model = ProgressPredictorDualHead(
                agent_state_dim=cfg.model.agent_state_dim,
                pretrained=pretrained,
                freeze_encoder=freeze_encoder,
                encoder_ckpt_path=encoder_ckpt_path,
                visual_feat_dim=cfg.model.get('visual_feat_dim', 512),
                state_feat_dim=cfg.model.get('state_feat_dim', 64),
                token_dim=cfg.model.get('token_dim', 256),
                gru_hidden_dim=cfg.model.get('gru_hidden_dim', 128),
                dropout=cfg.model.dropout
            ).to(self.device)
            
            # Print info
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in model.parameters())
            print(f"Model: {trainable_params:,} trainable params / {total_params:,} total params")
            print(f"Architecture: Dual-Head (Push + Reposition)")
            print(f"Window length: L={cfg.model.get('window_length', 8)}")
            if encoder_ckpt_path:
                print(f"Encoder loaded from: {encoder_ckpt_path}")
            if freeze_encoder:
                print("Encoder is FROZEN")
            else:
                print("Encoder is TRAINABLE")
            
            return model
        
        if windowed:
            if architecture == 'transformer':
                # Transformer-based windowed architecture
                model = ProgressPredictorTransformer(
                    agent_state_dim=cfg.model.agent_state_dim,
                    pretrained=pretrained,
                    num_bins=cfg.model.get('num_bins', 50),
                    freeze_encoder=freeze_encoder,
                    mode=mode,
                    encoder_ckpt_path=encoder_ckpt_path,
                    visual_feat_dim=cfg.model.get('visual_feat_dim', 512),
                    state_feat_dim=cfg.model.get('state_feat_dim', 256),
                    token_dim=cfg.model.get('token_dim', 512),
                    n_heads=cfg.model.get('n_heads', 8),
                    n_layers=cfg.model.get('n_layers', 4),
                    dim_feedforward=cfg.model.get('dim_feedforward', 1024),
                    dropout=cfg.model.dropout
                ).to(self.device)
            else:
                # GRU-based windowed architecture
                model = ProgressPredictorWindowed(
                    agent_state_dim=cfg.model.agent_state_dim,
                    pretrained=pretrained,
                    num_bins=cfg.model.get('num_bins', 50),
                    freeze_encoder=freeze_encoder,
                    mode=mode,
                    encoder_ckpt_path=encoder_ckpt_path,
                    visual_feat_dim=cfg.model.get('visual_feat_dim', 512),
                    state_feat_dim=cfg.model.get('state_feat_dim', 64),
                    token_dim=cfg.model.get('token_dim', 256),
                    gru_hidden_dim=cfg.model.get('gru_hidden_dim', 128),
                    dropout=cfg.model.dropout
                ).to(self.device)
        else:
            # Original single-frame architecture
            model = ProgressPredictor(
                agent_state_dim=cfg.model.agent_state_dim,
                pretrained=pretrained,
                dropout=cfg.model.dropout,
                num_bins=cfg.model.get('num_bins', 50),
                freeze_encoder=freeze_encoder,
                mode=mode,
                encoder_ckpt_path=encoder_ckpt_path,
                action_chunk_horizon=action_chunk_horizon,
                action_dim=action_dim,
                use_siamese=use_siamese
            ).to(self.device)
        
        # Print number of trainable params
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model: {trainable_params:,} trainable params / {total_params:,} total params")
        print(f"Mode: {mode.upper()}")
        if windowed:
            arch_name = 'Transformer' if architecture == 'transformer' else 'GRU'
            print(f"Architecture: Windowed {arch_name} (L={cfg.model.get('window_length', 8)})")
        else:
            print(f"Architecture: {'Siamese (two-tower)' if use_siamese else 'Single-tower'}")
        if action_chunk_horizon > 0 and not windowed:
            print(f"Action chunk horizon: {action_chunk_horizon}, Action dim: {action_dim}")
        if encoder_ckpt_path:
            print(f"Encoder loaded from: {encoder_ckpt_path}")
        if freeze_encoder:
            print("Encoder is FROZEN")
        else:
            print("Encoder is TRAINABLE")
        
        return model

    def _train_epoch(self, model, loader, criterion, optimizer, device, lr_getter=None, log_step_fn=None, dual_head=False):
        model.train()
        loss_sum = 0.0
        dual_head_stats = {'push_loss': 0.0, 'repo_loss': 0.0, 'push_count': 0, 'repo_count': 0}
        
        if not dual_head:
            mode = model.mode
            windowed = hasattr(model, 'gru') or hasattr(model, 'transformer')
        
        with tqdm(loader, desc=f"Training epoch {self.epoch}", leave=False, mininterval=self.cfg.training.get('tqdm_interval_sec', 1.0)) as tepoch:
            for batch_idx, batch in enumerate(tepoch):
                optimizer.zero_grad()
                
                if dual_head:
                    # Dual-head architecture
                    images = batch['images'].to(device)
                    states = batch['states'].to(device)
                    is_pushing = batch['is_pushing'].to(device)
                    push_target = batch['push_rate'].to(device)
                    repo_target = batch['repo_rate'].to(device)
                    
                    push_pred, repo_pred = model(images, states)
                    loss, loss_dict = criterion(push_pred, repo_pred, push_target, repo_target, is_pushing)
                    
                    # Accumulate stats
                    dual_head_stats['push_loss'] += loss_dict['push_loss']
                    dual_head_stats['repo_loss'] += loss_dict['repo_loss']
                    dual_head_stats['push_count'] += loss_dict['push_count']
                    dual_head_stats['repo_count'] += loss_dict['repo_count']
                elif windowed:
                    # Windowed architecture: expects sequences
                    images = batch['images'].to(device)
                    states = batch['states'].to(device)
                    output = model(images, states)
                    target = batch['progress'].to(device)
                    
                    if mode == 'categorical':
                        log_probs = torch.log_softmax(output, dim=-1)
                        loss = criterion(log_probs, target)
                    else:
                        loss = criterion(output, target)
                else:
                    # Original architecture
                    action_chunk = batch.get('action_chunk', None)
                    if action_chunk is not None:
                        action_chunk = action_chunk.to(device)
                    
                    if model.use_siamese:
                        start_image = batch.get('start_image', None)
                        if start_image is None:
                            raise ValueError("start_image is required in batch when use_siamese=True")
                        output = model(
                            batch['image'].to(device), 
                            batch['agent_state'].to(device), 
                            action_chunk=action_chunk,
                            start_image=start_image.to(device)
                        )
                    else:
                        output = model(batch['image'].to(device), batch['agent_state'].to(device), action_chunk=action_chunk)
                    
                    target = batch['progress'].to(device)
                    
                    if mode == 'categorical':
                        log_probs = torch.log_softmax(output, dim=-1)
                        loss = criterion(log_probs, target)
                    else:
                        loss = criterion(output, target)
                
                loss.backward()
                optimizer.step()

                loss_val = float(loss.item())
                loss_sum += loss_val
                tepoch.set_postfix(loss=loss_val, refresh=False)

                # log per step to wandb
                if log_step_fn is not None:
                    current_lr = lr_getter() if lr_getter is not None else optimizer.param_groups[0]['lr']
                    is_last_batch = (batch_idx == (len(loader)-1))
                    if not is_last_batch:
                        log_data = {
                            'train_loss': loss_val,
                            'lr': current_lr,
                            'epoch': self.epoch,
                            'global_step': self.global_step
                        }
                        if dual_head:
                            log_data['push_loss'] = loss_dict['push_loss']
                            log_data['repo_loss'] = loss_dict['repo_loss']
                        log_step_fn(log_data, step=self.global_step)
                self.global_step += 1
        
        avg_loss = loss_sum / max(1, len(loader))
        if dual_head:
            return avg_loss, {k: v / max(1, len(loader)) for k, v in dual_head_stats.items()}
        return avg_loss

    def _validate(self, model, loader, criterion, device, dual_head=False):
        model.eval()
        loss_sum = 0.0
        dual_head_stats = {'push_loss': 0.0, 'repo_loss': 0.0, 'push_count': 0, 'repo_count': 0}
        
        if not dual_head:
            mode = model.mode
            windowed = hasattr(model, 'gru') or hasattr(model, 'transformer')
        
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"Validation epoch {self.epoch}", leave=False, mininterval=self.cfg.training.get('tqdm_interval_sec', 1.0)):
                if dual_head:
                    # Dual-head architecture
                    images = batch['images'].to(device)
                    states = batch['states'].to(device)
                    is_pushing = batch['is_pushing'].to(device)
                    push_target = batch['push_rate'].to(device)
                    repo_target = batch['repo_rate'].to(device)
                    
                    push_pred, repo_pred = model(images, states)
                    loss, loss_dict = criterion(push_pred, repo_pred, push_target, repo_target, is_pushing)
                    
                    dual_head_stats['push_loss'] += loss_dict['push_loss']
                    dual_head_stats['repo_loss'] += loss_dict['repo_loss']
                    dual_head_stats['push_count'] += loss_dict['push_count']
                    dual_head_stats['repo_count'] += loss_dict['repo_count']
                elif windowed:
                    # Windowed architecture
                    images = batch['images'].to(device)
                    states = batch['states'].to(device)
                    output = model(images, states)
                    target = batch['progress'].to(device)
                    
                    if mode == 'categorical':
                        log_probs = torch.log_softmax(output, dim=-1)
                        loss = criterion(log_probs, target)
                    else:
                        loss = criterion(output, target)
                else:
                    # Original architecture
                    action_chunk = batch.get('action_chunk', None)
                    if action_chunk is not None:
                        action_chunk = action_chunk.to(device)
                    
                    if model.use_siamese:
                        start_image = batch.get('start_image', None)
                        if start_image is None:
                            raise ValueError("start_image is required in batch when use_siamese=True")
                        output = model(
                            batch['image'].to(device), 
                            batch['agent_state'].to(device), 
                            action_chunk=action_chunk,
                            start_image=start_image.to(device)
                        )
                    else:
                        output = model(batch['image'].to(device), batch['agent_state'].to(device), action_chunk=action_chunk)
                    
                    target = batch['progress'].to(device)
                    
                    if mode == 'categorical':
                        log_probs = torch.log_softmax(output, dim=-1)
                        loss = criterion(log_probs, target)
                    else:
                        loss = criterion(output, target)
                
                loss_sum += float(loss.item())
        
        avg_loss = loss_sum / max(1, len(loader))
        if dual_head:
            return avg_loss, {k: v / max(1, len(loader)) for k, v in dual_head_stats.items()}
        return avg_loss

    def run(self):
        cfg = self.cfg
        dual_head = cfg.model.get('dual_head', False)

        train_loader, val_loader = self._build_dataloaders()
        model = self._build_model()

        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.optimizer.learning_rate,
                                     weight_decay=cfg.optimizer.weight_decay)
        
        # Select loss based on model type
        if dual_head:
            push_weight = cfg.model.get('push_weight', 1.0)
            repo_weight = cfg.model.get('repo_weight', 1.0)
            criterion = DualHeadLoss(push_weight=push_weight, repo_weight=repo_weight)
            print(f"Using DualHeadLoss (push_weight={push_weight}, repo_weight={repo_weight})")
        else:
            mode = cfg.model.get('mode', 'categorical')
            if mode == 'categorical':
                criterion = nn.KLDivLoss(reduction='batchmean')
                print("Using KLDivLoss with Gaussian target distribution (categorical mode)")
            else:  
                criterion = nn.MSELoss()
                print("Using MSELoss (regression mode)")
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=cfg.scheduler.factor, patience=cfg.scheduler.patience, verbose=True)

        best_val = float('inf')
        patience_counter = 0

        num_epochs = cfg.training.num_epochs
        run_dir = os.getcwd()
        print(f"Hydra run dir: {run_dir}")
        # logging: wandb
        wandb_run = wandb.init(
            dir=run_dir,
            config=self.cfg,
            **cfg.logging
        )
        wandb.config.update({"output_dir": run_dir})

        for epoch in range(1, num_epochs + 1):
            self.epoch = epoch
            
            if dual_head:
                train_result = self._train_epoch(
                    model, train_loader, criterion, optimizer, self.device,
                    lr_getter=lambda: optimizer.param_groups[0]['lr'],
                    log_step_fn=lambda data, step: wandb.log(data, step=step),
                    dual_head=True
                )
                train_loss, train_stats = train_result
                
                val_result = self._validate(model, val_loader, criterion, self.device, dual_head=True)
                val_loss, val_stats = val_result
                
                print(f"Epoch {epoch}/{num_epochs}: train={train_loss:.4f} (push={train_stats['push_loss']:.4f}, repo={train_stats['repo_loss']:.4f}), "
                      f"val={val_loss:.4f} (push={val_stats['push_loss']:.4f}, repo={val_stats['repo_loss']:.4f}), lr={optimizer.param_groups[0]['lr']:.6f}")
                
                wandb.log({
                    'train_loss': train_loss,
                    'train_push_loss': train_stats['push_loss'],
                    'train_repo_loss': train_stats['repo_loss'],
                    'val_loss': val_loss,
                    'val_push_loss': val_stats['push_loss'],
                    'val_repo_loss': val_stats['repo_loss'],
                    'epoch': epoch,
                    'global_step': self.global_step,
                    'lr': optimizer.param_groups[0]['lr']
                }, step=self.global_step)
            else:
                train_loss = self._train_epoch(
                    model, train_loader, criterion, optimizer, self.device,
                    lr_getter=lambda: optimizer.param_groups[0]['lr'],
                    log_step_fn=lambda data, step: wandb.log(data, step=step)
                )
                val_loss = self._validate(model, val_loader, criterion, self.device)

                print(f"Epoch {epoch}/{num_epochs}: train={train_loss:.4f}, val={val_loss:.4f}, lr={optimizer.param_groups[0]['lr']:.6f}")
                
                wandb.log({
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'epoch': epoch,
                    'global_step': self.global_step,
                    'lr': optimizer.param_groups[0]['lr']
                }, step=self.global_step)

            scheduler.step(val_loss)

            if val_loss < best_val:
                best_val = val_loss
                patience_counter = 0
                os.makedirs(os.path.dirname(cfg.checkpoint.best_path) or '.', exist_ok=True)
                torch.save(model.state_dict(), cfg.checkpoint.best_path)
                print("Saved new best model")
            else:
                patience_counter += 1

            if patience_counter >= cfg.training.patience:
                print(f"Early stopping at epoch {epoch}")
                break

        print(f"Best model saved to {cfg.checkpoint.best_path}")


