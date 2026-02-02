import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class ProgressPredictor(nn.Module):
    """
    Predicts task progress for the pushT task using encoder architecture.
    Input: start_image (C, H, W) + current_image (C, H, W) + agent_state (agent_state_dim,) + action_chunk (horizon, action_dim)
           where action_chunk contains future actions: [action[t], action[t+1], ..., action[t+horizon-1]]
    Output: 
        - If categorical mode: logits for num_bins classes 
        - If regression mode: single value [0, 1] 
    
    Architecture:
        - Encoder: shared ResNet18 encodes both start and current images
        - Fusion: [z0, zt, zt - z0, zt ⊙ z0] where z0 = encoded start_image, zt = encoded current_image
        - MLP predictor head
    """
    
    def __init__(self, agent_state_dim=5, pretrained=False, dropout=0.3, num_bins=50, freeze_encoder=False, 
                       mode='categorical', encoder_ckpt_path=None, action_chunk_horizon=0, action_dim=2, use_siamese=True):
        super().__init__()
        self.num_bins = num_bins
        self.mode = mode
        self.action_chunk_horizon = action_chunk_horizon
        self.action_dim = action_dim
        self.use_siamese = use_siamese
        
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        resnet = resnet18(weights=weights)
        
        # Shared weights for encoding both start and current images
        self.encoder = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4, resnet.avgpool
        )
        
        # Load encoder weights from diffusion policy checkpoint if provided
        if encoder_ckpt_path is not None:
            self.load_encoder_from_checkpoint(encoder_ckpt_path)
        
        # Freeze encoder if set to True
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        # Calculate input dimension
        # With Siamese: fused visual features (512*4 = 2048) + agent_state + action_chunk
        # Without Siamese (backward compat): visual state (512) + agent_state + action_chunk
        if use_siamese:
            # Fusion: [z0, zt, zt - z0, zt ⊙ z0] = 4 * 512 = 2048 dimensions
            visual_feat_dim = 512 * 4
        else:
            visual_feat_dim = 512
        
        action_chunk_dim = action_chunk_horizon * action_dim if action_chunk_horizon > 0 else 0
        mlp_input_dim = visual_feat_dim + agent_state_dim + action_chunk_dim
        
        # MLP predictor
        if mode == 'categorical':
            self.predictor = nn.Sequential(
                nn.Linear(mlp_input_dim, 256),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(dropout / 2),
                nn.Linear(64, num_bins)
            )
        else:  # regression, outputs singular value between 0 and 1
            self.predictor = nn.Sequential(
                nn.Linear(mlp_input_dim, 256),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(dropout / 2),
                nn.Linear(64, 1),
                nn.Sigmoid() 
            )
        
    def forward(self, image, agent_state, action_chunk=None, start_image=None):
        """
        Args:
            image: [batch, C, H, W] - current image
            agent_state: [batch, agent_state_dim]
            action_chunk: [batch, horizon, action_dim] or None if action_chunk_horizon == 0
                          Contains future actions: [action[t], action[t+1], ..., action[t+horizon-1]]
            start_image: [batch, C, H, W] - start/initial image (required if use_siamese=True)
        """
        if self.use_siamese:
            if start_image is None:
                raise ValueError("start_image is required when use_siamese=True")
            
            # Encode both start and current images with shared encoder
            z0 = self.encoder(start_image).squeeze(-1).squeeze(-1)  # [batch, 512]
            zt = self.encoder(image).squeeze(-1).squeeze(-1)  # [batch, 512]
            
            # Fusion: [z0, zt, zt - z0, zt ⊙ z0]
            z_diff = zt - z0  # [batch, 512]
            z_mul = zt * z0  # [batch, 512] element-wise multiplication
            feat = torch.cat([z0, zt, z_diff, z_mul], dim=1)  # [batch, 2048]
        else:
            # Backward compatibility: single image encoding
            feat = self.encoder(image).squeeze(-1).squeeze(-1)  # [batch, 512]
        
        # Concatenate visual feature with action chunks
        if self.action_chunk_horizon > 0 and action_chunk is not None:
            action_chunk_flat = action_chunk.view(feat.shape[0], -1)
            feat = torch.cat([feat, action_chunk_flat], dim=1)  
        
        # Concatenate with agent state
        x = torch.cat([feat, agent_state], dim=1) 
        output = self.predictor(x)
        
        return output
    
    def progress_from_logits(self, logits):
        """
        Convert logits to progress value [0, 1] using expected value from distribution
        Computes weighted average: sum(prob[i] * bin_center[i])
        """
        probs = torch.softmax(logits, dim=-1) 
        
        # Compute bin centers
        bin_centers = torch.arange(0.5, self.num_bins, dtype=probs.dtype, device=probs.device) / self.num_bins
        
        progress = torch.sum(probs * bin_centers.unsqueeze(0), dim=-1) 
        return progress
    
    def get_distribution(self, logits):
        """
        Get probability distribution over bins
        Returns: [batch, num_bins] probability tensor
        """
        return torch.softmax(logits, dim=-1)
    
    def load_encoder_from_checkpoint(self, ckpt_path, encoder_key='rgb'):
        """
        Load ResNet18 encoder weights from a diffusion policy checkpoint
        
        Args:
            ckpt_path: Path to diffusion policy checkpoint (.ckpt file)
            encoder_key: Key in obs_encoder.key_model_map (usually 'rgb' or 'image')
        
        The checkpoint should have encoder weights under:
        - policy.obs_encoder.key_model_map.{encoder_key}.*
        """
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        
        # Find encoder weights
        encoder_prefixes = [
            f'policy.obs_encoder.key_model_map.{encoder_key}.',
            f'obs_encoder.key_model_map.{encoder_key}.',
            f'key_model_map.{encoder_key}.',
        ]
        
        encoder_state_dict = {}
        found_prefix = None
        
        for prefix in encoder_prefixes:
            for key, value in state_dict.items():
                if key.startswith(prefix):
                    found_prefix = prefix
                    suffix = key[len(prefix):]
                    encoder_state_dict[suffix] = value
            
            if encoder_state_dict:
                break
      
        missing_keys, unexpected_keys = self.encoder.load_state_dict(encoder_state_dict, strict=False)
        if missing_keys:
            print(f"Warning: Missing {len(missing_keys)} encoder keys (e.g., {missing_keys[:3]}...)")
        if unexpected_keys:
            print(f"Warning: Unexpected {len(unexpected_keys)} encoder keys (e.g., {unexpected_keys[:3]}...)")
        
        print(f"Loaded encoder weights from {ckpt_path} (prefix: {found_prefix})")
        print(f"Successfully loaded {len(encoder_state_dict) - len(missing_keys)}/{len(encoder_state_dict)} encoder parameters")

