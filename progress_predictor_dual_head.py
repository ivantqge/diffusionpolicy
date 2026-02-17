"""
Dual-Head Progress Predictor for Push-T task.

This model has two specialized heads:
1. Push Head: Predicts rate of T-block approach to goal (when in contact)
2. Reposition Head: Predicts rate of agent approach to next contact position (when repositioning)

Both heads output signed rate values:
- Positive: approaching target
- Negative: moving away from target
- Zero: stationary or perpendicular movement

The phase (pushing vs repositioning) is determined from ground truth labels during training,
and from n_contacts during inference.
"""

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class ProgressPredictorDualHead(nn.Module):
    """
    Dual-Head Progress Predictor with temporal aggregation.
    
    Architecture:
    1. Visual encoder: Shared ResNet18 encoder for each frame
    2. State embedding: 2-layer MLP for 5D state
    3. Token fusion: Linear(concat(zi, ui)) per timestep
    4. Start-only positional embedding: learned start_bias added only to token_0
    5. Temporal aggregator: GRU over tokens
    6. Dual output heads:
       - Push head: predicts rate of T-block approach to goal
       - Reposition head: predicts rate of agent approach to next contact
    
    Input format (for L frames):
        - images: [batch, L, C, H, W] - sequence of L images
        - states: [batch, L, state_dim] - sequence of L states
    
    Output:
        - push_rate: [batch, 1] - signed rate (positive = approaching goal)
        - repo_rate: [batch, 1] - signed rate (positive = approaching next contact)
    """
    
    def __init__(self, 
                 agent_state_dim=5,
                 pretrained=True,
                 freeze_encoder=False,
                 encoder_ckpt_path=None,
                 visual_feat_dim=512,
                 state_feat_dim=64,
                 token_dim=256,
                 gru_hidden_dim=128,
                 dropout=0.1):
        super().__init__()
        self.visual_feat_dim = visual_feat_dim
        self.state_feat_dim = state_feat_dim
        self.token_dim = token_dim
        self.gru_hidden_dim = gru_hidden_dim
        
        # Shared ResNet18 encoder
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        resnet = resnet18(weights=weights)
        
        self.encoder = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4, resnet.avgpool
        )
        
        # Load encoder weights from checkpoint if provided
        if encoder_ckpt_path is not None:
            self.load_encoder_from_checkpoint(encoder_ckpt_path)
        
        # Freeze encoder if set
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        # State embedding: 2-layer MLP
        self.state_embedder = nn.Sequential(
            nn.Linear(agent_state_dim, state_feat_dim),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(state_feat_dim, state_feat_dim),
            nn.GELU()
        )
        
        # Token fusion: Linear(concat(zi, ui))
        self.token_fusion = nn.Linear(visual_feat_dim + state_feat_dim, token_dim)
        
        # Start-only positional embedding: learned bias for token_0
        self.start_bias = nn.Parameter(torch.zeros(1, 1, token_dim))
        
        # Temporal aggregator: GRU
        self.gru = nn.GRU(
            input_size=token_dim,
            hidden_size=gru_hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )
        
        # Layer norm before heads
        self.norm = nn.LayerNorm(gru_hidden_dim)
        
        # Push head: predicts rate of T-block approach to goal
        # Output is unbounded signed float
        self.push_head = nn.Sequential(
            nn.Linear(gru_hidden_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(64, 1)  # unbounded signed output
        )
        
        # Reposition head: predicts rate of agent approach to next contact
        # Output is unbounded signed float
        self.reposition_head = nn.Sequential(
            nn.Linear(gru_hidden_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(64, 1)  # unbounded signed output
        )
    
    def forward(self, images, states):
        """
        Args:
            images: [batch, L, C, H, W] - sequence of L images
            states: [batch, L, state_dim] - sequence of L states (5D)
        
        Returns:
            push_rate: [batch, 1] - predicted push rate
            repo_rate: [batch, 1] - predicted reposition rate
        """
        batch_size, L, C, H, W = images.shape
        
        # Reshape for batch processing: [batch * L, C, H, W]
        images_flat = images.reshape(batch_size * L, C, H, W)
        
        # Encode all images with shared encoder
        visual_features = self.encoder(images_flat).squeeze(-1).squeeze(-1)
        visual_features = visual_features.view(batch_size, L, self.visual_feat_dim)
        
        # Embed states: [batch, L, state_dim] -> [batch, L, state_feat_dim]
        state_features = self.state_embedder(states)
        
        # Fuse per-timestep: concat(zi, ui) -> token_i
        fused = torch.cat([visual_features, state_features], dim=-1)
        tokens = self.token_fusion(fused)
        
        # Add start-only positional embedding
        tokens[:, 0:1, :] = tokens[:, 0:1, :] + self.start_bias
        
        # Temporal aggregation with GRU
        gru_out, hidden = self.gru(tokens)
        
        # Take final hidden state
        final_hidden = hidden.squeeze(0)  # [batch, gru_hidden_dim]
        
        # Layer norm
        final_hidden = self.norm(final_hidden)
        
        # Compute both head outputs
        push_rate = self.push_head(final_hidden)
        repo_rate = self.reposition_head(final_hidden)
        
        return push_rate, repo_rate
    
    def predict_with_phase(self, images, states, is_pushing):
        """
        Predict rate based on phase.
        
        Args:
            images: [batch, L, C, H, W]
            states: [batch, L, state_dim]
            is_pushing: [batch] boolean tensor indicating phase
        
        Returns:
            rate: [batch, 1] - appropriate rate based on phase
        """
        push_rate, repo_rate = self.forward(images, states)
        
        # Select appropriate rate based on phase
        is_pushing = is_pushing.view(-1, 1).float()
        rate = is_pushing * push_rate + (1 - is_pushing) * repo_rate
        
        return rate
    
    def load_encoder_from_checkpoint(self, ckpt_path, encoder_key='rgb'):
        """
        Load ResNet18 encoder weights from a diffusion policy checkpoint.
        """
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        
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
            print(f"Warning: Missing {len(missing_keys)} encoder keys")
        if unexpected_keys:
            print(f"Warning: Unexpected {len(unexpected_keys)} encoder keys")
        
        print(f"Loaded encoder weights from {ckpt_path} (prefix: {found_prefix})")


class DualHeadLoss(nn.Module):
    """
    Phase-masked loss for dual-head progress predictor.
    
    Only trains the relevant head based on the current phase:
    - Push head is trained when is_pushing=True
    - Reposition head is trained when is_pushing=False
    """
    
    def __init__(self, push_weight=1.0, repo_weight=1.0):
        super().__init__()
        self.push_weight = push_weight
        self.repo_weight = repo_weight
        self.mse = nn.MSELoss(reduction='none')
    
    def forward(self, push_pred, repo_pred, push_target, repo_target, is_pushing):
        """
        Compute phase-masked loss.
        
        Args:
            push_pred: [batch, 1] predicted push rate
            repo_pred: [batch, 1] predicted reposition rate
            push_target: [batch, 1] target push rate
            repo_target: [batch, 1] target reposition rate
            is_pushing: [batch] boolean tensor indicating phase
        
        Returns:
            loss: scalar loss value
            loss_dict: dictionary with individual losses for logging
        """
        batch_size = push_pred.shape[0]
        
        # Create masks
        push_mask = is_pushing.view(-1, 1).float()
        repo_mask = 1.0 - push_mask
        
        # Compute per-sample losses
        push_loss_per_sample = self.mse(push_pred, push_target)
        repo_loss_per_sample = self.mse(repo_pred, repo_target)
        
        # Apply masks and compute mean (avoiding division by zero)
        push_count = push_mask.sum() + 1e-6
        repo_count = repo_mask.sum() + 1e-6
        
        push_loss = (push_mask * push_loss_per_sample).sum() / push_count
        repo_loss = (repo_mask * repo_loss_per_sample).sum() / repo_count
        
        # Weighted total loss
        total_loss = self.push_weight * push_loss + self.repo_weight * repo_loss
        
        loss_dict = {
            'push_loss': push_loss.item(),
            'repo_loss': repo_loss.item(),
            'push_count': int(push_mask.sum().item()),
            'repo_count': int(repo_mask.sum().item()),
        }
        
        return total_loss, loss_dict


if __name__ == "__main__":
    # Test the model
    model = ProgressPredictorDualHead()
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test forward pass
    batch_size = 4
    L = 8
    images = torch.randn(batch_size, L, 3, 96, 96)
    states = torch.randn(batch_size, L, 5)
    
    push_rate, repo_rate = model(images, states)
    print(f"Push rate shape: {push_rate.shape}")
    print(f"Repo rate shape: {repo_rate.shape}")
    print(f"Push rate values: {push_rate.squeeze().tolist()}")
    print(f"Repo rate values: {repo_rate.squeeze().tolist()}")
    
    # Test loss computation
    criterion = DualHeadLoss()
    push_target = torch.randn(batch_size, 1)
    repo_target = torch.randn(batch_size, 1)
    is_pushing = torch.tensor([True, True, False, False])
    
    loss, loss_dict = criterion(push_rate, repo_rate, push_target, repo_target, is_pushing)
    print(f"\nLoss: {loss.item():.4f}")
    print(f"Loss dict: {loss_dict}")
