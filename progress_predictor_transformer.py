import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class ProgressPredictorTransformer(nn.Module):
    """
    Windowed Progress Predictor with Transformer temporal aggregation.
    
    Architecture:
    1. Visual encoder: Shared ResNet18 encoder for each frame
    2. State embedding: 2-layer MLP for 5D state
    3. Token fusion: Linear(concat(zi, ui)) per timestep
    4. Start-only positional embedding: learned start_bias added only to token_0
    5. Temporal aggregator: Transformer encoder
    6. Output head: MLP for progress prediction (using mean pooling)
    
    Input format (for L frames):
        - images: [batch, L, C, H, W] - sequence of L images
        - states: [batch, L, state_dim] - sequence of L states
    
    Output:
        - If categorical mode: logits [batch, num_bins]
        - If regression mode: progress [batch, 1] in [0, 1]
    """
    
    def __init__(self, 
                 agent_state_dim=5,
                 pretrained=True,
                 num_bins=50,
                 freeze_encoder=False,
                 mode='categorical',
                 encoder_ckpt_path=None,
                 visual_feat_dim=512,
                 state_feat_dim=256,
                 token_dim=512,
                 n_heads=8,
                 n_layers=4,
                 dim_feedforward=1024,
                 dropout=0.1):
        super().__init__()
        self.num_bins = num_bins
        self.mode = mode
        self.visual_feat_dim = visual_feat_dim
        self.state_feat_dim = state_feat_dim
        self.token_dim = token_dim
        
        # Shared ResNet18
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        resnet = resnet18(weights=weights)
        
        self.encoder = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4, resnet.avgpool
        )
        
        if encoder_ckpt_path is not None:
            self.load_encoder_from_checkpoint(encoder_ckpt_path)
        
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
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-norm for stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Layer norm before output head
        self.norm = nn.LayerNorm(token_dim)
        
        # Output head: MLP for progress prediction
        if mode == 'categorical':
            self.predictor = nn.Sequential(
                nn.Linear(token_dim, 256),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(256, 128),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(128, 64),
                nn.GELU(),
                nn.Dropout(dropout / 2),
                nn.Linear(64, num_bins)
            )
        else:  # regression
            self.predictor = nn.Sequential(
                nn.Linear(token_dim, 256),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(256, 128),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(128, 64),
                nn.GELU(),
                nn.Dropout(dropout / 2),
                nn.Linear(64, 1),
                nn.Sigmoid()
            )
    
    def forward(self, images, states):
        """
        Args:
            images: [batch, L, C, H, W] - sequence of L images
            states: [batch, L, state_dim] - sequence of L states (5D)
        
        Returns:
            output: [batch, num_bins] (categorical) or [batch, 1] (regression)
        """
        batch_size, L, C, H, W = images.shape
        
        # Encode all images: [batch * L, C, H, W] -> [batch, L, visual_feat_dim]
        images_flat = images.reshape(batch_size * L, C, H, W)
        visual_features = self.encoder(images_flat).squeeze(-1).squeeze(-1)
        visual_features = visual_features.view(batch_size, L, self.visual_feat_dim)
        
        # Embed states: [batch, L, state_feat_dim]
        state_features = self.state_embedder(states)
        
        # Fuse: [batch, L, token_dim]
        fused = torch.cat([visual_features, state_features], dim=-1)
        tokens = self.token_fusion(fused)
        
        # Add start-only bias to first token
        tokens[:, 0:1, :] = tokens[:, 0:1, :] + self.start_bias
        
        # Transformer encoding
        transformer_out = self.transformer(tokens)
        
        # Mean pooling over sequence
        pooled = transformer_out.mean(dim=1)
        
        # Layer norm
        pooled = self.norm(pooled)
        
        # Predict progress
        output = self.predictor(pooled)
        
        return output
    
    def progress_from_logits(self, logits):
        probs = torch.softmax(logits, dim=-1)
        bin_centers = torch.arange(0.5, self.num_bins, dtype=probs.dtype, device=probs.device) / self.num_bins
        progress = torch.sum(probs * bin_centers.unsqueeze(0), dim=-1)
        return progress
    
    def get_distribution(self, logits):
        return torch.softmax(logits, dim=-1)
    
    def load_encoder_from_checkpoint(self, ckpt_path, encoder_key='rgb'):
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


if __name__ == "__main__":
    model = ProgressPredictorTransformer(mode='categorical', n_layers=4)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Test forward pass
    images = torch.randn(4, 8, 3, 96, 96)
    states = torch.randn(4, 8, 5)
    output = model(images, states)
    print(f"Output shape: {output.shape}")
