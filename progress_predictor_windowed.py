import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class ProgressPredictorWindowed(nn.Module):
    """
    Windowed Progress Predictor with temporal aggregation.
    
    Architecture is as follows:
    1. Visual encoder: Shared ResNet18 encoder (pre-trained for now) for each frame
    2. State embedding: 2-layer MLP for a 5D state (agent_x, agent_y, block_x, block_y, block_angle)
    3. Token fusion: Linear(concat(zi, ui)) per timestep
    4. Start-only positional embedding: learned start_bias added only to token_0
    5. Temporal aggregator: GRU over tokens
    6. Output head: MLP for progress prediction (categorical or regression)
    
    Input format (for L frames):
        - images: [batch, L, C, H, W] - sequence of L images
        - states: [batch, L, state_dim] - sequence of L states (5D: agent_x, agent_y, block_x, block_y, block_angle)
        - Note: images[0] is the start frame o0
    
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
                 gru_hidden_dim=256,
                 dropout=0.1):
        super().__init__()
        self.num_bins = num_bins
        self.mode = mode
        self.visual_feat_dim = visual_feat_dim
        self.state_feat_dim = state_feat_dim
        self.token_dim = token_dim
        self.gru_hidden_dim = gru_hidden_dim
        
        # Shared ResNet18
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
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(state_feat_dim, state_feat_dim),
            nn.ReLU()
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
        
        # Output head: MLP for progress prediction
        if mode == 'categorical':
            self.predictor = nn.Sequential(
                nn.Linear(gru_hidden_dim, 256),
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
        else:  # regression
            self.predictor = nn.Sequential(
                nn.Linear(gru_hidden_dim, 256),
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
    
    def forward(self, images, states):
        """
        Args:
            images: [batch, L, C, H, W] - sequence of L images
                   images[:, 0, :, :, :] should be the start frame o0
            states: [batch, L, state_dim] - sequence of L states (5D)
        
        Returns:
            output: [batch, num_bins] (categorical) or [batch, 1] (regression)
        """
        batch_size, L, C, H, W = images.shape
        
        # Reshape for batch processing: [batch * L, C, H, W]
        images_flat = images.reshape(batch_size * L, C, H, W)
        
        # Encode all images with shared encoder, 
        # output: [batch * L, visual_feat_dim]
        visual_features = self.encoder(images_flat).squeeze(-1).squeeze(-1)
        visual_features = visual_features.view(batch_size, L, self.visual_feat_dim)
        
        # Embed states: [batch, L, state_dim] -> [batch, L, state_feat_dim]
        state_features = self.state_embedder(states)
        
        # Fuse per-timestep: concat(zi, ui) -> token_i
        # [batch, L, visual_feat_dim + state_feat_dim]
        fused = torch.cat([visual_features, state_features], dim=-1)
        # [batch, L, token_dim]
        tokens = self.token_fusion(fused)
        
        # Add start-only positional embedding: token_0 += start_bias
        # start_bias: [1, 1, token_dim] -> [token_dim]
        start_bias_expanded = self.start_bias.squeeze(0)  # [1, token_dim]
        tokens[:, 0:1, :] = tokens[:, 0:1, :] + start_bias_expanded
        
        gru_out, hidden = self.gru(tokens)
        
        # Take final hidden state (from last timestep)
        # hidden: [1, batch, gru_hidden_dim] -> [batch, gru_hidden_dim]
        final_hidden = hidden.squeeze(0)
        
        # Predict progress
        output = self.predictor(final_hidden)
        
        return output
    
    def progress_from_logits(self, logits):
        """
        Convert logits to progress value [0, 1] using expected value from distribution
        """
        probs = torch.softmax(logits, dim=-1)
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
        print(f"Successfully loaded {len(encoder_state_dict) - len(missing_keys)}/{len(encoder_state_dict)} encoder parameters")
