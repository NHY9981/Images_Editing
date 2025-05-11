import torch
import torch.nn as nn
import timm
import math

class EfficientViT(nn.Module):
    def __init__(self,
                 img_size=384,
                 cnn_model_name='efficientnetv2_s',
                 # Transformer Encoder parameters
                 embed_dim=384,      # Dimension of tokens going into Transformer
                 depth=6,            # Number of Transformer encoder layers
                 num_heads=6,        # Number of attention heads
                 mlp_ratio=4.0,      # Ratio for MLP hidden dim in Transformer
                 dropout_rate=0.1,
                 # MLP Head parameters
                 head_hidden_dim=128):
        super().__init__()
        self.img_size = img_size
        self.embed_dim = embed_dim

        # 1. CNN Backbone (EfficientNetV2-S)
        # We want pretrained weights for feature extraction
        self.cnn_backbone = timm.create_model(
            cnn_model_name,
            pretrained=False, # Use True if you want to leverage ImageNet pretraining
            features_only=True,
            # output_stride=32 # Not strictly needed with features_only, let's rely on feature_info
        )
        
        # Get the number of output channels from the last feature map of the CNN
        cnn_feature_info = self.cnn_backbone.feature_info.channels()[-1]
        
        # Calculate the spatial dimensions of the CNN output feature map
        # Assuming the total stride of efficientnetv2_s for the last feature map is 32
        # For a 384x384 input, this would be 384/32 = 12x12
        # We can also try to get it dynamically, but for now, let's assume stride 32.
        # A more robust way would be a dummy forward pass if this was highly variable.
        self.cnn_output_stride = 32 # Common for EfficientNets last stage
        self.num_patches_h = img_size // self.cnn_output_stride
        self.num_patches_w = img_size // self.cnn_output_stride
        self.num_patches = self.num_patches_h * self.num_patches_w # e.g., 12*12 = 144 for 384 input

        # 2. Linear Projection (similar to Patch Embedding in ViT)
        # This projects the CNN feature map channels to the Transformer's embed_dim
        # It's effectively a 1x1 convolution.
        self.linear_proj = nn.Conv2d(cnn_feature_info, embed_dim, kernel_size=1)

        # 3. CLS Token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # 4. Positional Embedding
        # For CLS token + all patches from CNN
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout_rate)

        # 5. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout_rate,
            activation='gelu', # GELU is common in Transformers
            batch_first=True   # Crucial: (batch, seq, feature)
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=depth
        )

        # 6. MLP Head
        # Norm before feeding to the head (common practice)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, head_hidden_dim),
            nn.GELU(), # Or nn.ReLU()
            nn.Dropout(dropout_rate),
            nn.Linear(head_hidden_dim, 1),
            nn.Sigmoid() # For probability output
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        # Positional encoding (sinusoidal or learned)
        # Using sinusoidal for fixed pos_embed
        pos_embed_sinusoid = self.get_sinusoid_encoding(self.num_patches + 1, self.embed_dim)
        self.pos_embed.data.copy_(pos_embed_sinusoid.float().unsqueeze(0))

        # CLS token
        nn.init.normal_(self.cls_token, std=.02)
        
        # Initialize Linear Projection
        nn.init.kaiming_normal_(self.linear_proj.weight, mode='fan_out', nonlinearity='relu')
        if self.linear_proj.bias is not None:
            nn.init.zeros_(self.linear_proj.bias)

        # Initialize MLP Head
        for m in self.head:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        # Transformer weights are typically initialized well by PyTorch defaults,
        # but one could apply custom init if needed.
        # LayerNorm weights are typically 1, bias 0 (PyTorch default)

    def get_sinusoid_encoding(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        def get_position_angle_vec(position):
            return [position / math.pow(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = torch.FloatTensor([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = torch.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = torch.cos(sinusoid_table[:, 1::2])  # dim 2i+1
        return sinusoid_table

    def forward(self, x):
        B = x.shape[0]

        # 1. CNN Backbone
        # `features_only=True` returns a list of feature maps from different stages.
        # We usually take the last one for this kind of architecture.
        cnn_features = self.cnn_backbone(x)[-1] # Shape: (B, cnn_feature_info, H_cnn_out, W_cnn_out)
                                                # e.g., (B, 512, 12, 12) for efficientnetv2_s with 384 input

        # 2. Linear Projection
        x = self.linear_proj(cnn_features)      # Shape: (B, embed_dim, H_cnn_out, W_cnn_out)
                                                # e.g., (B, 384, 12, 12)
        
        # Flatten spatial dimensions and permute to (Batch, NumPatches, EmbedDim)
        x = x.flatten(2).transpose(1, 2)        # Shape: (B, num_patches, embed_dim)
                                                # e.g., (B, 144, 384)

        # 3. Prepend CLS Token
        cls_tokens = self.cls_token.expand(B, -1, -1) # Shape: (B, 1, embed_dim)
        x = torch.cat((cls_tokens, x), dim=1)         # Shape: (B, num_patches + 1, embed_dim)

        # 4. Add Positional Embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # 5. Transformer Encoder
        x = self.transformer_encoder(x) # Shape: (B, num_patches + 1, embed_dim)

        # 6. MLP Head
        # We take the output corresponding to the CLS token
        cls_output = x[:, 0]            # Shape: (B, embed_dim)
        cls_output = self.norm(cls_output) # Apply LayerNorm
        
        output_prob = self.head(cls_output) # Shape: (B, 1)

        return output_prob

if __name__ == '__main__':
    # Example Usage:
    img_size = 384
    batch_size = 4

    # Model instantiation
    model = EfficientViT(
        img_size=img_size,
        cnn_model_name='efficientnetv2_s', # Or other efficientnet variants
        embed_dim=384,        # Keep consistent with ViT literature if possible, or adjust for efficiency
        depth=6,              # Number of transformer layers (e.g., ViT-Small has 8-12)
        num_heads=6,          # embed_dim must be divisible by num_heads
        mlp_ratio=2.0,        # Lighter MLP in transformer blocks
        dropout_rate=0.1,
        head_hidden_dim=128
    )

    # Create a dummy input tensor
    dummy_input = torch.randn(batch_size, 3, img_size, img_size)

    # Perform a forward pass
    print(f"Input shape: {dummy_input.shape}")
    output = model(dummy_input)
    print(f"Output shape: {output.shape}") # Should be (batch_size, 1)
    print(f"Output example:\n{output}")

    # Check number of parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")
    