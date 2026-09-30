import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
# Import DINOv3 components
from dinov3.layers import SelfAttentionBlock, RopePositionEmbedding, LayerScale, Mlp
from dinov3.layers.attention import SelfAttention
import math

class LocationEncoder(nn.Module):
    """Location Encoder to process bbox coordinates (x,y,w,h)"""
    def __init__(self, embed_dim=768, coord_dim=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.coord_dim = coord_dim
        
        # Project bbox coordinates to embedding dimension
        self.coord_proj = nn.Sequential(
            nn.Linear(coord_dim, embed_dim // 4),
            nn.LayerNorm(embed_dim // 4),
            nn.ReLU(),
            nn.Linear(embed_dim // 4, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, embed_dim)
        )
        
        # Positional encoding for location
        self.pos_encoding = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # Layer norm for final output
        self.layer_norm = nn.LayerNorm(embed_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, bbox_coords):
        """
        Args:
            bbox_coords: (B, 4) - bbox coordinates (x_center, y_center, width, height) in [0,1]
        Returns:
            location_embeddings: (B, 1, embed_dim) - location embeddings
        """
        # Project coordinates to embedding space
        coord_embed = self.coord_proj(bbox_coords)  # (B, embed_dim)
        
        # Add positional encoding
        coord_embed = coord_embed.unsqueeze(1) + self.pos_encoding  # (B, 1, embed_dim)
        
        # Layer normalization
        coord_embed = self.layer_norm(coord_embed)
        
        return coord_embed

# class LocationEncoder(nn.Module):
#     """Encode contour points using INR-style positional encoding"""
#     def __init__(self, embed_dim=768, num_freq_bands=64, num_points=128):
#         super().__init__()
#         self.embed_dim = embed_dim
#         self.num_freq_bands = num_freq_bands
#         self.num_points = num_points  # Fixed number of sampled points
        
#         # Frequency bands for sinusoidal encoding (类似你的DepthEncoder)
#         self.freq_bands = nn.Parameter(
#             2**(torch.linspace(0, 6, num_freq_bands)), 
#             requires_grad=False
#         )
        
#         # Phase generator for adaptive encoding
#         self.phase_generator = nn.Sequential(
#             nn.Linear(2, 128),  # x, y coordinates
#             nn.SiLU(),
#             nn.Linear(128, 128),
#             nn.SiLU(),
#             nn.Linear(128, num_freq_bands)
#         )
        
#         # Project encoded points to embedding dimension
#         # Input: 2 (x,y) + 2*num_freq_bands (sin/cos)
#         input_dim = 2 + 2 * num_freq_bands
        
#         self.point_proj = nn.Sequential(
#             nn.Linear(input_dim, embed_dim // 2),
#             nn.LayerNorm(embed_dim // 2),
#             nn.GELU(),
#             nn.Linear(embed_dim // 2, embed_dim)
#         )
        
#         # Aggregate multiple points (Set-based aggregation)
#         self.point_aggregator = nn.MultiheadAttention(
#             embed_dim=embed_dim,
#             num_heads=8,
#             batch_first=True
#         )
        
#         # Learnable query for aggregation
#         self.contour_query = nn.Parameter(torch.randn(1, 1, embed_dim))
        
#         self.layer_norm = nn.LayerNorm(embed_dim)
        
#     def forward(self, contour_points):
#         """
#         Args:
#             contour_points: (B, N, 2) - sampled contour points (x, y) in [0, 1]
#         Returns:
#             contour_embeddings: (B, 1, embed_dim)
#         """
#         B, N, _ = contour_points.shape
        
#         # Generate phase for each point
#         phi = self.phase_generator(contour_points)  # (B, N, num_freq_bands)
        
#         # Apply sinusoidal encoding (类似你的DepthEncoder)
#         # (B, N, 2) * (num_freq_bands,) -> (B, N, num_freq_bands)
#         coord_scaled = contour_points.unsqueeze(-1) * self.freq_bands.unsqueeze(0).unsqueeze(0)  # (B, N, 2, num_freq_bands)
#         coord_sin = torch.sin(coord_scaled + phi.unsqueeze(2))  # (B, N, 2, num_freq_bands)
#         coord_cos = torch.cos(coord_scaled + phi.unsqueeze(2))  # (B, N, 2, num_freq_bands)
        
#         # Flatten frequency dimension
#         coord_sin = coord_sin.flatten(-2)  # (B, N, 2*num_freq_bands)
#         coord_cos = coord_cos.flatten(-2)
        
#         # Concatenate: [x, y, sin(...), cos(...)]
#         point_features = torch.cat([
#             contour_points,  # (B, N, 2)
#             coord_sin,       # (B, N, 2*num_freq_bands)
#             coord_cos        # (B, N, 2*num_freq_bands)
#         ], dim=-1)
        
#         # Project each point
#         point_embeds = self.point_proj(point_features)  # (B, N, embed_dim)
        
#         # Aggregate points using cross-attention
#         contour_query = self.contour_query.expand(B, -1, -1)  # (B, 1, embed_dim)
#         contour_embed, _ = self.point_aggregator(
#             query=contour_query,
#             key=point_embeds,
#             value=point_embeds
#         )
        
#         return self.layer_norm(contour_embed)

class DecoderBlock(nn.Module):
    """
    Cross-attention block that combines DINOv3's SelfAttentionBlock with cross-attention capability
    """
    def __init__(self, embed_dim, num_heads, ffn_ratio=4.0, qkv_bias=True, proj_bias=True, 
                 ffn_bias=True, drop_path=0.0, layerscale_init=None, norm_layer=nn.LayerNorm,
                 ffn_layer=Mlp, device=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # Simplified cross-attention using standard MultiheadAttention
        self.cross_norm1 = norm_layer(embed_dim)
        self.cross_norm2 = norm_layer(embed_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=0.0,
            bias=qkv_bias,
            batch_first=True,
            device=device
        )
        self.cross_ls = LayerScale(embed_dim, init_values=layerscale_init, device=device) if layerscale_init else nn.Identity()
        
        # Self-attention block using DINOv3's implementation
        self.self_attn_block = SelfAttentionBlock(
            dim=embed_dim,
            num_heads=num_heads,
            ffn_ratio=ffn_ratio,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            ffn_bias=ffn_bias,
            drop_path=drop_path,
            norm_layer=norm_layer,
            ffn_layer=ffn_layer,
            init_values=layerscale_init,
            device=device
        )
    
    def forward(self, features, context_embeddings, rope=None):
        """
        Args:
            features: (B, N, embed_dim) - input features
            context_embeddings: (B, M, embed_dim) - class or location embeddings  
            rope: RoPE embeddings for spatial features
        """
        # Cross-attention: features attend to context embeddings
        features_norm = self.cross_norm1(features)
        context_norm = self.cross_norm2(context_embeddings)
        
        cross_attn_out, _ = self.cross_attn(
            query=features_norm,
            key=context_norm, 
            value=context_norm
        )
        
        # Apply cross-attention residual
        features = features + self.cross_ls(cross_attn_out)
        
        # Self-attention with RoPE - use the correct forward method
        features = self.self_attn_block(features, rope_or_rope_list=rope)
        
        return features


class LightweightDeconvHead(nn.Module):
    """Lightweight deconvolution head for mask prediction"""
    def __init__(self, embed_dim, patch_size=16):
        super().__init__()
        self.patch_size = patch_size
        
        # Project features to higher dimension for better representation
        self.feature_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        
        # Lightweight deconvolution layers
        self.deconv1 = nn.ConvTranspose2d(embed_dim, embed_dim // 2, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(embed_dim // 2, embed_dim // 4, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(embed_dim // 4, embed_dim // 8, kernel_size=4, stride=2, padding=1)
        self.deconv4 = nn.ConvTranspose2d(embed_dim // 8, embed_dim // 16, kernel_size=4, stride=2, padding=1)
        
        # Final prediction head - output single channel mask
        self.final_conv = nn.Conv2d(embed_dim // 16, 1, kernel_size=1)
        
        # Batch normalization for stable training
        self.bn1 = nn.BatchNorm2d(embed_dim // 2)
        self.bn2 = nn.BatchNorm2d(embed_dim // 4)
        self.bn3 = nn.BatchNorm2d(embed_dim // 8)
        self.bn4 = nn.BatchNorm2d(embed_dim // 16)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, features, H, W):
        """
        Args:
            features: (B, H*W, embed_dim) - decoder output features
            H, W: height and width of feature map
        """
        # Project features
        features = self.feature_proj(features)  # (B, H*W, embed_dim)
        
        # Reshape to spatial format
        features = features.transpose(1, 2).reshape(-1, features.size(-1), H, W)  # (B, embed_dim, H, W)
        
        # Deconvolution layers with residual connections
        x1 = F.relu(self.bn1(self.deconv1(features)))
        x1 = self.dropout(x1)
        
        x2 = F.relu(self.bn2(self.deconv2(x1)))
        x2 = self.dropout(x2)
        
        x3 = F.relu(self.bn3(self.deconv3(x2)))
        x3 = self.dropout(x3)
        
        x4 = F.relu(self.bn4(self.deconv4(x3)))
        x4 = self.dropout(x4)
        
        # Final prediction - single channel mask
        logits = self.final_conv(x4)  # (B, 1, H*16, W*16)
        
        return logits

class LightWeightMLPHead(nn.Module):
    """Lightweight MLP head for class prediction"""
    def __init__(self, embed_dim, num_classes):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, embed_dim // 4),
            nn.LayerNorm(embed_dim // 4),
            nn.GELU(),
            nn.Linear(embed_dim // 4, num_classes),
            nn.LayerNorm(num_classes)
        )
        
    def forward(self, features):
        """
        Args:
            features: (B, H*W, embed_dim) - decoder output features
        """
        return self.mlp(features)

class Model(nn.Module):
    def __init__(self, backbone, config):
        super().__init__()
        self.backbone = backbone
        self.config = config
        
        if 'semantic' in config.prompt_type:
            # Class embeddings for each class
            self.class_embed = nn.Embedding(config.num_classes, backbone.embed_dim)
            # Lightweight deconvolution head
            self.deconv_head = LightweightDeconvHead(
                embed_dim=backbone.embed_dim,
                patch_size=backbone.patch_size
            )
        
        if 'location' in config.prompt_type:        
            # Location embedding for a instance
            self.location_embed = LocationEncoder(
                embed_dim=backbone.embed_dim,
                coord_dim=4
            )
        
            # class token for class prediction
            self.class_token = nn.Parameter(torch.randn(1, 1, backbone.embed_dim))
            
            # Lightweight MLP head
            self.mlp_head = LightWeightMLPHead(
                embed_dim=backbone.embed_dim,
                num_classes=config.num_classes
            )
            
        # RoPE position embedding for decoder (same as backbone)
        self.rope_embed = RopePositionEmbedding(
            embed_dim=backbone.embed_dim,
            num_heads=backbone.num_heads,
            base=100.0,  # Same as DINOv3 default
            normalize_coords="separate",
            dtype=None,  # Let it use default dtype for consistency
        )
        
        # Decoder blocks using DINOv3-style components
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(
                embed_dim=backbone.embed_dim,
                num_heads=backbone.num_heads,
                ffn_ratio=4.0,
                qkv_bias=True,
                proj_bias=True,
                ffn_bias=True,
                drop_path=config.drop_path if hasattr(config, 'drop_path') else 0.0,
                layerscale_init=1,
                norm_layer=nn.LayerNorm,
                ffn_layer=Mlp,
                device=None  # Let PyTorch handle device placement
            ) for _ in range(config.num_decoder_layers)
        ])
        

        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        # Initialize RoPE
        self.rope_embed._init_weights()
        
        if 'semantic' in self.config.prompt_type:
            # Initialize class embeddings
            nn.init.normal_(self.class_embed.weight, std=0.02)
        
        if 'location' in self.config.prompt_type:
            # Initialize class token
            nn.init.normal_(self.class_token, std=0.02)
        
        # Initialize LayerScale parameters in decoder blocks
        for block in self.decoder_blocks:
            if hasattr(block, 'cross_attn'):
                nn.init.xavier_uniform_(block.cross_attn.in_proj_weight)
                nn.init.xavier_uniform_(block.cross_attn.out_proj.weight)
            if hasattr(block.cross_ls, 'reset_parameters'):
                block.cross_ls.reset_parameters()
            # SelfAttentionBlock has its own initialization
    
    def forward_backbone(self, x):
        """Extract features from DINOv3 backbone"""
        feats = self.backbone.get_intermediate_layers(
            x, 
            # n=range(self.config.num_layers),
            reshape=True, 
            norm=True
        )
        return feats
    
    def forward_decoder(self, features, context_embeddings, add_class_token=False):
        """
        Forward pass through decoder blocks with RoPE
        
        Args:
            features: (B, C, H, W) - backbone features
            context_embeddings: (B, num_context, embed_dim) - class or location embeddings
            add_class_token: bool - whether to add class token for classification
        """
        B, C, H, W = features.shape
        
        # Reshape features to sequence format
        features = features.flatten(2).transpose(1, 2)  # (B, H*W, C)
        
        # Generate RoPE embeddings for spatial features only (not class token)
        rope_sin, rope_cos = self.rope_embed(H=H, W=W)
        rope = (rope_sin, rope_cos)
        
        if add_class_token:
            # Add class token
            class_token = self.class_token.repeat(B, 1, 1)
            features = torch.cat([class_token, features], dim=1)  # (B, 1+H*W, C)

        # Pass through decoder blocks
        for block in self.decoder_blocks:
            features = block(features, context_embeddings, rope=rope)
        
        return features, H, W
    
    def forward(self, x, class_ids=None, bbox_coords=None):
        """
        Forward pass for multi-class segmentation
        
        Args:
            x: (B, 3, H, W) - input images
            class_ids: (B,) - class IDs for each sample in the batch, or None to use all classes
            bbox_coords: (B, 4) - bbox coordinates for location-based classification
        """
        # Extract backbone features
        features = self.forward_backbone(x)[-1]
        logits = {
            'features': features,
            'mask_logits': None,
            'class_logits': None}
        
        if class_ids is not None:
            # Semantic segmentation task
            class_ids = class_ids.unsqueeze(1)  # (B, 1)
            class_embeddings = self.class_embed(class_ids)  # (B, 1, embed_dim)
            add_class_token = False if 'semantic' in self.config.prompt_type else True
            decoded_features, H, W = self.forward_decoder(features, class_embeddings, add_class_token=add_class_token)
            if add_class_token:
                mask_features = decoded_features[:, 1:, :] # without class token
            else:
                mask_features = decoded_features
            logits['mask_logits'] = self.deconv_head(mask_features, H, W)
        
        if bbox_coords is not None:
            # Location-based classification task
            location_embeddings = self.location_embed(bbox_coords)  # (B, 1, embed_dim)
            decoded_features, H, W = self.forward_decoder(features, location_embeddings, add_class_token=True)
            mask_features = decoded_features[:, 1:, :] # without class token
            class_token = decoded_features[:, 0, :] # only class token
            logits['mask_logits'] = self.deconv_head(mask_features, H, W)
            logits['class_logits'] = self.mlp_head(class_token)
        
        if class_ids is None and bbox_coords is None:
            raise ValueError("Either class_ids or bbox_coords must be provided")
        
        return logits
