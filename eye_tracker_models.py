import torch
import torch.nn as nn
import numpy as np # Required for KalmanFilter2D

# ------------------ Kalman Filter for smoothing ------------------

class KalmanFilter2D:
    """
    A 2D Kalman Filter for smoothing noisy position data.
    Tracks position (x, y) and velocity (vx, vy).
    """
    def __init__(self):
        # State vector: [pos_x, pos_y, vel_x, vel_y]
        self.x = np.array([0, 0, 0, 0], dtype=float)
        
        # Process covariance matrix: Represents uncertainty in the state estimate
        # Initialized with a relatively high uncertainty (1.0 * Identity matrix)
        self.P = np.eye(4) * 1.0
        
        # State transition matrix: Defines how the state evolves over time
        # Assumes constant velocity model (x_new = x_old + vx, y_new = y_old + vy)
        self.F = np.array([[1,0,1,0], # x_new = x + vx
                           [0,1,0,1], # y_new = y + vy
                           [0,0,1,0], # vx_new = vx
                           [0,0,0,1]], dtype=float)
        
        # Measurement matrix: Relates the state vector to the measurement vector
        # We measure only position (x, y), so it extracts first two elements of state
        self.H = np.array([[1,0,0,0],
                           [0,1,0,0]], dtype=float)
        
        # Measurement noise covariance: Represents uncertainty in the measurements
        # Small value (0.01 * Identity matrix) implies relatively clean measurements
        self.R = np.eye(2) * 0.01
        
        # Process noise covariance: Represents uncertainty due to unmodeled dynamics
        # (e.g., slight accelerations, jerks not captured by constant velocity model)
        # Small value (0.001 * Identity matrix)
        self.Q = np.eye(4) * 0.001

    def update(self, measurement):
        """
        Updates the Kalman filter with a new 2D position measurement.

        Args:
            measurement (np.array): A 1D numpy array [measured_pos_x, measured_pos_y].

        Returns:
            tuple: The smoothed (predicted) position (pos_x, pos_y).
        """
        # Predict step: Project the current state and covariance forward
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

        # Update step: Correct the predicted state using the new measurement
        y = measurement - (self.H @ self.x) # Measurement residual (difference between actual and predicted measurement)
        S = self.H @ self.P @ self.H.T + self.R # Innovation (or residual) covariance
        K = self.P @ self.H.T @ np.linalg.inv(S) # Kalman gain (optimally weights prediction vs. measurement)
        self.x = self.x + (K @ y) # Update state estimate
        self.P = (np.eye(4) - K @ self.H) @ self.P # Update estimate covariance

        # Return the smoothed position components
        return self.x[0], self.x[1]

# ------------------ CNN Backbone + Transformer Architecture ------------------

class CNNBackbone(nn.Module):
    """
    A Convolutional Neural Network (CNN) backbone to extract features from eye images.
    It processes input images (e.g., 64x64 grayscale eye images)
    and outputs a feature map (e.g., 16x16 with 64 channels).
    """
    def __init__(self, in_channels=2, out_channels=64):
        super().__init__()
        self.features = nn.Sequential(
            # Input: (B, in_channels, 64, 64)
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),  # -> (B, 32, 64, 64)
            nn.ReLU(),
            nn.MaxPool2d(2),  # -> (B, 32, 32, 32)
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), # -> (B, 64, 32, 32)
            nn.ReLU(),
            nn.MaxPool2d(2),  # -> (B, 64, 16, 16)
            nn.Conv2d(64, out_channels, kernel_size=3, stride=1, padding=1), # -> (B, out_channels, 16, 16)
            nn.ReLU()
        )
    def forward(self, x):
        return self.features(x)  # Output shape: (B, out_channels, 16, 16)

class PatchEmbedding(nn.Module):
    """
    Converts a CNN feature map into a sequence of flattened patches suitable for a Transformer.
    """
    def __init__(self, img_size=16, patch_size=4, in_channels=64, embed_dim=128):
        super().__init__()
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2 # Number of patches (e.g., (16/4)^2 = 16 patches)
        
        # Convolutional layer to project each patch into the embedding dimension
        # Stride = patch_size ensures non-overlapping patches
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # Input x: (B, in_channels, img_size, img_size) e.g., (B, 64, 16, 16)
        x = self.proj(x)  # After conv: (B, embed_dim, n_patches_sqrt, n_patches_sqrt) e.g., (B, 128, 4, 4)
        x = x.flatten(2)  # Flatten spatial dimensions: (B, embed_dim, n_patches) e.g., (B, 128, 16)
        x = x.transpose(1, 2)  # Transpose to (B, n_patches, embed_dim) for Transformer input
        return x # Output shape: (Batch_size, Sequence_Length (n_patches), Embedding_Dimension)

class TransformerEncoderLayer(nn.Module):
    """
    A single layer of a Transformer Encoder, consisting of Multi-head Self-Attention and a Feed-Forward Network.
    """
    def __init__(self, embed_dim=128, num_heads=4, mlp_dim=256, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim) # Layer normalization before attention
        # MultiheadAttention: Allows the model to jointly attend to information from different representation subspaces.
        # batch_first=True makes input/output shapes (batch_size, sequence_length, embedding_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim) # Layer normalization before MLP
        self.mlp = nn.Sequential( # Multi-Layer Perceptron (Feed-Forward Network)
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(), # Gaussian Error Linear Unit activation function
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # Input x: (B, SeqLen, EmbedDim)
        # Apply layer norm then attention with residual connection
        x_norm = self.norm1(x)
        attn_output, _ = self.attn(x_norm, x_norm, x_norm) # x_norm as query, key, value
        x = x + attn_output # Add residual connection
        
        # Apply layer norm then MLP with residual connection
        x_norm = self.norm2(x)
        x = x + self.mlp(x_norm) # Add residual connection
        return x

class EyeTransformer(nn.Module):
    """
    The main EyeTransformer model combining a CNN backbone with a Transformer Encoder
    for gaze estimation.
    """
    def __init__(self, embed_dim=128, depth=4, num_heads=4, mlp_dim=256):
        super().__init__()
        # CNN Backbone to extract visual features from eye images
        self.cnn_backbone = CNNBackbone(in_channels=2, out_channels=64)  # 2 channels for stacked grayscale left/right eye images
        
        # Patch Embedding to convert CNN features into Transformer-compatible sequences
        self.patch_embed = PatchEmbedding(img_size=16, patch_size=4, in_channels=64, embed_dim=embed_dim)
        
        # Learnable Class Token (CLS Token): Used to aggregate information from all patches for downstream prediction.
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Learnable Positional Embedding: Adds positional information to patches since Transformers are permutation-invariant.
        # (16 // 4) ** 2 is the number of patches (16 in this case). +1 for the CLS token.
        self.pos_embed = nn.Parameter(torch.zeros(1, (16 // 4) ** 2 + 1, embed_dim))
        
        # Stack of Transformer Encoder Layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, mlp_dim) for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim) # Final Layer Normalization
        
        # MLP Head: Maps the aggregated CLS token's output to the final gaze coordinates (x, y)
        self.mlp_head = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, 2)  # Output: 2D screen coordinates (x, y)
        )
        self._init_weights() # Initialize weights for CLS and positional embeddings

    def _init_weights(self):
        # Truncated normal initialization for better training stability
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        """
        Forward pass through the EyeTransformer model.

        Args:
            x (torch.Tensor): Input tensor of eye images (Batch_size, 2, 64, 64).

        Returns:
            torch.Tensor: Predicted 2D gaze coordinates (Batch_size, 2).
        """
        B = x.shape[0] # Get batch size
        
        x = self.cnn_backbone(x)  # Pass through CNN: (B, 64, 16, 16)
        x = self.patch_embed(x)    # Convert to patches: (B, n_patches, embed_dim)
        
        # Prepend the learnable CLS token to the sequence of patches
        cls_tokens = self.cls_token.expand(B, -1, -1) # Expand CLS token to match batch size
        x = torch.cat((cls_tokens, x), dim=1) # Concatenate CLS token at the beginning
        
        # Add positional embeddings to the combined sequence (CLS + patches)
        x = x + self.pos_embed
        
        # Pass the sequence through the Transformer Encoder layers
        for layer in self.encoder_layers:
            x = layer(x)
        
        x = self.norm(x) # Apply final layer normalization
        
        # Extract the output of the CLS token (first token in the sequence)
        cls_output = x[:, 0] # (B, embed_dim)
        
        # Pass CLS token output through the MLP head to get final gaze prediction
        out = self.mlp_head(cls_output) # (B, 2)
        return out