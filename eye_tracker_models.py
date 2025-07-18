import torch
import torch.nn as nn
import numpy as np  # Required for KalmanFilter2D

# ------------------ Kalman Filter for smoothing ------------------


class KalmanFilter2D:
    """
    A 2D Kalman Filter for smoothing noisy position data.
    Tracks position (x, y) and velocity (vx, vy).
    """

    def __init__(self):
        # State vector: [pos_x, pos_y, vel_x, vel_y]
        self.x = np.array([0, 0, 0, 0], dtype=float)

        self.P = np.eye(4) * 1.0

        self.F = np.array(
            [
                [1, 0, 1, 0],  # x_new = x + vx
                [0, 1, 0, 1],  # y_new = y + vy
                [0, 0, 1, 0],  # vx_new = vx
                [0, 0, 0, 1],
            ],
            dtype=float,
        )

        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)

        self.R = np.eye(2) * 0.01

        self.Q = np.eye(4) * 0.001

    def update(self, measurement):
        """
        Updates the Kalman filter with a new 2D position measurement.
        """
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

        y = measurement - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + (K @ y)
        self.P = (np.eye(4) - K @ self.H) @ self.P

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
            nn.Conv2d(
                in_channels, 32, kernel_size=3, stride=1, padding=1
            ),  # -> (B, 32, 64, 64)
            nn.ReLU(),
            nn.MaxPool2d(2),  # -> (B, 32, 32, 32)
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # -> (B, 64, 32, 32)
            nn.ReLU(),
            nn.MaxPool2d(2),  # -> (B, 64, 16, 16)
            nn.Conv2d(
                64, out_channels, kernel_size=3, stride=1, padding=1
            ),  # -> (B, out_channels, 16, 16)
            nn.ReLU(),
        )

    def forward(self, x):
        return self.features(x)


class PatchEmbedding(nn.Module):
    """
    Converts a CNN feature map into a sequence of flattened patches suitable for a Transformer.
    """

    def __init__(self, img_size=16, patch_size=4, in_channels=64, embed_dim=128):
        super().__init__()
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x


class TransformerEncoderLayer(nn.Module):
    """
    A single layer of a Transformer Encoder, consisting of Multi-head Self-Attention and a Feed-Forward Network.
    """

    def __init__(self, embed_dim=128, num_heads=4, mlp_dim=256, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),  # Gaussian Error Linear Unit activation function
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # Input x: (B, SeqLen, EmbedDim)
        x_norm = self.norm1(x)
        attn_output, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_output

        x_norm = self.norm2(x)
        x = x + self.mlp(x_norm)
        return x


class EyeTransformer(nn.Module):
    """
    The main EyeTransformer model combining a CNN backbone with a Transformer Encoder
    for gaze estimation.
    """

    def __init__(self, embed_dim=128, depth=4, num_heads=4, mlp_dim=256):
        super().__init__()
        self.cnn_backbone = CNNBackbone(in_channels=2, out_channels=64)

        self.patch_embed = PatchEmbedding(
            img_size=16, patch_size=4, in_channels=64, embed_dim=embed_dim
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.pos_embed = nn.Parameter(torch.zeros(1, (16 // 4) ** 2 + 1, embed_dim))

        self.encoder_layers = nn.ModuleList(
            [
                TransformerEncoderLayer(embed_dim, num_heads, mlp_dim)
                for _ in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim)

        self.mlp_head = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim), nn.ReLU(), nn.Linear(mlp_dim, 2)
        )
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        """
        Forward pass through the EyeTransformer model.
        """
        B = x.shape[0]

        x = self.cnn_backbone(x)
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        x = x + self.pos_embed

        for layer in self.encoder_layers:
            x = layer(x)

        x = self.norm(x)

        cls_output = x[:, 0]  # (B, embed_dim)

        out = self.mlp_head(cls_output)  # (B, 2)
        return out
