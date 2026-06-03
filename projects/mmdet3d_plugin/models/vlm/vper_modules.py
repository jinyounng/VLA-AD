import torch
import torch.nn as nn


class GateHead(nn.Module):
    """VLM hidden state → gate ∈ [0, 1] per waypoint axis.

    Determines how much of the ego-conditioned delta to apply to each
    waypoint coordinate (x, y).
    """

    def __init__(self, hidden_dim, output_dim=2):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, output_dim),
        )

    def forward(self, hidden_state):
        # hidden_state: (B, D) or (B, N, D)
        return torch.sigmoid(self.head(hidden_state))


class DeltaMLP(nn.Module):
    """Raw ego features + waypoint index → raw_delta before tanh clamping.

    Each waypoint gets a learnable embedding so the MLP can produce
    position-specific deltas from the same ego feature vector.
    """

    def __init__(self, ego_dim, hidden_dim, num_waypoints=6, output_dim=2):
        super().__init__()
        self.wp_embedding = nn.Embedding(num_waypoints, hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(ego_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, ego_features, wp_index):
        # ego_features: (B, D_ego)
        if isinstance(wp_index, int):
            wp_index = torch.full(
                (ego_features.size(0),), wp_index,
                device=ego_features.device, dtype=torch.long,
            )
        wp_emb = self.wp_embedding(wp_index)  # (B, hidden_dim)
        return self.head(torch.cat([ego_features, wp_emb], dim=-1))  # (B, 2)
