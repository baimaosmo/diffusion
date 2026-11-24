import torch.nn as nn


def build_classifier(input_dim, hidden_dim):
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(hidden_dim, 32),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(32, 2),
    )


class ClusterGatedMLP(nn.Module):
    def __init__(self, input_dim=6670, num_clusters=8, hidden_dim=64):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Linear(num_clusters, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim),
            nn.Sigmoid(),
        )
        self.classifier = build_classifier(input_dim, hidden_dim)

    def forward(self, x, cluster_probs):
        return self.classifier(x * self.gate_net(cluster_probs) + x)


class PlainMLP(nn.Module):
    """Classifier used for the no-gating ablation with the same MLP head."""

    def __init__(self, input_dim=6670, hidden_dim=64):
        super().__init__()
        self.classifier = build_classifier(input_dim, hidden_dim)

    def forward(self, x):
        return self.classifier(x)
