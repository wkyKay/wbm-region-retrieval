"""ResNet-18 with a small-image stem, a pretraining projector and a directional scorer."""

import torch
from torch import nn
from torch.nn import functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.shortcut = nn.Identity() if stride == 1 and in_channels == channels else nn.Sequential(
            nn.Conv2d(in_channels, channels, 1, stride, bias=False), nn.BatchNorm2d(channels))

    def forward(self, image):
        value = F.relu(self.bn1(self.conv1(image)), inplace=True)
        value = self.bn2(self.conv2(value))
        return F.relu(value + self.shortcut(image), inplace=True)


class Encoder(nn.Module):
    """[2,2,2,2] residual stages; 3x3 stride-1 stem and no initial max-pooling."""

    def __init__(self, embedding_dim=256):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(1, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        stages, previous = [], 64
        for channels, stride in ((64, 1), (128, 2), (256, 2), (512, 2)):
            stages.append(nn.Sequential(BasicBlock(previous, channels, stride), BasicBlock(channels, channels)))
            previous = channels
        self.stages = nn.Sequential(*stages)
        self.embedding = nn.Linear(512, embedding_dim)
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, image):
        value = self.stages(self.stem(image))
        value = F.adaptive_avg_pool2d(value, 1).flatten(1)
        return F.normalize(self.embedding(value), dim=1)


class Projector(nn.Module):
    def __init__(self, embedding_dim, projection_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(embedding_dim, embedding_dim), nn.ReLU(inplace=True),
                                 nn.Linear(embedding_dim, projection_dim))

    def forward(self, embedding):
        return F.normalize(self.net(embedding), dim=1)


class DirectionalScorer(nn.Module):
    """Ordered input allows S(Q,C) != S(C,Q) with one shared image encoder."""

    def __init__(self, embedding_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(embedding_dim * 4, hidden_dim), nn.ReLU(inplace=True),
                                 nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(inplace=True),
                                 nn.Linear(hidden_dim // 2, 1))

    def forward(self, query, candidate):
        features = torch.cat([query, candidate, query * candidate, query - candidate], dim=1)
        return self.net(features).flatten().sigmoid()


class GlobalRanker(nn.Module):
    def __init__(self, embedding_dim=256, hidden_dim=256):
        super().__init__()
        self.encoder = Encoder(embedding_dim)
        self.scorer = DirectionalScorer(embedding_dim, hidden_dim)

    def forward(self, query, candidate):
        # One encoder, one pass through the concatenated images, shared parameters.
        count = query.shape[0]
        representations = self.encoder(torch.cat([query, candidate]))
        return self.scorer(representations[:count], representations[count:])
