"""Single-channel WaPIRL ResNet-18, linear projector and directional scorer."""

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
    """WaPIRL's [2,2,2,2] backbone followed by GAP, without a learned bottleneck.

    Return raw 512-D features. Cosine normalization belongs to the loss or
    downstream scoring, so it does not change the original memory-bank EMA.
    """

    def __init__(self, embedding_dim=512):
        super().__init__()
        if embedding_dim != 512:
            raise ValueError("The WaPIRL-aligned encoder has a fixed 512-D GAP output")
        self.stem = nn.Sequential(nn.Conv2d(1, 64, 7, 2, 3, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                                  nn.MaxPool2d(3, 2, 1))
        stages, previous = [], 64
        for channels, stride in ((64, 1), (128, 2), (256, 2), (512, 2)):
            stages.append(nn.Sequential(BasicBlock(previous, channels, stride), BasicBlock(channels, channels)))
            previous = channels
        self.stages = nn.Sequential(*stages)
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                # Match WaPIRL/utils/initialization.py, including its BN bias.
                nn.init.ones_(module.weight)
                nn.init.ones_(module.bias)

    def forward(self, image):
        value = self.stages(self.stem(image))
        return F.adaptive_avg_pool2d(value, 1).flatten(1)


class Projector(nn.Module):
    def __init__(self, embedding_dim, projection_dim):
        super().__init__()
        self.net = nn.Linear(embedding_dim, projection_dim)
        nn.init.normal_(self.net.weight, 0, 0.02)
        nn.init.zeros_(self.net.bias)

    def forward(self, embedding):
        return self.net(embedding)


class DirectionalScorer(nn.Module):
    """Ordered input allows S(Q,C) != S(C,Q) with one shared image encoder."""

    def __init__(self, embedding_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(embedding_dim * 4, hidden_dim), nn.ReLU(inplace=True),
                                 nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(inplace=True),
                                 nn.Linear(hidden_dim // 2, 1))

    def forward(self, query, candidate):
        query, candidate = F.normalize(query, dim=1), F.normalize(candidate, dim=1)
        features = torch.cat([query, candidate, query * candidate, query - candidate], dim=1)
        return self.net(features).flatten().sigmoid()


class GlobalRanker(nn.Module):
    def __init__(self, embedding_dim=512, hidden_dim=256):
        super().__init__()
        self.encoder = Encoder(embedding_dim)
        self.scorer = DirectionalScorer(embedding_dim, hidden_dim)

    def forward(self, query, candidate):
        # One encoder, one pass through the concatenated images, shared parameters.
        count = query.shape[0]
        representations = self.encoder(torch.cat([query, candidate]))
        return self.scorer(representations[:count], representations[count:])
