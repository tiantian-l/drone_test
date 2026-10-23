"""Small azimuth CNN; LiDAR remains in [0,1], state uses fixed symlog."""
import torch
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class NavigationFeatures(BaseFeaturesExtractor):
    def __init__(self, observation_space):
        super().__init__(observation_space, features_dim=384)
        layers, _ = observation_space['lidar'].shape
        state_size = observation_space['state'].shape[0]
        self.state_net = nn.Sequential(nn.Linear(state_size, 128), nn.SiLU(),
                                       nn.Linear(128, 128), nn.SiLU())
        # Elevation layers are channels, azimuth is periodic (as in our Dreamer encoder).
        self.lidar_net = nn.Sequential(
            nn.Conv1d(layers, 32, 5, stride=2, padding=2, padding_mode='circular'), nn.SiLU(),
            nn.Conv1d(32, 64, 5, stride=2, padding=2, padding_mode='circular'), nn.SiLU(),
            nn.Conv1d(64, 64, 5, stride=2, padding=2, padding_mode='circular'), nn.SiLU(),
            nn.Flatten())
        with torch.no_grad():
            size = self.lidar_net(torch.zeros(1, *observation_space['lidar'].shape)).shape[-1]
        self.lidar_projection = nn.Sequential(nn.Linear(size, 256), nn.SiLU())

    def forward(self, observations):
        state = observations['state']
        state = state.sign() * torch.log1p(state.abs())
        return torch.cat((self.state_net(state),
                          self.lidar_projection(self.lidar_net(observations['lidar']))), dim=-1)
