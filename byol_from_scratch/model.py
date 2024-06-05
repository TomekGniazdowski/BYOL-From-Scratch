import copy
import torch
import torch.nn.functional as F
from torch import nn

from byol_from_scratch.transforms import SimCLR_Transforms
from byol_from_scratch.utils import get_model_n_layers


class BYOL(nn.Module):
    def __init__(
        self, 
        base_model: nn.Module, 
        image_size: int = 32, 
        projection_hidden_size: int = 4096, 
        projection_size: int = 256, 
        moving_avg_decay: float = 0.99
        ):
        
        super(BYOL, self).__init__()
        
        self.base_model = base_model
        self._base_model_output_size = self.base_model.fc.in_features
        self.image_size = image_size

        self._aug1 = SimCLR_Transforms()
        self._aug2 = SimCLR_Transforms()

        self._target_EMA_updater = EMA(tau=moving_avg_decay)

        self._encoder_online = get_model_n_layers(self.base_model, n=-1)
        self._encoder_target = self.get_target_model(self._encoder_online)

        self._projector_online = MLP(projection_hidden_size, projection_size)
        self._projector_target = MLP(projection_hidden_size, projection_size)
        self._predictor_online = MLP(projection_size, projection_size)

    @torch.no_grad()
    def update_target_network(self):
        self._target_EMA_updater.update_moving_avg(self._encoder_online, self._encoder_target)

    def get_target_model(self, model):
        target_model = copy.deepcopy(model)
        for param in target_model:
            param.requires_grad = False
        return target_model

    def forward(self, x, get_embedding=False):

        if get_embedding:
            return self._encoder_online(x).squeeze()

        x_v1, x_v2 = self._aug1(x), self._aug2(x)

        out_online_x_v1 = self._encoder_online(x_v1).squeeze()
        out_online_x_v1 = self._projector_online(out_online_x_v1)
        out_online_x_v1 = self._predictor_online(out_online_x_v1)

        out_online_x_v2 = self._encoder_online(x_v2).squeeze()
        out_online_x_v2 = self._projector_online(out_online_x_v2)
        out_online_x_v2 = self._predictor_online(out_online_x_v2)

        with torch.no_grad():
            out_target_x_v1 = self._encoder_target(x_v1).squeeze()
            out_target_x_v1 = self._projector_target(out_target_x_v1)

            out_target_x_v2 = self._encoder_target(x_v2).squeeze()
            out_target_x_v2 = self._projector_target(out_target_x_v2)

        assert not out_target_x_v1.requires_grad
        assert not out_target_x_v2.requires_grad
        loss_1 = loss_fn(out_online_x_v1, out_target_x_v2)
        loss_2 = loss_fn(out_online_x_v2, out_target_x_v1)
        loss = loss_1 + loss_2

        return loss.sum()


class MLP(nn.Module):
    def __init__(
        self, 
        hidden_size: int = 4096, 
        projection_size: int = 256
        ):
        super(MLP, self).__init__()
        self._layers = nn.Sequential(
            nn.LazyLinear(hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.LazyLinear(projection_size)
        )
    def forward(self, x):
        return self._layers(x)


def loss_fn(x: torch.Tensor, y: torch.Tensor):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)


class EMA:
    def __init__(self, tau: float):
        self.tau = tau
    
    def ema_update(self, xi_weights: nn.Module, theta_weights: nn.Module):
        return self.tau * xi_weights + (1 - self.tau) * theta_weights
    
    def update_moving_avg(self, online_model: nn.Module, target_model: nn.Module):
        for online_params, target_params in zip(online_model.parameters(), target_model.parameters()):
            online_weights, target_weights = online_params.data, target_params.data
            target_params.data = self.ema_update(target_weights, online_weights)