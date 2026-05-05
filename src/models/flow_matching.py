import torch
import torch.nn as nn
from src.models.unet import UNet

class FlowMatching(nn.Module):
    def __init__(self, config):
        super().__init__()
        fm_cfg = config["flow_matching"]

        self.sigma_min = fm_cfg["sigma_min"]

        self.unet = UNet(
            in_ch        = fm_cfg["in_channels"],
            base_ch      = fm_cfg["base_channels"],
            ch_mults     = fm_cfg["channel_mults"],
            num_res_blocks = fm_cfg["num_res_blocks"],
            time_emb_dim = fm_cfg["time_emb_dim"]
        )

    def forward(self, x1, noise, t):
        """
        x1    : real bitemporal pair (B, 6, H, W)
        noise : gaussian noise sample (B, 6, H, W)
        t     : timestep in [0, 1] (B,)

        conditional flow matching interpolates between
        noise at t=0 and real data at t=1 via straight line path.
        """
        # interpolate along straight line path
        t_expanded = t[:, None, None, None]
        x_t = (1 - (1 - self.sigma_min) * t_expanded) * noise + t_expanded * x1

        # target vector field (straight line direction)
        target = x1 - (1 - self.sigma_min) * noise

        # predict vector field
        pred = self.unet(x_t, t)

        return pred, target

    def compute_loss(self, x1):
        """full training step loss computation."""
        B = x1.shape[0]
        device = x1.device

        # sample noise and timesteps
        noise = torch.randn_like(x1)
        t     = torch.rand(B, device=device)

        pred, target = self.forward(x1, noise, t)

        # mean squared error on vector field
        loss = ((pred - target) ** 2).mean()
        return loss

    @torch.no_grad()
    def sample(self, shape, device, steps=50):
        """
        generate a bitemporal pair using euler integration
        of the learned vector field from t=0 to t=1.
        """
        x = torch.randn(shape, device=device)
        dt = 1.0 / steps

        for i in range(steps):
            t_val = i / steps
            t     = torch.full((shape[0],), t_val, device=device)
            v     = self.unet(x, t)
            x     = x + v * dt

        return x
