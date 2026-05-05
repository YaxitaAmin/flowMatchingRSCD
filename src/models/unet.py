import torch
import torch.nn as nn
import math

# sinusoidal time embedding
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half   = self.dim // 2
        freqs  = torch.exp(
            -math.log(10000) * torch.arange(half, device=device) / (half - 1)
        )
        args = t[:, None].float() * freqs[None]
        return torch.cat([args.sin(), args.cos()], dim=-1)

# residual block with time conditioning
class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.norm1    = nn.GroupNorm(8, in_ch)
        self.conv1    = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2    = nn.GroupNorm(8, out_ch)
        self.conv2    = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_ch)
        )
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.act  = nn.SiLU()

    def forward(self, x, t_emb):
        h = self.act(self.norm1(x))
        h = self.conv1(h)
        h = h + self.time_mlp(t_emb)[:, :, None, None]
        h = self.act(self.norm2(h))
        h = self.conv2(h)
        return h + self.skip(x)

class Downsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)

class Upsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)

# full unet
class UNet(nn.Module):
    def __init__(self, in_ch, base_ch, ch_mults, num_res_blocks, time_emb_dim):
        super().__init__()

        self.time_emb = nn.Sequential(
            SinusoidalTimeEmbedding(base_ch),
            nn.Linear(base_ch, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )

        # input projection
        self.input_conv = nn.Conv2d(in_ch, base_ch, 3, padding=1)

        # encoder
        self.downs       = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        ch_now = base_ch
        self.enc_chs = [base_ch]

        for mult in ch_mults:
            out_ch = base_ch * mult
            for _ in range(num_res_blocks):
                self.downs.append(ResBlock(ch_now, out_ch, time_emb_dim))
                self.enc_chs.append(out_ch)
                ch_now = out_ch
            self.downsamples.append(Downsample(ch_now))

        # bottleneck
        self.mid1 = ResBlock(ch_now, ch_now, time_emb_dim)
        self.mid2 = ResBlock(ch_now, ch_now, time_emb_dim)

        # decoder
        self.ups       = nn.ModuleList()
        self.upsamples = nn.ModuleList()

        for mult in reversed(ch_mults):
            out_ch = base_ch * mult
            self.upsamples.append(Upsample(ch_now))
            for _ in range(num_res_blocks):
                skip_ch = self.enc_chs.pop()
                self.ups.append(ResBlock(ch_now + skip_ch, out_ch, time_emb_dim))
                ch_now = out_ch

        # output projection
        self.out_norm = nn.GroupNorm(8, ch_now)
        self.out_conv = nn.Conv2d(ch_now, in_ch, 1)
        self.act      = nn.SiLU()

    def forward(self, x, t):
        t_emb = self.time_emb(t)
        h     = self.input_conv(x)

        # encoder
        skips = [h]
        down_iter = iter(self.downs)
        for i, ds in enumerate(self.downsamples):
            for _ in range(len(self.downs) // len(self.downsamples)):
                h = next(down_iter)(h, t_emb)
                skips.append(h)
            h = ds(h)

        # bottleneck
        h = self.mid1(h, t_emb)
        h = self.mid2(h, t_emb)

        # decoder
        up_iter = iter(self.ups)
        for us in self.upsamples:
            h = us(h)
            for _ in range(len(self.ups) // len(self.upsamples)):
                skip = skips.pop()
                h    = torch.cat([h, skip], dim=1)
                h    = next(up_iter)(h, t_emb)

        h = self.act(self.out_norm(h))
        return self.out_conv(h)
