"""Tiny pixel-space U-Net backbone for MeanFlow.

Design notes (all matter for the MeanFlow JVP):
  * The network predicts the *average velocity* u(z, r, t), so forward() takes
    BOTH times r and t (each embedded separately, then summed).
  * GroupNorm only -- NEVER BatchNorm. torch.func.jvp (forward-mode AD) does not
    compose with BatchNorm's running statistics, and batch-coupled norms make the
    per-sample JVP ill-defined.
  * No in-place activations. In-place ops can silently break functorch transforms.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _num_groups(c: int, max_groups: int = 8) -> int:
    for g in (max_groups, 4, 2, 1):
        if c % g == 0:
            return g
    return 1


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Fourier embedding for CONTINUOUS time t in [0, 1].

    We use a proper Fourier basis sin/cos(2*pi*k*t), k = 1..dim/2. This resolves
    fractional t without aliasing -- unlike the classic diffusion embedding that
    assumes integer timesteps and would alias once t is scaled up.
    """
    half = dim // 2
    k = torch.arange(1, half + 1, device=t.device, dtype=torch.float32)  # (half,)
    ang = 2.0 * math.pi * t.float().unsqueeze(-1) * k.unsqueeze(0)       # (B, half)
    emb = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)
    if emb.shape[-1] < dim:
        emb = F.pad(emb, (0, dim - emb.shape[-1]))
    return emb


class ResBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, c_emb: int):
        super().__init__()
        self.act = nn.SiLU()
        self.norm1 = nn.GroupNorm(_num_groups(c_in), c_in)
        self.conv1 = nn.Conv2d(c_in, c_out, 3, padding=1)
        self.emb_proj = nn.Linear(c_emb, c_out)
        self.norm2 = nn.GroupNorm(_num_groups(c_out), c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1)
        self.skip = nn.Conv2d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x, emb):
        h = self.conv1(self.act(self.norm1(x)))
        h = h + self.emb_proj(self.act(emb))[:, :, None, None]
        h = self.conv2(self.act(self.norm2(h)))
        return h + self.skip(x)


class Downsample(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.op = nn.Conv2d(c, c, 3, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.op = nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.op(x)


class UNet(nn.Module):
    """Small U-Net. Predicts u(z, r, t) with the same shape as z."""

    def __init__(self, in_ch=3, base=64, ch_mult=(1, 2, 2), num_res=2, emb_dim=256):
        super().__init__()
        self.base = base
        self.act = nn.SiLU()

        # Separate MLPs for r and t, summed -> a single conditioning embedding.
        self.t_mlp = nn.Sequential(nn.Linear(base, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim))
        self.r_mlp = nn.Sequential(nn.Linear(base, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim))

        self.in_conv = nn.Conv2d(in_ch, base, 3, padding=1)

        # ----- Encoder -----
        self.downs = nn.ModuleList()
        chs = [base]
        c = base
        for i, m in enumerate(ch_mult):
            c_out = base * m
            for _ in range(num_res):
                self.downs.append(ResBlock(c, c_out, emb_dim))
                c = c_out
                chs.append(c)
            if i != len(ch_mult) - 1:
                self.downs.append(Downsample(c))
                chs.append(c)

        # ----- Bottleneck -----
        self.mid1 = ResBlock(c, c, emb_dim)
        self.mid2 = ResBlock(c, c, emb_dim)

        # ----- Decoder (mirrors encoder, with skip concatenations) -----
        self.ups = nn.ModuleList()
        for i, m in reversed(list(enumerate(ch_mult))):
            c_out = base * m
            for _ in range(num_res + 1):
                self.ups.append(ResBlock(c + chs.pop(), c_out, emb_dim))
                c = c_out
            if i != 0:
                self.ups.append(Upsample(c))

        self.out_norm = nn.GroupNorm(_num_groups(c), c)
        self.out_conv = nn.Conv2d(c, in_ch, 3, padding=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, z, r, t):
        emb = self.t_mlp(timestep_embedding(t, self.base)) + self.r_mlp(
            timestep_embedding(r, self.base)
        )
        h = self.in_conv(z)
        hs = [h]
        for layer in self.downs:
            h = layer(h, emb) if isinstance(layer, ResBlock) else layer(h)
            hs.append(h)
        h = self.mid2(self.mid1(h, emb), emb)
        for layer in self.ups:
            if isinstance(layer, ResBlock):
                h = layer(torch.cat([h, hs.pop()], dim=1), emb)
            else:
                h = layer(h)
        return self.out_conv(self.act(self.out_norm(h)))


if __name__ == "__main__":
    net = UNet(base=48, ch_mult=(1, 2))
    n = sum(p.numel() for p in net.parameters())
    z = torch.randn(4, 3, 32, 32)
    r = torch.rand(4)
    t = torch.rand(4)
    print("params", f"{n/1e6:.2f}M", "| out", tuple(net(z, r, t).shape))
