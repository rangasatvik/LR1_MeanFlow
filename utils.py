"""Small helpers: image grids, loss-curve plot, EMA."""
import copy
import numpy as np
import torch
from PIL import Image


def denorm(x: torch.Tensor) -> torch.Tensor:
    return ((x.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)


def image_grid(x: torch.Tensor, nrow: int = 8) -> Image.Image:
    x = denorm(x).permute(0, 2, 3, 1).cpu().numpy()  # (N,H,W,3)
    N, H, W, _ = x.shape
    ncol = min(nrow, N)
    nr = (N + ncol - 1) // ncol
    canvas = np.zeros((nr * H, ncol * W, 3), np.uint8)
    for i in range(N):
        r, c = divmod(i, ncol)
        canvas[r * H:(r + 1) * H, c * W:(c + 1) * W] = x[i]
    return Image.fromarray(canvas)


def save_grid(x: torch.Tensor, path: str, nrow: int = 8, scale: int = 4):
    im = image_grid(x, nrow)
    if scale > 1:
        im = im.resize((im.width * scale, im.height * scale), Image.NEAREST)
    im.save(path)
    return path


def save_loss_curve(losses, path, title="overfit-3 loss"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 3.2), dpi=130)
    ax.plot(losses, lw=0.8)
    ax.set_yscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel("loss (log)")
    ax.set_title(title)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


class EMA:
    """Exponential moving average of model parameters (improves sample quality)."""
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for s, p in zip(self.shadow.parameters(), model.parameters()):
            s.mul_(self.decay).add_(p, alpha=1 - self.decay)
        for s, p in zip(self.shadow.buffers(), model.buffers()):
            s.copy_(p)

    def state_dict(self):
        return self.shadow.state_dict()

    def load_state_dict(self, sd):
        self.shadow.load_state_dict(sd)
