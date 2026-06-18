"""Fixed images for the overfit-3 test, normalized to [-1, 1], shape (N,3,size,size).

Primary: three bundled scikit-image photos (astronaut, coffee, chelsea-the-cat).
Fallback: three distinct procedural patterns, so the code runs with zero datasets.
"""
import numpy as np
import torch
from PIL import Image


def _to_tensor(img_uint8: np.ndarray, size: int) -> torch.Tensor:
    im = Image.fromarray(img_uint8).convert("RGB").resize((size, size), Image.BICUBIC)
    arr = np.asarray(im).astype(np.float32) / 127.5 - 1.0  # [-1, 1]
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def _procedural(size: int):
    ys, xs = np.mgrid[0:size, 0:size].astype(np.float32) / max(size - 1, 1)
    imgs = []
    # 1) RGB gradient
    g = np.stack([xs, ys, 1 - xs], -1)
    # 2) centered disk
    cx, cy = 0.5, 0.5
    disk = (((xs - cx) ** 2 + (ys - cy) ** 2) ** 0.5 < 0.3).astype(np.float32)
    d = np.stack([disk, 1 - disk, 0.5 * np.ones_like(disk)], -1)
    # 3) checkerboard
    chk = (((xs * 6).astype(int) + (ys * 6).astype(int)) % 2).astype(np.float32)
    c = np.stack([chk, 0.3 * np.ones_like(chk), 1 - chk], -1)
    for a in (g, d, c):
        imgs.append(torch.from_numpy((a * 2 - 1).astype(np.float32)).permute(2, 0, 1).contiguous())
    return torch.stack(imgs), ["gradient", "disk", "checker"]


def get_three_images(size: int = 32):
    try:
        from skimage import data
        triples = [
            (data.astronaut(), "astronaut"),
            (data.coffee(), "coffee"),
            (data.chelsea(), "chelsea(cat)"),
        ]
        x = torch.stack([_to_tensor(im, size) for im, _ in triples])
        return x, [n for _, n in triples]
    except Exception as e:  # pragma: no cover
        print(f"[data] skimage unavailable ({e}); using procedural images")
        return _procedural(size)


if __name__ == "__main__":
    x, names = get_three_images(32)
    print(names, tuple(x.shape), float(x.min()), float(x.max()))
