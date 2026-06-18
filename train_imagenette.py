"""MeanFlow training on a 10-class subset (Imagenette or CIFAR-10) -- the BONUS task.

This is the same MeanFlow core (model.py / meanflow.py) as the verified overfit-3,
scaled up for GPU. It is written to run on Colab/A100; I did not have a GPU in the
build sandbox, so this script is provided for you to run. Recipe defaults are the ones
that stabilized training in the overfit study (see report.html):
  * adaptive L2 weight (gamma=1)
  * high r==t fraction (p_eq=0.75) to anchor the instantaneous velocity
  * gap curriculum: cap (t-r) small early, ramp to 1.0
  * EMA weights for sampling

Examples:
  python train_imagenette.py --data cifar10 --size 32 --base 128 --steps 60000 --bs 128
  python train_imagenette.py --data /path/to/imagenette2/train --size 64 --base 128 --steps 100000
"""
import argparse, os, time
import torch
from torch.utils.data import DataLoader

from model import UNet
from meanflow import meanflow_loss, generate
from utils import save_grid, save_loss_curve, EMA


def build_dataset(name, size):
    import torchvision.transforms as T
    tf = T.Compose([
        T.Resize(size), T.CenterCrop(size), T.RandomHorizontalFlip(),
        T.ToTensor(), T.Normalize([0.5] * 3, [0.5] * 3),  # -> [-1, 1]
    ])
    if name == "cifar10":
        import torchvision
        return torchvision.datasets.CIFAR10("./data", train=True, download=True, transform=tf)
    # else: an ImageFolder (e.g. Imagenette's train/ dir = 10 class folders)
    import torchvision
    return torchvision.datasets.ImageFolder(name, transform=tf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="cifar10", help="'cifar10' or path to an ImageFolder (Imagenette train/)")
    ap.add_argument("--size", type=int, default=32)
    ap.add_argument("--base", type=int, default=128)
    ap.add_argument("--ch_mult", default="1,2,2")
    ap.add_argument("--num_res", type=int, default=2)
    ap.add_argument("--emb", type=int, default=256)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--p_eq", type=float, default=0.75)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--gap0", type=float, default=0.25)
    ap.add_argument("--gap_ramp", type=float, default=0.5, help="ramp gap->1 over this frac of steps")
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--sample_every", type=int, default=2000)
    ap.add_argument("--outdir", default="runs/imagenette")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    ds = build_dataset(args.data, args.size)
    dl = DataLoader(ds, batch_size=args.bs, shuffle=True, num_workers=4, drop_last=True, pin_memory=True)
    print(f"dataset: {len(ds)} imgs | device {device}")

    model = UNet(3, base=args.base, ch_mult=tuple(int(c) for c in args.ch_mult.split(",")),
                 num_res=args.num_res, emb_dim=args.emb).to(device)
    print(f"params {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0, betas=(0.9, 0.95))
    ema = EMA(model, decay=args.ema)

    step, losses, t0 = 0, [], time.time()
    shape = (3, args.size, args.size)
    while step < args.steps:
        for x, _ in dl:
            if step >= args.steps:
                break
            x = x.to(device)
            mg = min(1.0, args.gap0 + (1 - args.gap0) * step / max(1, int(args.gap_ramp * args.steps)))
            opt.zero_grad(set_to_none=True)
            loss, res = meanflow_loss(model, x, dist="uniform", p_eq=args.p_eq, max_gap=mg, gamma=args.gamma)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step(); ema.update(model)
            losses.append(float(res)); step += 1

            if step % 200 == 0:
                print(f"step {step}/{args.steps} | res {res:.4f} | {step/(time.time()-t0):.1f} it/s | gap {mg:.2f}")
            if step % args.sample_every == 0 or step == args.steps:
                ema.shadow.eval()
                g = torch.Generator(device).manual_seed(0)
                s1 = generate(ema.shadow, 36, shape=shape, steps=1, device=device, generator=g)
                s2 = generate(ema.shadow, 36, shape=shape, steps=2, device=device, generator=g)
                save_grid(s1.cpu(), f"{args.outdir}/samples_1step_{step}.png", nrow=6, scale=2)
                save_grid(s2.cpu(), f"{args.outdir}/samples_2step_{step}.png", nrow=6, scale=2)
                save_loss_curve(losses, f"{args.outdir}/loss_curve.png", title="MeanFlow residual MSE")
                torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "step": step},
                           f"{args.outdir}/ckpt.pt")
    print("done")


if __name__ == "__main__":
    main()
