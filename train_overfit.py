"""Overfit-3 trainer for MeanFlow (the required correctness test).

Runs on CPU in small wall-clock chunks: each invocation resumes from a checkpoint,
trains for `--budget` seconds (or until `--max_steps`), saves checkpoint + loss log,
and prints progress. At `--max_steps` (or with `--finalize_only`) it writes report
assets: target grid, 1-step / 2-step reproductions, random 1-step samples, loss curve,
and metrics.json.

Two couplings:
  * fixed  (default): each image is paired with a FIXED noise. Each path then has
    CONSTANT velocity (true du/dt = 0), so the MeanFlow target is stable and one-step
    generation from those noises must reconstruct the images. This cleanly tests the
    objective + JVP + sampler end-to-end.
  * random: independent noise each step -> the 3-image velocity field is extremely
    peaky and the JVP bootstrap is unstable at the (r=0,t=1) corner (see report).

Usage:
    python train_overfit.py --budget 35                 # train a chunk (auto-resume)
    python train_overfit.py --finalize_only             # (re)write assets from ckpt
"""
import argparse, json, os, time
import torch

from model import UNet
from meanflow import meanflow_loss, generate
from utils import save_grid, save_loss_curve, EMA


def build(cfg):
    return UNet(in_ch=3, base=cfg["base"], ch_mult=tuple(cfg["ch_mult"]),
                num_res=cfg["num_res"], emb_dim=cfg["emb"])


def get_images(size, path="images.pt"):
    if os.path.exists(path):
        d = torch.load(path, weights_only=False)
        if d["x"].shape[-1] == size:
            return d["x"], d["names"]
    from data import get_three_images
    x, names = get_three_images(size)
    torch.save({"x": x, "names": names}, path)
    return x, names


def finalize(model, images, names, outdir, res_hist, cfg, eps_fixed, coupling):
    model.eval()
    os.makedirs(outdir, exist_ok=True)
    shape = tuple(images.shape[1:])
    scale = max(1, 96 // images.shape[-1])

    # Reproduction: one-step from the FIXED noises must return the training images.
    repro1 = generate(model, len(images), shape=shape, steps=1, eps=eps_fixed)
    repro2 = generate(model, len(images), shape=shape, steps=2, eps=eps_fixed)
    g = torch.Generator().manual_seed(1234)
    rand1 = generate(model, 16, shape=shape, steps=1, generator=g)

    save_grid(images, os.path.join(outdir, "targets.png"), nrow=3, scale=scale)
    save_grid(repro1, os.path.join(outdir, "repro_1step.png"), nrow=3, scale=scale)
    save_grid(repro2, os.path.join(outdir, "repro_2step.png"), nrow=3, scale=scale)
    save_grid(rand1, os.path.join(outdir, "samples_random_1step.png"), nrow=8, scale=scale)
    save_loss_curve(res_hist, os.path.join(outdir, "loss_curve.png"),
                    title=f"overfit-3 residual MSE ({coupling} coupling, log)")

    per = ((repro1 - images) ** 2).mean(dim=(1, 2, 3))  # aligned i<->i
    metrics = {
        "coupling": coupling, "config": cfg, "names": names, "steps": len(res_hist),
        "final_residual_mse": round(float(res_hist[-1]), 5) if res_hist else None,
        "reproduction_mse_per_image": {names[i]: round(float(per[i]), 5) for i in range(len(images))},
        "reproduction_mse_mean": round(float(per.mean()), 6),
    }
    with open(os.path.join(outdir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print("FINALIZED:", json.dumps({k: metrics[k] for k in
          ["steps", "final_residual_mse", "reproduction_mse_mean"]}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="runs/overfit")
    ap.add_argument("--coupling", choices=["fixed", "random"], default="fixed")
    ap.add_argument("--size", type=int, default=32)
    ap.add_argument("--base", type=int, default=24)
    ap.add_argument("--ch_mult", default="1,2")
    ap.add_argument("--num_res", type=int, default=1)
    ap.add_argument("--emb", type=int, default=128)
    ap.add_argument("--tile", type=int, default=4, help="batch = 3 * tile")
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--p_eq", type=float, default=0.5)
    ap.add_argument("--dist", default="uniform")
    ap.add_argument("--gamma", type=float, default=0.0)
    ap.add_argument("--ema", type=float, default=0.995)
    ap.add_argument("--max_steps", type=int, default=900)
    ap.add_argument("--budget", type=float, default=35.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--gap0", type=float, default=0.25, help="initial cap on (t-r)")
    ap.add_argument("--gap_ramp", type=float, default=0.6, help="ramp gap to 1.0 over this frac of max_steps")
    ap.add_argument("--finalize_only", action="store_true")
    args = ap.parse_args()

    torch.set_num_threads(4)
    os.makedirs(args.outdir, exist_ok=True)
    ckpt_path = os.path.join(args.outdir, "ckpt.pt")
    cfg = dict(size=args.size, base=args.base, ch_mult=[int(c) for c in args.ch_mult.split(",")],
               num_res=args.num_res, emb=args.emb, tile=args.tile, lr=args.lr, p_eq=args.p_eq,
               dist=args.dist, gamma=args.gamma, coupling=args.coupling,
               gap0=args.gap0, gap_ramp=args.gap_ramp)

    images, names = get_images(args.size)
    xb = images.repeat(args.tile, 1, 1, 1)
    g = torch.Generator().manual_seed(args.seed + 777)
    eps_fixed = torch.randn(images.shape, generator=g)          # (3,C,H,W) fixed noises
    eps_b = eps_fixed.repeat(args.tile, 1, 1, 1) if args.coupling == "fixed" else None

    torch.manual_seed(args.seed)
    model = build(cfg)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    ema = EMA(model, decay=args.ema)
    res_hist, step = [], 0

    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, weights_only=False)
        if ck["cfg"] == cfg:
            model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
            ema.load_state_dict(ck["ema"]); res_hist = ck["res"]; step = ck["step"]
            eps_fixed = ck["eps_fixed"]; eps_b = eps_fixed.repeat(args.tile, 1, 1, 1) if args.coupling == "fixed" else None
            print(f"resumed at step {step}")
        else:
            print("config changed -> fresh start")

    if args.finalize_only:
        finalize(model, images, names, args.outdir, res_hist, cfg, eps_fixed, args.coupling)
        return

    model.train()
    t0, n_this = time.time(), 0
    while step < args.max_steps and (time.time() - t0) < args.budget:
        opt.zero_grad(set_to_none=True)
        mg = min(1.0, args.gap0 + (1 - args.gap0) * step / max(1, int(args.gap_ramp * args.max_steps)))
        loss, res = meanflow_loss(model, xb, eps=eps_b, dist=args.dist, p_eq=args.p_eq,
                                  max_gap=mg, gamma=args.gamma)
        loss.backward()
        if step == 0:
            ng = sum(p.grad is not None for p in model.parameters())
            gn = sum(p.grad.pow(2).sum() for p in model.parameters() if p.grad is not None).sqrt()
            print(f"GRAD CHECK step0: {ng}/{len(list(model.parameters()))} params grad, norm={gn.item():.3f}")
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        opt.step(); ema.update(model)
        res_hist.append(float(res)); step += 1; n_this += 1

    dt = time.time() - t0
    torch.save({"cfg": cfg, "model": model.state_dict(), "opt": opt.state_dict(),
                "ema": ema.state_dict(), "res": res_hist, "step": step, "eps_fixed": eps_fixed}, ckpt_path)
    tail = sum(res_hist[-50:]) / min(len(res_hist), 50)
    print(f"step {step}/{args.max_steps} | +{n_this} in {dt:.1f}s ({dt/max(n_this,1)*1000:.0f} ms/step) "
          f"| res(last) {res_hist[-1]:.4f} | res(tail50) {tail:.4f}")
    if step >= args.max_steps:
        finalize(model, images, names, args.outdir, res_hist, cfg, eps_fixed, args.coupling)


if __name__ == "__main__":
    main()
